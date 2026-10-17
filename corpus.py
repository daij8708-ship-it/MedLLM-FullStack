"""可复现的 CDC 语料；原始 HTML 与切好的块一起留在工作目录下（见 workspace.py）。"""
import hashlib
import io
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from workspace import HOME, RAG as DATA
SOURCES=[
 dict(id='food',title='食物安全：清洁、分开、煮熟与冷藏',url='https://www.cdc.gov/food-safety/prevention/',
      summary='预防食源性疾病的四个步骤是清洁、生熟分开、彻底煮熟和及时冷藏。处理生肉与即食食品应使用不同的砧板，以免生肉上的病菌污染即食食品。易腐食品通常应在两小时内冷藏；环境温度超过90°F（约32°C）时应在一小时内冷藏。冰箱冷藏室应保持40°F（约4°C）或以下。'),
 dict(id='oral',title='成人口腔健康：刷牙、牙线与定期检查',url='https://www.cdc.gov/oral-health/prevention/oral-health-tips-for-adults.html',
      summary='成人应每天使用含氟牙膏认真刷牙两次，并使用牙线清洁牙缝以去除牙菌斑。即使戴假牙或已经失去所有牙齿，也应至少每年看一次牙医。尽量减少添加糖的食品和饮料。'),
 dict(id='prediabetes',title='糖尿病前期：降低2型糖尿病风险',url='https://www.cdc.gov/diabetes/prevention-type-2/prediabetes-prevent-type-2.html',
      summary='糖尿病前期可通过血糖检查发现，应咨询医生是否需要检查。对超重且有糖尿病前期的人，减轻约5%到7%的体重并规律活动可降低发展为2型糖尿病的风险；活动参考为每周至少150分钟快走或类似活动。这不是个人诊断或用药建议。'),
 dict(id='co',title='一氧化碳与便携发电机安全',url='https://www.cdc.gov/carbon-monoxide/about/index.html',
      summary='一氧化碳是无色无味的有毒气体。不要在室内、地下室或车库使用便携发电机，即使门窗敞开也不可以。发电机只应在室外、远离门窗和通风口至少20英尺（约6米）的位置使用。'),
 dict(id='heat',title='高温健康：保持凉爽与补水',url='https://www.cdc.gov/heat-health/about/index.html',
      summary='高温天气应保持凉爽、保持水分并了解热相关疾病的症状。不要只依赖电风扇来降温；应到有空调的地方避暑。安排户外活动时尽量避开最热时段，并经常休息。具体饮水限制应咨询医疗专业人员。'),
 dict(id='smoke',title='家庭二手烟暴露防护',url='https://www.cdc.gov/tobacco/secondhand-smoke/home.html',
      summary='二手烟暴露不存在安全量。开窗、使用风扇或空调、空气净化器以及限制在某个房间吸烟，都不能消除二手烟。保护家人应让室内和车辆完全无烟；不要让任何人在家中或车内吸烟。'),
]


PDF_OCR_MODES=('off','rapidocr')
_OCR_ENGINE=None


def pdf_ocr_mode():
    """显式开关：off 保持"没有文本层就报错"的旧行为，rapidocr 才走 OCR。"""
    mode=os.environ.get('RAG_PDF_OCR','off')
    if mode not in PDF_OCR_MODES:
        raise ValueError('RAG_PDF_OCR 只支持 '+' / '.join(PDF_OCR_MODES)+'。')
    return mode


def _ocr_engine():
    """惰性加载；权重随 rapidocr-onnxruntime 一起安装，无需联网。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE=RapidOCR()
    return _OCR_ENGINE


def ocr_pdf_pages(data,page_numbers,dpi=200):
    """只对没有文本层的页做 OCR，返回与 page_numbers 等长的文本列表。"""
    try:
        import pypdfium2 as pdfium
        engine=_ocr_engine()
    except ImportError as exc:
        raise ValueError('这份 PDF 没有文本层，需要 OCR：请安装 requirements-ocr.txt'
                         '（rapidocr-onnxruntime + pypdfium2），并设置 RAG_PDF_OCR=rapidocr。') from exc
    document=pdfium.PdfDocument(io.BytesIO(data))
    texts=[]
    for number in page_numbers:
        image=document[number-1].render(scale=dpi/72).to_pil().convert('RGB')
        lines,_=engine(image)
        texts.append('\n'.join(line[1] for line in lines if len(line)>1).strip() if lines else '')
    return texts


def pages_needing_ocr(pages):
    """返回没有文本层的页码：有文本的页永远走 pypdf，不被 OCR 覆盖。"""
    return [page['page'] for page in pages if not page['text'].strip()]


def _is_url_fragment(s):
    return bool(re.search(r'(?:https?://|www\.)$',s,re.I) or s.endswith('/'))


def _join_separator(prev,line):
    """拉丁字母/数字之间补一个空格；中文与 URL 续行不补。"""
    if not line or not prev:return ''
    if re.match(r'^[)\]}.,;:!?]',line):return ''
    if line.startswith('(') and re.search(r'[A-Za-z0-9]$',prev):return ' '
    if _is_url_fragment(prev):return ''
    if re.match(r'[A-Za-z0-9]',line) and re.search(r'[A-Za-z0-9,;:)\].!?]$',prev):return ' '
    if re.match(r'[A-Za-z0-9]',line) and re.search(r'[\u4e00-\u9fff]$',prev):return ' '
    return ''


def _ends_sentence(s):
    return bool(re.search(r'[。！？!?;；:：]$',s) or (s.endswith('.') and not _is_url_fragment(s)))


def _starts_new_block(line):
    """判断一行是否开启新块，以及该块是否不允许下一行并入。

    ``starts_new_block``：项目符号、编号条目、脚注、表格/图表行。
    ``keep_isolated``：此外还包括页码、表格行、标题、作者行——下一行不得并进来。
    """
    s=line.strip()
    if not s:return True,True
    if re.fullmatch(r'\d{1,3}',s):return True,True
    if '\t' in s or re.search(r'\s{2,}',s) or '|' in s:return True,True
    if len(s)<=40 and re.search(r'[$%]',s):return True,True
    if re.fullmatch(r'[\d.,]+',s):return True,True
    if re.match(r'^(?:[•·▪◦]|\d+[.)、]\s|[一二三四五六七八九十]+、|[*†‡])',s):return True,False
    if len(s)<=40 and not re.search(r'[。！？!?;；:：]$',s) and s.upper()==s and re.search(r'[A-Z]',s):return True,True
    if re.match(r'^[A-Z]\.\s',s):return True,True
    return False,False


def reflow_text(text):
    """切块前还原 PDF/OCR 的硬换行。

    PDF 与 OCR 文本是按版面逐行给出的：一句话在列宽处被切断，单词可能带行尾连字符。
    这里把同一段内的行接回去，``evidence_sentences`` 才能看到完整句子。
    标题、项目符号、表格/图表行和页码保留为块边界；HTML/TXT/MD 不经过本函数。
    """
    text=text.replace('\r\n','\n').replace('\r','\n')
    paragraphs=re.split(r'\n\s*\n',text)
    out=[]
    for para in paragraphs:
        lines=[l.strip() for l in para.splitlines() if l.strip()]
        blocks=[];isolated=[]
        for line in lines:
            new,iso=_starts_new_block(line)
            if not blocks or new or isolated[-1] or _ends_sentence(blocks[-1]):
                blocks.append(line);isolated.append(iso)
            elif blocks[-1].endswith('-') and re.match(r'[A-Za-z0-9]',line):
                blocks[-1]=blocks[-1][:-1]+line
            else:
                blocks[-1]=blocks[-1]+_join_separator(blocks[-1],line)+line
        if blocks:out.append('\n'.join(blocks))
    return '\n\n'.join(out)


def extract_sections(data, filename):
    """Return page/section text; scanned pages go through OCR only when RAG_PDF_OCR enables it."""
    suffix=Path(filename).suffix.lower()
    if suffix=='.pdf':
        from pypdf import PdfReader
        reader=PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError('暂不支持加密PDF。')
        if len(reader.pages)>200:raise ValueError('PDF最多支持200页，请拆分文档。')
        result=[{'section':f'第{i+1}页','page':i+1,'text':reflow_text((p.extract_text() or '').strip())} for i,p in enumerate(reader.pages)]
        blank=pages_needing_ocr(result)
        if blank and pdf_ocr_mode()=='rapidocr':
            by_number={page['page']:page for page in result}
            for number,text in zip(blank,ocr_pdf_pages(data,blank)):
                by_number[number]['text']=reflow_text(text);by_number[number]['ocr']=True
        if not any(x['text'].strip() for x in result):
            if pdf_ocr_mode()=='rapidocr':
                raise ValueError('PDF未提取到文字，OCR 也没有识别出内容；请检查扫描质量或改用文字版 PDF。')
            raise ValueError('PDF未提取到文字（扫描件）；可安装 requirements-ocr.txt 并设置 RAG_PDF_OCR=rapidocr 后再上传。')
        return result
    text=data.decode('utf-8-sig',errors='strict')
    if suffix in ('.html','.htm'):
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(text,'html.parser')
        main=soup.find('main') or soup.find('article') or soup.body or soup
        for node in main.select('script,style,nav,footer,header,aside'):
            node.decompose()
        sections=[];heading='正文';buffer=[]
        for node in main.find_all(['h1','h2','h3','p','li']):
            # Avoid nested list duplication.
            if node.name=='p' and node.find_parent('li'):
                continue
            content=node.get_text(' ',strip=True)
            if not content:continue
            if node.name.startswith('h'):
                if buffer:sections.append({'section':heading,'page':0,'text':'\n'.join(buffer)})
                heading=content;buffer=[]
            else:buffer.append(content)
        if buffer:sections.append({'section':heading,'page':0,'text':'\n'.join(buffer)})
        return sections
    if suffix not in ('.txt','.md'):
        raise ValueError('仅支持UTF-8 TXT、Markdown、HTML及文字版PDF。')
    sections=[];heading='正文';buffer=[]
    for line in text.splitlines():
        if suffix=='.md' and re.match(r'^#{1,6}\s+',line):
            if buffer:sections.append({'section':heading,'page':0,'text':'\n'.join(buffer)})
            heading=re.sub(r'^#+\s+','',line).strip();buffer=[]
        else:buffer.append(line)
    if buffer:sections.append({'section':heading,'page':0,'text':'\n'.join(buffer)})
    return sections


def make_chunks(doc,sections,size=700,overlap=80):
    chunks=[]
    for sec in sections:
        text=re.sub(r'\n{3,}','\n\n',sec['text']).strip()
        if len(text)<25:continue
        for start in range(0,len(text),size-overlap):
            part=text[start:start+size].strip()
            if len(part)<25:continue
            ident=hashlib.sha256(f"{doc['id']}:{sec['page']}:{sec['section']}:{start}:{part}".encode()).hexdigest()[:24]
            chunks.append({'id':ident,'text':part,'metadata':{
                'doc_id':doc['id'],'title':doc['title'],'source':doc['url'],
                'section':sec['section'],'page':sec['page'],'start':start,'end':start+len(part),
                'updated':doc.get('updated','unknown'),'fetched':doc.get('fetched','unknown'),
                'license':doc.get('license','user-provided'),'kind':sec.get('kind','original'),
                'ocr':bool(sec.get('ocr',False)),
                'raw_sha256':doc.get('raw_sha256','')}})
    return chunks


def fetch_corpus():
    from bs4 import BeautifulSoup
    DATA.mkdir(parents=True,exist_ok=True);(DATA/'raw').mkdir(exist_ok=True)
    manifest=[];all_chunks=[]
    for item in SOURCES:
        target=DATA/'raw'/(item['id']+'.html')
        if not target.exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(item['url'],timeout=30) as response:
                        target.write_bytes(response.read());break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2)
        raw=target.read_bytes();soup=BeautifulSoup(raw,'html.parser')
        def meta(name):
            node=soup.find('meta',attrs={'name':name}) or soup.find('meta',attrs={'property':name})
            return node.get('content','unknown') if node else 'unknown'
        updated=meta('cdc:last_updated')
        if updated=='unknown':updated=meta('og:updated_time')
        doc={**item,'updated':updated,'published':meta('cdc:first_published'),'last_published':meta('cdc:last_published'),
             'update_field':'cdc:last_updated or og:updated_time; source-declared, not fetch date',
             'fetched':'2026-10-16','license':'CDC agency text; public-domain policy subject to page exceptions',
             'license_url':'https://www.cdc.gov/other/agencymaterials.html','raw_file':str(target),
             'raw_sha256':hashlib.sha256(raw).hexdigest(),'summary_note':'Chinese editorial summary, not official translation or clinician review.'}
        original=extract_sections(raw,target.name)
        if not original:raise ValueError('官方文档未解析出正文：'+item['url'])
        sections=[{'section':'中文导读（人工整理，非官方译文）','page':0,'text':item['summary'],'kind':'editorial_summary'}]+original
        chunks=make_chunks(doc,sections)
        doc['sections']=sections;doc['chunk_count']=len(chunks);manifest.append(doc);all_chunks.extend(chunks)
        print(item['id'],len(original),'sections',len(chunks),'chunks',flush=True)
    (DATA/'documents.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (DATA/'chunks.json').write_text(json.dumps(all_chunks,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CORPUS',len(manifest),'documents',len(all_chunks),'chunks',flush=True)


if __name__=='__main__':fetch_corpus()
