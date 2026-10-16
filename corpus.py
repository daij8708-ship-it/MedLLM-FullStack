"""可复现的 CDC 语料；原始 HTML 与切好的块一起留在工作目录下（见 workspace.py）。"""
import hashlib
import json
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


def extract_sections(data, filename):
    """Return page/section text; scanned PDFs without text are explicitly rejected."""
    suffix=Path(filename).suffix.lower()
    if suffix=='.pdf':
        from pypdf import PdfReader
        import io
        reader=PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError('暂不支持加密PDF。')
        if len(reader.pages)>200:raise ValueError('PDF最多支持200页，请拆分文档。')
        result=[{'section':f'第{i+1}页','page':i+1,'text':p.extract_text() or ''} for i,p in enumerate(reader.pages)]
        if not any(x['text'].strip() for x in result):
            raise ValueError('PDF未提取到文字；当前不支持扫描件OCR。')
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
