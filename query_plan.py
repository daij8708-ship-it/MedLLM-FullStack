"""Explicit query planning and limited evidence requirements, not an LLM agent."""
import re
import os


def split_questions(question):
    parts=re.split(r'以及|同时(?:请)?|另外(?:请)?|[；;]',question)
    parts=[re.sub(r'^(?:请)?(?:分别)?(?:说明|回答|介绍)?','',p).strip(' ，,。？?') for p in parts]
    return parts if 1<len(parts)<=3 and all(len(p)>=4 for p in parts) else [question]


def personal_diagnosis(question):
    return bool(re.search(r'(?:我|本人).{0,40}(?:是不是|是否|会不会).{0,20}(?:病|癌|感染|中毒)|(?:我|本人).{0,20}(?:得了|患有)|诊断我|给我确诊',question))


def missing_detail(question,chunks):
    text='\n'.join(c['text'] for c in chunks)
    match=re.search(r'(?:多少|几)(分钟|小时|微克|毫克|千克|公斤|毫升|厘米|毫米|克|升)',question)
    if match and match.group(1) not in text:
        return '现有资料没有提供问题所要求的'+match.group(1)+'信息，无法据此回答。'
    # Inspectable topic predicates. Presence is necessary, never sufficient for support.
    for predicate in ['充电','保修','电池寿命','生产日期','转速']:
        if predicate in question and predicate not in text:
            return '现有资料没有提供'+predicate+'方面的信息，无法据此回答。'
    return None


def evidence_requirements(question,chunks):
    """Necessary evidence facets; independent of model selection and not sufficient proof."""
    baseline=missing_detail(question,chunks)
    if baseline or os.environ.get('RAG_EVIDENCE_CHECKS')=='legacy':return baseline
    text='\n'.join(c['text'] for c in chunks)
    checks=[(r'更换|换新',r'更换|换新|replace','更换周期'),
            (r'功率|瓦数',r'功率|瓦|\bwatt','设备功率'),
            (r'植物|盆栽',r'植物|盆栽|\bplant','植物防护'),
            (r'长度|多长(?!时间)',r'厘米|毫米|英尺|米|\bcm\b|\bfeet\b','长度')]
    for query_pattern,evidence_pattern,label in checks:
        if re.search(query_pattern,question,re.I) and not re.search(evidence_pattern,text,re.I):
            return '现有资料缺少'+label+'方面的直接证据，无法据此回答。'
    match=re.search(r'(?:多少|几)(?:个)?(秒钟|分钟|小时|厘米|毫米|毫升|摄氏度|个月|天|周|年|克|升|度)',question)
    if match:
        unit=match.group(1)
        aliases={'秒钟':r'秒|second','分钟':r'分钟|minute','小时':r'小时|hour','厘米':r'厘米|\bcm\b',
                 '毫米':r'毫米|\bmm\b','毫升':r'毫升|\bml\b','摄氏度':r'摄氏度|°C|℃',
                 '个月':r'月|month','天':r'天|day','周':r'周|week','年':r'年|year','克':r'克|gram',
                 '升':r'升|litre|liter','度':r'度|°|degree'}
        if not re.search(aliases[unit],text,re.I):return '现有资料没有提供问题要求的'+unit+'信息，无法据此回答。'
    return None


def evidence_alignment(question,chunks):
    """Check narrowly parsed entity/relation pairs in the same evidence sentence."""
    text='\n'.join(c['text'] for c in chunks)
    # Nested component requests: X's Y length cannot be supported by X's clearance.
    component=re.search(r'的([\u4e00-\u9fff]{2,12}?)(?:应该|应当|需要|应|要|需)?(?:有)?(?:多长(?!时间)|多宽|多高|多粗|长度|宽度|高度|直径)',question)
    if component:
        entity=component.group(1).strip('的')
        if entity not in text:return '现有资料未提供“'+entity+'”这一对象的尺寸证据，不能用其他对象的距离代替。'
        sentences=re.split(r'[。！？\n]',text)
        if not any(entity in s and re.search(r'长|宽|高|直径|尺寸',s) and re.search(r'\d|[一二三四五六七八九十]',s) for s in sentences):
            return '现有资料没有将“'+entity+'”与所问尺寸直接对应，无法据此回答。'
    maintenance=(re.search(r'(?:多久|多长时间|每隔|几天|几周|几个月).{0,8}(?:清洗|清洁|维护|保养)',question)
                 or re.search(r'(?:清洗|清洁|维护|保养).{0,8}(?:多久|频率|周期|几天|几周)',question))
    if maintenance:
        sentences=re.split(r'[。！？\n]',text)
        prefix=re.split(r'应该|应当|需要|应|要|需|多久|多长时间|每隔',question,maxsplit=1)[0]
        entity=re.split(r'时|对于|关于|的|，',prefix)[-1].strip('请问')
        aliases=[entity]
        if '过滤器' in entity or '净化器' in entity:aliases=['过滤器','净化器']
        require_entity=2<=len(entity)<=12 and entity not in ['设备','这些设备','材料','物品','它们']
        if not any((not require_entity or any(word in s for word in aliases)) and
                   re.search(r'清洗|清洁|维护|保养|clean|maintain',s,re.I) and
                   re.search(r'每|间隔|周期|小时|天|周|月|年|every|hour|day|week|month',s,re.I) for s in sentences):
            return '现有资料没有给出清洗或维护周期，不能用一般防护建议替代。'
    return None


def combine_results(question,parts,results):
    citations=[];texts=[];retrieved=[]
    for part,result in zip(parts,results):
        mapping={}
        for c in result.get('citations',[]):
            existing=next((v for v in citations if v['id']==c['id']),None)
            if existing is None:
                existing={**c,'label':len(citations)+1};citations.append(existing)
            mapping[c['label']]=existing['label']
        text=re.sub(r'\[(\d+)\]',lambda m:'['+str(mapping.get(int(m.group(1)),int(m.group(1))))+']',result['text'])
        texts.append(part+'：\n'+text)
        for c in result.get('retrieved',[]):
            if c['id'] not in {r['id'] for r in retrieved}:retrieved.append(c)
    return {'model':results[0]['model'],'text':'\n\n'.join(texts),'citations':citations,'retrieved':retrieved,
            'delivery_mode':'multi_part_evidence','subquestions':parts,'subresults':results,
            'abstained':all(r.get('abstained',False) for r in results),'partial_abstention':any(r.get('abstained',False) for r in results),
            'warnings':['问题已拆分逐项检索；每项分别展示答案或资料不足说明。']+
                       (['部分子问题证据不足，请以各项说明为准。'] if any(r.get('abstained',False) for r in results) else []),
            'seconds':round(sum(r['seconds'] for r in results),2),'tokens':sum(r['tokens'] for r in results),
            'invalid_citations':[],'truncated':any(r['truncated'] for r in results),
            'cancelled':any(r['cancelled'] for r in results),'rewritten':False,'retrieval_query':question}
