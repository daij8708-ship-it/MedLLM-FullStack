"""Frozen new-document ablation. Saves raw and fallback outputs separately on F:."""
import hashlib,json,os,re,time
from pathlib import Path
from corpus import DATA,HOME
from rag_core import RAGService

CASES=[
 {'id':'oral_1','doc':'oral','q':'成人每天刷牙几次，牙膏需要含什么成分？','groups':[['两次','2次','twice'],['含氟','fluoride']]},
 {'id':'oral_2','doc':'oral','q':'已经戴全口假牙的成年人还要定期看牙医吗，频率如何？','groups':[['每年','一年','once a year'],['一次','1次','once']]},
 {'id':'food_1','doc':'food','q':'做饭时为什么不应把生肉和即食食品放在同一块砧板上？','groups':[['交叉污染','污染','cross-contamin','germs']]},
 {'id':'food_2','doc':'food','q':'易腐食品放在室温下多久应冷藏，超过32摄氏度时呢？','groups':[['两小时','2小时','2 hours'],['一小时','1小时','1 hour']]},
 {'id':'pred_1','doc':'prediabetes','q':'超重且有糖尿病前期的人，减重参考比例和每周活动量是多少？','groups':[['5%','百分之五'],['7%','百分之七'],['150','一百五十']]},
 {'id':'pred_2','doc':'prediabetes','q':'怀疑有糖尿病前期，可以通过什么检查了解情况？','groups':[['血糖','blood sugar']]},
 {'id':'co_1','doc':'co','q':'便携发电机在车库里开着门使用可以吗？','groups':[['不要','不能','不可以','never'],['车库','garage']]},
 {'id':'co_2','doc':'co','q':'备用发电机在室外使用，应离门窗和通风口多远？','groups':[['20英尺','20 feet','6米']]},
 {'id':'heat_1','doc':'heat','q':'天热时户外活动安排在什么时段更合适？','groups':[['凉爽','最热','coolest','早','晚']]},
 {'id':'heat_2','doc':'heat','q':'高温天防护需要注意哪些基本方面？','groups':[['凉爽','cool'],['水分','补水','hydrated']]},
 {'id':'smoke_1','doc':'smoke','q':'二手烟存在一个安全的暴露量吗？','groups':[['不存在','没有','无安全','no safe']]},
 {'id':'smoke_2','doc':'smoke','q':'在室内吸烟时开窗并使用风扇，能完全保护其他人吗？','groups':[['不能','无法','不可以','not'],['无烟','smoke-free','smokefree','不吸烟','不要']]},
 {'id':'abs_1','doc':None,'q':'帮我写一个Python快速排序函数。','groups':[]},
 {'id':'abs_2','doc':None,'q':'明天买哪只股票能赚最多钱？','groups':[]},
 {'id':'abs_3','doc':None,'q':'知识库推荐什么品牌的牙膏，价格多少？','groups':[]},
 {'id':'abs_4','doc':None,'q':'请给我阿莫西林的具体剂量。','groups':[]},
]
FOLLOWUPS=[
 {'doc':'oral','first':'成年人平时应该怎样保护牙齿？','follow':'那一年要检查几次？'},
 {'doc':'food','first':'易腐食品在常温下怎样保存？','follow':'那超过32摄氏度呢？'},
 {'doc':'co','first':'便携发电机应该放在哪里使用？','follow':'那离窗户多远？'},
 {'doc':'smoke','first':'家庭应该怎样防止二手烟暴露？','follow':'那只开窗够不够？'},
]


def covered(answer,groups):
    answer=answer.replace(' ','').lower()
    return bool(groups) and all(any(x.replace(' ','').lower() in answer for x in group) for group in groups)


def run():
    DATA.mkdir(parents=True,exist_ok=True)
    frozen=DATA/'evaluation-cases.json'
    content=json.dumps({'cases':CASES,'followups':FOLLOWUPS},ensure_ascii=False,indent=2)
    if frozen.exists() and frozen.read_text(encoding='utf-8')!=content:raise RuntimeError('Frozen case file changed')
    frozen.write_text(content,encoding='utf-8')
    metadata={'case_sha256':hashlib.sha256(content.encode()).hexdigest(),'corpus_sha256':hashlib.sha256((DATA/'chunks.json').read_bytes()).hexdigest(),
      'note':'Pilot new-document benchmark. All six documents excluded from v4 training; Chinese editorial summaries belong to retrieval corpus, not SFT. Keyword checks are not clinical correctness.',
      'modes':['base','sft','rag','sft_rag'],'max_new_tokens':192,'threshold':.45}
    (DATA/'evaluation-manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    service=RAGService();retriever=service.retrieval_ready();retrieval=[];records=[]
    for case in CASES:
        hits=retriever.search(case['q'],3)
        retrieval.append({'case':case['id'],'gold_doc':case['doc'],'hits':[{'id':h['id'],'doc':h['metadata']['doc_id'],'score':h['score']} for h in hits]})
    for mode in metadata['modes']:
        for case in CASES:
            print('RUN',mode,case['id'],flush=True)
            result=service.ask(case['q'],mode)
            raw=result.get('raw_model_text',result['text']);raw_cites=result.get('raw_model_citations',result.get('citations',[]))
            row={'case':case,'mode':mode,'result':result,
                 'raw_keyword_pass':covered(raw,case['groups']), 'delivered_keyword_pass':covered(result['text'],case['groups']),
                 'raw_valid_citations':bool(raw_cites) and all(c['metadata']['doc_id']==case['doc'] for c in raw_cites),
                 'delivered_valid_citations':bool(result.get('citations')) and all(c['metadata']['doc_id']==case['doc'] for c in result['citations']),
                 'abstention_pass':case['doc'] is None and (result.get('abstained',False) or bool(re.search(r'无法|不能|资料不足|没有找到|不提供|对不起',raw)))}
            records.append(row)
            (DATA/'ablation-results.json').write_text(json.dumps({'manifest':metadata,'retrieval':retrieval,'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
            print('DONE',mode,case['id'],'raw',row['raw_keyword_pass'],'fallback',result.get('delivery_mode'),flush=True)
    follow=[]
    for case in FOLLOWUPS:
        first=service.ask(case['first'],'sft_rag')
        second=service.ask(case['follow'],'sft_rag',[{'question':case['first'],'answer':first['text']}])
        hits=second.get('raw_ranked_hits',second['retrieved'])
        follow.append({'case':case,'first':first,'second':second,'rewrite_applied':second['rewritten'],
                       'doc_hit_at_3':any(h['metadata']['doc_id']==case['doc'] for h in hits)})
        print('FOLLOWUP',case['doc'],follow[-1]['doc_hit_at_3'],flush=True)
    summaries=[]
    for mode in metadata['modes']:
        rows=[r for r in records if r['mode']==mode];answerable=[r for r in rows if r['case']['doc']]
        summaries.append({'mode':mode,'n':len(rows),'answerable':12,'unanswerable':4,
             'keyword_pass':sum(r['raw_keyword_pass'] for r in answerable),
             'delivered_keyword_pass':sum(r['delivered_keyword_pass'] for r in answerable),
             'valid_citation_answers':sum(r['raw_valid_citations'] for r in answerable),
             'delivered_valid_citation_answers':sum(r['delivered_valid_citations'] for r in answerable),
             'abstention_pass':sum(r['abstention_pass'] for r in rows),
             'fallbacks':sum(r['result'].get('delivery_mode')=='evidence_excerpt_fallback' for r in rows)})
    relevant=[r for r in retrieval if r['gold_doc']]
    summary={'state':'complete','scope':'6份未用于微调的新文档，67段；12道可答题、4道无答案题、4组两轮追问。小规模研究评测。',
      'modes':summaries,'retrieval':{'n':12,'recall_at_1':sum(r['hits'][0]['doc']==r['gold_doc'] for r in relevant)/12,
      'recall_at_3':sum(any(h['doc']==r['gold_doc'] for h in r['hits']) for r in relevant)/12},
      'followups':{'n':4,'doc_hit_at_3':sum(r['doc_hit_at_3'] for r in follow),'rewrite_applied':sum(r['rewrite_applied'] for r in follow)},
      'limitations':'表中关键点和引用统计针对模型原始回答；资料摘录保底单独统计，不能算作微调或模型生成能力。原始输出仍可能错误。非临床专家评测。',
      'manifest':metadata}
    (DATA/'evaluation-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (DATA/'ablation-results.json').write_text(json.dumps({'manifest':metadata,'retrieval':retrieval,'records':records,'followups':follow,'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__':run()
