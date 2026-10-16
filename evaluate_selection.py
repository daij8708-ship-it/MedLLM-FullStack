"""Regression for extractive pipeline; never overwrites original generation ablation."""
import json,hashlib
from pathlib import Path
from rag_core import RAGService
from evaluate_rag import CASES,FOLLOWUPS,covered
from workspace import RAG as DATA
EXTRA=[
 {'q':'知识库有没有建议每次刷牙使用多少克牙膏？','groups':[]},
 {'q':'CDC这些资料有没有给出我所在城市明天的最高温度？','groups':[]},
 {'q':'做饭切生肉用的砧板，能不清洗就接着切直接入口的食物吗？','groups':[['污染']]},
 {'q':'室内吸烟后用空气净化器就能消除所有二手烟风险吗？','groups':[['不能','无法','不可以']]},
]
s=RAGService();rows=[];follow=[]
manifest={'type':'existing-case regression plus newly frozen 4-case challenge; not independent clinical evaluation',
 'extra_sha256':hashlib.sha256(json.dumps(EXTRA,ensure_ascii=False).encode()).hexdigest(),'answer_style':s.answer_style}
(DATA/'selection-challenge-frozen.json').write_text(json.dumps(EXTRA,ensure_ascii=False,indent=2),encoding='utf-8')
for mode in ['rag','sft_rag']:
 for case in CASES:
  r=s.ask(case['q'],mode)
  exact=all(any(line.rsplit(' [',1)[0] in c['text'] for c in r.get('citations',[])) for line in r['text'].splitlines() if line.strip()) if r.get('citations') else False
  rows.append({'case':case,'mode':mode,'result':r,'keyword_pass':covered(r['text'],case['groups']),'exact_source':exact})
  print(mode,case['id'],r.get('delivery_mode'),rows[-1]['keyword_pass'],flush=True)
for case in EXTRA:
 r=s.ask(case['q'],'rag');rows.append({'case':case,'mode':'challenge','result':r,'keyword_pass':covered(r['text'],case['groups'])})
 print('challenge',case['q'],r['text'],flush=True)
for case in FOLLOWUPS:
 first=s.ask(case['first'],'sft_rag')
 second=s.ask(case['follow'],'sft_rag',[{'question':case['first'],'answer':first['text']}])
 follow.append({'case':case,'first':first,'second':second})
 print('followup',case['doc'],second['text'],flush=True)
summary={'state':'complete','manifest':manifest,'modes':[]}
for mode in ['rag','sft_rag']:
 rs=[r for r in rows if r['mode']==mode];ans=[r for r in rs if r['case']['doc']]
 summary['modes'].append({'mode':mode,'answerable':12,'keyword_pass':sum(r['keyword_pass'] for r in ans),
  'exact_source':sum(r['exact_source'] for r in ans),'selected':sum(r['result'].get('delivery_mode')=='model_selected_evidence' for r in ans),
  'fallbacks':sum(r['result'].get('delivery_mode')=='evidence_excerpt_fallback' for r in ans),
  'abstention':sum(r['result'].get('abstained',False) for r in rs if not r['case']['doc'])})
summary['challenge']=[{'q':r['case']['q'],'answer':r['result']['text'],'abstained':r['result'].get('abstained',False)} for r in rows if r['mode']=='challenge']
summary['followups']=[{'doc':r['case']['doc'],'answer':r['second']['text'],'rewritten':r['second']['rewritten']} for r in follow]
(DATA/'selection-results.json').write_text(json.dumps({'summary':summary,'records':rows,'followups':follow},ensure_ascii=False,indent=2),encoding='utf-8')
(DATA/'selection-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
