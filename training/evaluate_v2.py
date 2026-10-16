"""Frozen v2 validation; separate raw selector and pipeline metrics."""
import hashlib,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str((Path(__file__).resolve().parents[1]/'rag') if (Path(__file__).resolve().parents[1]/'rag').exists() else Path(__file__).resolve().parents[1]))
from rag_core import RAGService,evidence_sentences,selected_evidence
from evidence_contract import selection_prompt
from workspace import DATA as RAGDATA, DATA as WORKSPACE_DATA, RUNS
DATA=RAGDATA;os.environ['MEDLLM_ADAPTER']=str(RUNS/'evidence-lf-v2')
service=RAGService();retriever=service.retrieval_ready()
from rag_core import Generator
service.generator=Generator();g=service.generator
g.model.load_adapter(str(RUNS/'evidence-lf-v1'),adapter_name='v1')
g.model.set_adapter('default')
def save(name,data):(DATA/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def correct(case,result):
 if case.get('refuse'):return bool(result.get('abstained',False))
 return not result.get('abstained',False) and all(any(w in result['text'] for w in group) for group in case['groups'])

# Validation is used only to check this fixed training run, not tune final-test rules.
validation=[]
for case in json.loads((WORKSPACE_DATA/'evidence-lf-v2/audit.json').read_text(encoding='utf-8')):
 if case['split']!='validation':continue
 r=g.generate(selection_prompt(case['question'],case['options']),'sft',max_tokens=64)
 raw=r['text'].strip()
 if raw.startswith('```json\n') and raw.endswith('\n```'):raw=raw[8:-4].strip()
 try:ids=json.loads(raw)['sentences'];ok=type(ids)is list and all(type(n)is int for n in ids) and sorted(ids)==case['gold']
 except (ValueError,KeyError,TypeError):ok=False
 validation.append({'case':case,'raw':r['text'],'pass':ok})
 print('validation',len(validation),ok,flush=True)
save('lf-v2-validation.json',validation)

frozen=DATA/'final-health-v2-frozen.json';cases=json.loads(frozen.read_text(encoding='utf-8'))
pipeline=[];raw_rows=[]
for case in cases:
 for mode in ['rag','sft_rag']:
  r=service.ask(case['q'],mode);pipeline.append({'case':case,'mode':mode,'result':r,'pass':correct(case,r)})
  print('pipeline',mode,case['id'],pipeline[-1]['pass'],repr(r['text']),flush=True)
 # No rule checks or fallback below: directly evaluate evidence selection on retrieved options.
 hits=retriever.search(case['q'],3);chunks=retriever.expand_context(hits);options=evidence_sentences(chunks)
 vectors=retriever.encode([case['q']]+[s['text'] for s in options])
 options=sorted(options,key=lambda s:sum(a*b for a,b in zip(vectors[0],vectors[s['option']])),reverse=True)
 for i,s in enumerate(options):s['option']=i+1
 for model in ['base','v1','v2']:
  g.model.set_adapter('v1' if model=='v1' else 'default')
  r=g.generate(selection_prompt(case['q'],options),'base' if model=='base' else 'sft',max_tokens=64)
  selected=selected_evidence(r['text'],options,chunks)
  result={'text':selected[0] if selected else '', 'abstained':selected is not None and not bool(selected[0])}
  ok=selected is not None and correct(case,result)
  raw_rows.append({'case':case,'model':model,'raw':r['text'],'selected_text':result['text'],'valid':selected is not None,'pass':ok})
  print('raw',model,case['id'],ok,flush=True)
 g.model.set_adapter('default')
 save('lf-v2-final-raw.json',raw_rows);save('lf-v2-final-pipeline.json',pipeline)

regression=[]
old=json.loads((DATA/'user-acceptance-20261016/cases.json').read_text(encoding='utf-8'))['cases']
for mode in ['rag','sft_rag']:
 histories={}
 for case in old:
  history=histories.get(case.get('previous'),[]);r=service.ask(case['q'],mode,history)
  histories[case['id']]=(history+[{'question':case['q'],'answer':r['text']}])[-4:]
  regression.append({'case':case,'mode':mode,'result':r})
  print('regression',mode,case['id'],repr(r['text']),flush=True)
save('lf-v2-old22.json',regression)

summary={'case_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
 'validation':{'n':len(validation),'pass':sum(r['pass'] for r in validation)},'pipeline':[],'raw_selector':[],
 'note':'16 frozen new questions: 8 answerable and 8 missing-detail negatives. Pilot keyword/refusal checks, not clinical correctness.'}
for mode in ['rag','sft_rag']:
 rows=[r for r in pipeline if r['mode']==mode]
 summary['pipeline'].append({'mode':mode,'answerable_pass':sum(r['pass'] for r in rows if not r['case'].get('refuse')),
  'refusal_pass':sum(r['pass'] for r in rows if r['case'].get('refuse')),
  'rule_refusals':sum(r['result'].get('abstention_reason') in ['missing_requested_evidence','selected_evidence_missing_requirement','low_retrieval_score','scope_policy'] for r in rows),
  'model_refusals':sum(r['result'].get('abstention_reason')=='model_selected_no_evidence' for r in rows)})
for model in ['base','v1','v2']:
 rows=[r for r in raw_rows if r['model']==model]
 summary['raw_selector'].append({'model':model,'answerable_pass':sum(r['pass'] for r in rows if not r['case'].get('refuse')),
  'refusal_pass':sum(r['pass'] for r in rows if r['case'].get('refuse')),'valid_json':sum(r['valid'] for r in rows)})
save('lf-v2-summary.json',summary)
print('SUMMARY',json.dumps(summary),flush=True)
