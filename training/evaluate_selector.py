import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str((Path(__file__).resolve().parents[1]/'rag') if (Path(__file__).resolve().parents[1]/'rag').exists() else Path(__file__).resolve().parents[1]))
from rag_core import Generator
from evidence_contract import selection_prompt
from workspace import DATA, RAG
p=argparse.ArgumentParser();p.add_argument('--adapter');p.add_argument('--split',default='test');args=p.parse_args()
if args.adapter:os.environ['MEDLLM_ADAPTER']=args.adapter
g=Generator();root=DATA/'evidence-lf-v1'
cases=[r for r in json.loads((root/'audit.json').read_text(encoding='utf-8')) if r['split']==args.split]
records=[];mode='sft' if args.adapter else 'base'
for i,case in enumerate(cases):
 r=g.generate(selection_prompt(case['question'],case['options']),mode,max_tokens=64)
 raw=r['text'].strip()
 if raw.startswith('```json\n') and raw.endswith('\n```'):raw=raw[8:-4].strip()
 try:
  ids=json.loads(raw)['sentences'];valid=isinstance(ids,list) and all(type(n)is int and 1<=n<=len(case['options']) for n in ids) and len(ids)==len(set(ids))
 except (ValueError,TypeError,KeyError):ids=None;valid=False
 records.append({'case':case,'raw':r['text'],'valid':valid,'pass':valid and sorted(ids)==case['gold']})
 print(i+1,len(cases),records[-1]['pass'],flush=True)
label='lf-v1' if args.adapter else 'base'
summary={'model':label,'split':args.split,'n':len(records),'exact_set_pass':sum(r['pass'] for r in records),
 'valid_json':sum(r['valid'] for r in records),
 'no_evidence_n':sum(not r['case']['gold'] for r in records),
 'no_evidence_pass':sum(r['pass'] for r in records if not r['case']['gold']),
 'note':'Synthetic held-out document/domain evidence selection, not clinical QA accuracy.'}
out=RAG/('lf-selector-'+label+'-'+args.split+'.json')
out.write_text(json.dumps({'summary':summary,'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
print('SUMMARY',json.dumps(summary),flush=True)
