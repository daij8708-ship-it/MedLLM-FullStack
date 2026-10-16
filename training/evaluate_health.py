import argparse,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str((Path(__file__).resolve().parents[1]/'rag') if (Path(__file__).resolve().parents[1]/'rag').exists() else Path(__file__).resolve().parents[1]))
from rag_core import RAGService
from workspace import RAG
parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
path=Path(__file__).with_name('health_holdout_cases.json');cases=json.loads(path.read_text(encoding='utf-8'))
s=RAGService();rows=[]
for mode in ['rag','sft_rag']:
 for case in cases:
  r=s.ask(case['q'],mode);text=r['text']
  ok=r.get('abstained',False) if case.get('refuse') else (not r.get('abstained',False) and all(any(w in text for w in group) for group in case['groups']))
  rows.append({'mode':mode,'case':case,'result':r,'check':ok})
  print(args.label,mode,case['id'],ok,repr(text),flush=True)
summary={'label':args.label,'case_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'note':'Frozen before new adapter training evaluation; pilot source-based keyword/refusal checks, not clinical accuracy.',
 'modes':[{'mode':m,'n':12,'answerable_pass':sum(r['check'] for r in rows if r['mode']==m and not r['case'].get('refuse')),
 'refusal_pass':sum(r['check'] for r in rows if r['mode']==m and r['case'].get('refuse'))} for m in ['rag','sft_rag']]}
out=RAG/('health-holdout-'+args.label+'.json');out.write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print('SUMMARY',json.dumps(summary),flush=True)
