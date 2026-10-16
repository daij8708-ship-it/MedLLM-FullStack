import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str((Path(__file__).resolve().parents[1]/'rag') if (Path(__file__).resolve().parents[1]/'rag').exists() else Path(__file__).resolve().parents[1]))
from rag_core import RAGService
from workspace import RAG
p=argparse.ArgumentParser();p.add_argument('--label',required=True);args=p.parse_args()
root=RAG;cases=json.loads((root/'user-acceptance-20261016/cases.json').read_text(encoding='utf-8'))['cases']
s=RAGService();rows=[]
for mode in ['rag','sft_rag']:
 histories={}
 for case in cases:
  history=histories.get(case.get('previous'),[])
  r=s.ask(case['q'],mode,history)
  histories[case['id']]=(history+[{'question':case['q'],'answer':r['text']}])[-4:]
  keys=case.get('keys',[])
  check=r.get('abstained',False) if case.get('refuse') else (all(any(w in r['text'] for w in group) for group in keys) if keys else None)
  rows.append({'mode':mode,'case':case,'result':r,'check':check})
  print(mode,case['id'],check,repr(r['text']),flush=True)
out=root/('acceptance-repaired-'+args.label+'.json')
out.write_text(json.dumps({'label':args.label,'note':'Existing 22-case regression after engineering changes; raw outputs retained, not clinical accuracy.','rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',flush=True)
