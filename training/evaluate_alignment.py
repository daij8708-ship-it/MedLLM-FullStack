"""Alignment + single constrained retry acceptance; no training."""
import hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str((Path(__file__).resolve().parents[1]/'rag') if (Path(__file__).resolve().parents[1]/'rag').exists() else Path(__file__).resolve().parents[1]))
from rag_core import RAGService
from workspace import RAG as DATA, RUNS
os.environ['MEDLLM_ADAPTER']=str(RUNS/'evidence-lf-v2')
s=RAGService();rows=[]
dev=[r['case'] for r in json.loads((DATA/'health-holdout-lf-v1.json').read_text(encoding='utf-8'))['rows'] if r['mode']=='rag']
regression=json.loads((DATA/'final-health-v2-frozen.json').read_text(encoding='utf-8'))
final=DATA/'alignment-final-frozen.json';new=json.loads(final.read_text(encoding='utf-8'))
for name,cases in [('dev12',dev),('regression16',regression),('frozen16',new)]:
 for mode in ['rag','sft_rag']:
  for case in cases:
   result=s.ask(case['q'],mode)
   ok=result.get('abstained',False) if case.get('refuse') else (not result.get('abstained',False) and all(any(w in result['text'] for w in group) for group in case['groups']))
   rows.append({'suite':name,'mode':mode,'case':case,'result':result,'pass':ok})
   print(name,mode,case['id'],ok,'retry',result.get('format_retry',False),repr(result['text']),flush=True)
   (DATA/'alignment-evaluation-results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
summary={'case_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'suites':[],
 'note':'Frozen source-based pilot; keyword and abstention checks are not clinical correctness. Development suites are not held-out.'}
for name,cases in [('dev12',dev),('regression16',regression),('frozen16',new)]:
 for mode in ['rag','sft_rag']:
  selected=[r for r in rows if r['suite']==name and r['mode']==mode]
  summary['suites'].append({'suite':name,'mode':mode,'n':len(selected),
   'answerable_n':sum(not r['case'].get('refuse',False) for r in selected),
   'answerable_pass':sum(r['pass'] for r in selected if not r['case'].get('refuse',False)),
   'refusal_n':sum(r['case'].get('refuse',False) for r in selected),
   'refusal_pass':sum(r['pass'] for r in selected if r['case'].get('refuse',False)),
   'format_retries':sum(r['result'].get('format_retry',False) for r in selected),
   'retry_valid':sum(r['result'].get('format_retry',False) and r['result'].get('delivery_mode')!='evidence_selection_rejected' for r in selected)})
(DATA/'alignment-evaluation-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('SUMMARY',json.dumps(summary),flush=True)
