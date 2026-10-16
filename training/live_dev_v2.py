import json,time
from pathlib import Path
import httpx

from workspace import RAG as root

cases=json.loads((root/'health-holdout-lf-v1.json').read_text(encoding='utf-8'))['rows']
cases=[r['case'] for r in cases if r['mode']=='rag']
c=httpx.Client(base_url='http://127.0.0.1:7860',timeout=90);rows=[]
for mode in ['rag','sft_rag']:
 for case in cases:
  created=c.post('/api/jobs',json={'prompt':case['q'],'mode':mode});created.raise_for_status();ident=created.json()['id']
  for _ in range(180):
   job=c.get('/api/jobs/'+ident);job.raise_for_status();job=job.json()
   if job['state']!='running':break
   time.sleep(.5)
  if job['state']!='done':raise RuntimeError(job)
  r=job['results'][0];ok=r.get('abstained',False) if case.get('refuse') else all(any(w in r['text'] for w in group) for group in case['groups'])
  rows.append({'mode':mode,'case':case,'result':r,'pass':ok})
  print(mode,case['id'],ok,repr(r['text']),flush=True)
(root/'lf-v2-dev12-live.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',flush=True)
