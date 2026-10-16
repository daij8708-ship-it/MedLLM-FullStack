import json,time,httpx
from pathlib import Path
c=httpx.Client(base_url='http://127.0.0.1:7860',timeout=180)
raw='# Acceptance fixture\nThis synthetic document has verification code RAGFIXTURE20261016. It is not health advice.\n'
r=c.post('/api/documents',data={'permission':'true','title':'验收用合成文档（非健康资料）','source':'local acceptance fixture','updated':'2026-10-16'},files={'file':('acceptance-fixture.md',raw.encode(),'text/markdown')})
r.raise_for_status()
upload=r.json()
s=c.post('/api/search',json={'query':'RAGFIXTURE20261016','k':3});s.raise_for_status()
result={'upload':upload,'search':s.json(),'status':c.get('/api/status').json(),'openapi_paths':len(c.get('/openapi.json').json()['paths'])}
Path('outputs/rag-api-acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('Upload/search/OpenAPI verified',result['status'])
