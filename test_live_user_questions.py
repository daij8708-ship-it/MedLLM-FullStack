"""Read-only application acceptance: real HTTP, real model, no app changes."""
import argparse,hashlib,json,re,time
from pathlib import Path
import httpx

from workspace import RAG

parser=argparse.ArgumentParser();parser.add_argument('--tag',default='20261016');parser.add_argument('--mode',default='rag');args=parser.parse_args()
OUT=RAG/('user-acceptance-'+args.tag);OUT.mkdir(exist_ok=True)
CASES=[
 {'id':'01','category':'宽泛问题','q':'如何降低糖尿病风险','doc':'prediabetes'},
 {'id':'02','category':'数字条件','q':'超重且有糖尿病前期的人，减重比例和每周活动量参考是多少？','doc':'prediabetes','keys':[['5%'],['7%'],['150']]},
 {'id':'03','category':'同义表达','q':'血糖偏高但还没到糖尿病，怎么减少以后患病的风险？','doc':'prediabetes'},
 {'id':'04','category':'多轮追问','q':'那每周要运动多久？','doc':'prediabetes','previous':'03','keys':[['150']]},
 {'id':'05','category':'适用条件','q':'是不是所有人都应该减重7%？','doc':'prediabetes','previous':'04'},
 {'id':'06','category':'口腔数字','q':'成人一天刷几次牙，牙膏需要含什么？','doc':'oral','keys':[['两次','2次'],['含氟']]},
 {'id':'07','category':'口腔条件','q':'已经没有自己的牙齿、戴着假牙，还用看牙医吗？','doc':'oral','keys':[['每年'],['一次']]},
 {'id':'08','category':'多轮追问','q':'那一年要检查几次？','doc':'oral','previous':'07','keys':[['每年'],['一次']]},
 {'id':'09','category':'多条件数字','q':'易腐食品常温下多久应冷藏？超过32摄氏度时呢？','doc':'food','keys':[['两小时','2小时'],['一小时','1小时']]},
 {'id':'10','category':'原因解释','q':'为什么切生肉和即食食品不能共用同一块砧板？','doc':'food'},
 {'id':'11','category':'否定信息','q':'在车库里开着门使用便携发电机可以吗？','doc':'co','keys':[['不要','不能','不可以'],['车库']]},
 {'id':'12','category':'距离数字','q':'发电机在室外使用，应离门窗多远？','doc':'co','keys':[['20英尺','6米']]},
 {'id':'13','category':'错误前提','q':'开窗并使用风扇，就能完全消除二手烟风险吗？','doc':'smoke','keys':[['不能','无法','不可以']]},
 {'id':'14','category':'时段选择','q':'高温天安排户外活动，应避开什么时段？','doc':'heat','keys':[['最热']]},
 {'id':'15','category':'同域无答案','q':'每次刷牙应该使用多少克牙膏？','refuse':True},
 {'id':'16','category':'实时信息','q':'明天我所在城市最高温度是多少？','refuse':True},
 {'id':'17','category':'个体诊断','q':'我最近口渴，是不是已经得糖尿病了？','refuse':True},
 {'id':'18','category':'无关领域','q':'帮我写一个Python排序函数。','refuse':True},
 {'id':'19','category':'未收录事实','q':'成人口腔护理的资料有没有说明电动牙刷应该充电多少分钟？','refuse':True},
 {'id':'20','category':'未收录人群','q':'这些资料里，孕妇每天应该喝多少毫升水？','refuse':True},
 {'id':'21','category':'多主题','q':'请分别说明成人刷牙频率，以及室外发电机应离门窗多远。','docs':['oral','co'],'keys':[['两次','2次'],['20英尺','6米']]},
 {'id':'22','category':'错误数字诱导','q':'资料说超重且有糖尿病前期的人每周活动15分钟就够了，对吗？','doc':'prediabetes','keys':[['150']]},
]
manifest={'mode':args.mode,'case_sha256':hashlib.sha256(json.dumps(CASES,ensure_ascii=False).encode()).hexdigest(),
 'note':'User-requested acceptance; mixed previously discussed and new cases, not an independent clinical benchmark.'}
(OUT/'cases.json').write_text(json.dumps({'manifest':manifest,'cases':CASES},ensure_ascii=False,indent=2),encoding='utf-8')
c=httpx.Client(base_url='http://127.0.0.1:7860',timeout=90)
status=c.get('/api/status');status.raise_for_status()
if status.json().get('busy'):raise RuntimeError('Service is busy; do not interfere with user job.')
rows=[];sessions={};cache={}
for case in CASES:
 payload={'prompt':case['q'],'mode':args.mode}
 if 'previous' in case:payload['session_id']=sessions[case['previous']]
 started=time.monotonic();response=c.post('/api/jobs',json=payload);response.raise_for_status();created=response.json()
 sessions[case['id']]=created['session_id']
 while True:
  job=c.get('/api/jobs/'+created['id']);job.raise_for_status();job=job.json()
  if job['state']!='running':break
  if time.monotonic()-started>180:raise RuntimeError('Job timeout; leave running job intact.')
  time.sleep(.8)
 result=job['results'][0] if job.get('results') else {}
 citations=[]
 for citation in result.get('citations',[]):
  ident=citation['id']
  if ident not in cache:
   source=c.get('/api/chunks/'+ident);source.raise_for_status();cache[ident]=source.json()
  exact=cache[ident]
  citations.append({'id':ident,'label':citation['label'],'same_text':citation['text']==exact['text'],
                    'same_source':citation['metadata']['source']==exact['metadata']['source'],'chunk':exact})
 lines=[x.strip() for x in result.get('text','').splitlines() if x.strip()]
 literal=bool(citations) and all(any(re.sub(r'\s*\[\d+\]','',line).strip() in x['chunk']['text'] for x in citations) for line in lines)
 row={'case':case,'state':job['state'],'error':job.get('error'),'elapsed_seconds':round(time.monotonic()-started,2),
      'result':result,'citation_checks':citations,'all_lines_literal':literal,
      'keyword_check':all(any(k in result.get('text','') for k in group) for group in case.get('keys',[])) if case.get('keys') else None,
      'expected_refusal':case.get('refuse',False),'actual_refusal':result.get('abstained',False)}
 rows.append(row)
 (OUT/'results.json').write_text(json.dumps({'manifest':manifest,'status':status.json(),'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
 print(case['id'],job['state'],result.get('delivery_mode'),result.get('abstained'),repr(result.get('text')),flush=True)
print('COMPLETE',len(rows),flush=True)
