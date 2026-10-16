"""Synthetic evidence-selection behavior SFT; no CDC facts or acceptance questions."""
import hashlib,json,random,sys
from pathlib import Path
sys.path.insert(0,str((Path(__file__).resolve().parents[1]/'rag') if (Path(__file__).resolve().parents[1]/'rag').exists() else Path(__file__).resolve().parents[1]))
from evidence_contract import selection_prompt
from workspace import DATA
ROOT=DATA/'evidence-lf-v1';ROOT.mkdir(parents=True,exist_ok=True)
DOMAINS=[
 ('图书馆','借阅期限','天','借书证颜色','借阅期为{v}天。','每人最多借3本书。'),
 ('博物馆','开放时长','小时','停车场面积','每日开放{v}小时。','周一闭馆。'),
 ('快递站','包裹保存期限','天','工作人员工资','包裹最多保存{v}天。','凭取件码领取。'),
 ('公交站','发车间隔','分钟','驾驶员年龄','发车间隔为{v}分钟。','末班车在晚上九点发车。'),
 ('体育馆','储物柜押金','元','屋顶高度','储物柜押金为{v}元。','入场需要预约。'),
 ('展览馆','团体人数上限','人','照明灯功率','团体人数上限为{v}人。','请勿使用闪光灯。'),
 ('租车店','归还宽限时间','分钟','车辆生产日期','归还宽限时间为{v}分钟。','取车时请检查外观。'),
 ('自习室','预约提前时间','天','地板厚度','最多提前{v}天预约。','保持安静。'),
 # validation and test domains deliberately disjoint from training domains
 ('游船码头','航行时长','分钟','船长生日','每趟航行{v}分钟。','大风天气停航。'),
 ('音乐厅','入场提前时间','分钟','座椅重量','需提前{v}分钟入场。','演出期间请关闭手机铃声。'),
 ('露营地','帐篷租金','元','管理员身高','帐篷租金为{v}元。','禁止在营区生火。'),
 ('洗衣店','取衣等待时间','小时','洗衣机转速','取衣需要等待{v}小时。','取衣请出示凭据。'),
]
splits={'train':[],'validation':[],'test':[]};audit=[]
for domain_index,(domain,field,unit,unknown,fact,other) in enumerate(DOMAINS):
 split='train' if domain_index<8 else 'validation' if domain_index<10 else 'test'
 for n in range(8):
  name=f'{chr(65+domain_index)}{n+1}号{domain}';value=7+n*3
  facts=[name+fact.format(v=value),name+other,'该资料由场所公告整理，不包含其他信息。']
  variants=[(f'{name}的{field}是多少？',[0]),(f'请告诉我{name}的{unknown}。',[]),
            (f'有人说{name}的{field}是{value+5}{unit}，资料实际怎么说？',[0]),
            (f'{name}的{unknown}具体是多少？',[]),
            (f'关于{name}，请找出{field}与另一项使用规定。',[0,1])]
  for variant,(q,gold) in enumerate(variants):
   rng=random.Random(1000+domain_index*100+n*5+variant);order=list(range(3));rng.shuffle(order)
   options=[{'option':i+1,'text':facts[j]} for i,j in enumerate(order)]
   selected=sorted(order.index(g)+1 for g in gold)
   prompt=selection_prompt(q,options);answer=json.dumps({'sentences':selected},ensure_ascii=False,separators=(',',':'))
   item={'instruction':prompt,'input':'','output':answer}
   splits[split].append(item);audit.append({'split':split,'domain':domain,'document':name,'question':q,'options':options,'gold':selected})
for split,items in splits.items():
 random.Random(42).shuffle(items)
 (ROOT/(split+'.json')).write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'dataset_info.json').write_text(json.dumps({f'evidence_{s}':{'file_name':s+'.json','columns':{'prompt':'instruction','query':'input','response':'output'}} for s in splits},indent=2),encoding='utf-8')
(ROOT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
assert not ({x['document'] for x in audit if x['split']=='train'} & {x['document'] for x in audit if x['split']!='train'})
manifest={'kind':'self-authored synthetic non-medical instruction tuning; not medical knowledge training',
 'counts':{s:len(v) for s,v in splits.items()},'split':'by domain/documents, not random QA split',
 'sha256':{s:hashlib.sha256((ROOT/(s+'.json')).read_bytes()).hexdigest() for s in splits}}
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
