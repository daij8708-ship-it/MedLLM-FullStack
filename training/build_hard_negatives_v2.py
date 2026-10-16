"""Source-backed health selection behavior; six RAG corpus documents excluded."""
import hashlib,json,random,re,sys
from pathlib import Path
sys.path.insert(0,str((Path(__file__).resolve().parents[1]/'rag') if (Path(__file__).resolve().parents[1]/'rag').exists() else Path(__file__).resolve().parents[1]))
from evidence_contract import selection_prompt
from workspace import DATA, HOME, RUNS
ROOT=DATA/'evidence-lf-v2';ROOT.mkdir(exist_ok=True)
manifest_path=Path('outputs/medical-v4-source-manifest.json')
if not manifest_path.exists():manifest_path=ROOT/'source-manifest.json'
sources=json.loads(manifest_path.read_text(encoding='utf-8'))['topics']
NEGATIVES={
 'antibiotics':['抗生素每天服用几次？','服用抗生素的疗程是多少天？','哪种抗生素用于哪种细菌？','抗生素的购买价格是多少？','头孢类抗生素具体叫什么名字？','抗生素需要用多热的水服用？','抗生素放冰箱保存多久？','抗生素能与哪一种食物同服？','感冒药的包装颜色是什么？','病毒感染后多久会发烧？','普通感冒的潜伏期是多少？','这里推荐的抗生素每粒多少克？'],
 'handwashing':['搓洗时水的温度需要多少摄氏度？','一次洗手消耗多少升水？','肥皂每次用多少克？','洗手液开封后多久更换？','哪种品牌的洗手液最好？','消毒洗手液需要多少浓度？','擦手纸的尺寸要多大？','洗手台应有多高？','水龙头出水流量是多少？','洗完手后隔多久要再洗？','指甲应该剪到多长？','清洁流动水的具体水压是多少？'],
 'sleep':['成年人午睡最长可以睡多久？','睡前多久应该关闭手机？','卧室应保持多少摄氏度？','床垫应有多厚？','枕头应该几厘米高？','助眠药具体怎么服用？','睡前可以喝多少杯咖啡？','卧室湿度应为多少？','睡眠手环的价格是多少？','睡到几点起床最适合？','室内照明应为多少流明？','午睡需要使用什么型号的闹钟？'],
 'activity':['锻炼时心率应达到多少？','每次运动前要喝多少毫升水？','跑鞋多久需要更换？','运动鞋应该选什么尺码？','跑步机应该调几档坡度？','每次力量训练要用多少公斤重量？','每分钟应该走多少步？','运动完要补充多少克蛋白质？','健身房需要缴纳多少费用？','膝盖疼痛的确切原因是什么？','手环的电池可使用几天？','运动后洗澡的水温是多少？'],
}
train=[];validation=[];audit=[]
for ti,topic in enumerate(sources):
 sentences=[x.strip() for x in re.split(r'(?<=。)',topic['answer']) if x.strip()]
 for qi,q in enumerate(topic['train']+NEGATIVES[topic['id']]):
  supported=qi<len(topic['train'])
  split='validation' if qi in [6,7,18,19] else 'train'
  # Explicit labels from the supplied full source-backed answer; no target facts invented.
  targets=list(range(len(sentences))) if supported else []
  distractor=sources[(ti+1)%len(sources)]['answer'].split('。')[0]+'。'
  all_sentences=sentences+[distractor]
  for variation in range(4):
   rng=random.Random(31000+ti*1000+qi*10+variation);order=list(range(len(all_sentences)));rng.shuffle(order)
   options=[{'option':i+1,'text':all_sentences[j]} for i,j in enumerate(order)]
   gold=sorted(order.index(i)+1 for i in targets)
   assert len(gold)<=3
   item={'instruction':selection_prompt(q,options),'input':'','output':json.dumps({'sentences':gold},separators=(',',':'))}
   (train if split=='train' else validation).append(item)
   audit.append({'split':split,'source':topic['url'],'topic':topic['id'],'question':q,'options':options,'gold':gold,
                 'kind':'source-backed answer selection' if supported else 'missing detail hard negative'})
# Keep generic selection behavior to reduce forgetting; only original training split.
generic=json.loads((DATA/'evidence-lf-v1/train.json').read_text(encoding='utf-8'))
train+=random.Random(42).sample(generic,160)
for name,items in [('train',train),('validation',validation)]:
 random.Random(42).shuffle(items)
 (ROOT/(name+'.json')).write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'source-manifest.json').write_text(json.dumps({'topics':sources},ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'dataset_info.json').write_text(json.dumps({'hard_train':{'file_name':'train.json'},'hard_validation':{'file_name':'validation.json'}},indent=2),encoding='utf-8')
metadata={'train':len(train),'validation':len(validation),'health_topics':4,
 'split_note':'health validation question-disjoint but same source topics; no independent generalization claim',
 'excluded':'All six current RAG corpus source documents and user acceptance questions excluded from SFT.',
 'hashes':{n:hashlib.sha256((ROOT/(n+'.json')).read_bytes()).hexdigest() for n in ['train','validation']}}
(ROOT/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
config=json.loads((HOME/'evidence-lf-v1.json').read_text(encoding='utf-8'))
config.update(adapter_name_or_path=str(RUNS/'evidence-lf-v1'),dataset_dir=str(ROOT),dataset='hard_train',
 eval_dataset='hard_validation',output_dir=str(RUNS/'evidence-lf-v2'),max_steps=160,learning_rate=5e-5)
(HOME/'evidence-lf-v2.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print(json.dumps(metadata))
