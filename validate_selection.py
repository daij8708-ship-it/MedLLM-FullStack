import json,os
from rag_core import RAGService
from workspace import RAG
cases=[('刷完牙以外，怎样清洁牙缝？',['牙线']),('冰箱温度应保持多少？',['4']),('前期糖尿病应该找谁讨论检查？',['医生']),('一氧化碳有气味吗？',['无色','无味']),('热天如何选择户外活动时间？',['凉爽']),('家里禁止吸烟是否有帮助？',['无烟'])]
s=RAGService();rows=[]
for mode in ['rag','sft_rag']:
 for q,keys in cases:
  r=s.ask(q,mode);rows.append({'q':q,'mode':mode,'result':r})
  print(mode,q,repr(r.get('selection_raw')),r['delivery_mode'],r['text'],flush=True)
(RAG/'selection-validation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
