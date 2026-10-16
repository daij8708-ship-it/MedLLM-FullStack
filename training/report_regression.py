import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workspace import RAG
root=RAG/'user-acceptance-lf-pipeline'
data=json.loads((root/'results.json').read_text(encoding='utf-8'));rows=data['rows'];assert len(rows)==22
manual={'01':['覆盖','不代表完整'],'03':['超重且有糖尿病前期'],'05':['不能','所有人'],
        '10':['病菌','污染']}
review=[]
for row in rows:
 case=row['case'];text=row['result']['text'];checks=[]
 if case.get('refuse'):checks.append(row['actual_refusal'])
 if case.get('keys'):checks.append(row['keyword_check'])
 if case['id'] in manual:checks.append(all(w in text for w in manual[case['id']]))
 if row['citation_checks']:checks.append(all(c['same_text'] and c['same_source'] for c in row['citation_checks']))
 checks.append(row['state']=='done')
 review.append({'id':case['id'],'pass':all(checks)})
assert all(r['pass'] for r in review)
lines=['# 修复后的22题实际接口回归','',
 '2026-10-16，默认仅 RAG 模式，通过实际 FastAPI 服务逐题执行。此前为16题通过、4题部分通过、2题失败；本轮按同一功能验收要求核对后，22题均达到要求。系统范围提示、拒答与多问题拆分属于程序行为，不是模型独立生成能力。',
 '', '这22题已经用于指导修复，因此是开发回归集。不能将22/22称为独立泛化准确率或医学正确率。另12道健康资料新问法仍有4道资料不足题拒答失败，详见 LLaMA Factory 报告。',
 '', '|编号|问题|实际输出|','|---|---|---|']
for r in rows:lines.append('|'+r['case']['id']+'|'+r['case']['q']+'|'+r['result']['text'].replace('\n',' / ')+'|')
citations=[c for r in rows for c in r['citation_checks']]
lines += ['',f'引用接口共核对{len(citations)}次，返回的段落文本和来源均与回答引用一致。两个主题回答分别绑定[1]、[2]，没有丢失子问题。6道预期拒答题均明确拒答。',
 '', '完整任务输出和引用接口返回保存在 `$MEDLLM_HOME\\rag\\user-acceptance-lf-pipeline`。旧报告和旧原始结果没有覆盖。']
Path('outputs/RAG_22_QUESTION_REGRESSION_AFTER_LF.md').write_text('\n'.join(lines),encoding='utf-8')
summary={'mode':'rag','n':22,'regression_pass':22,'refusal_pass':6,'refusal_n':6,'citation_checks':len(citations),
         'note':'Development regression after engineering fixes; not independent accuracy.'}
Path('outputs/rag-lf-regression-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
