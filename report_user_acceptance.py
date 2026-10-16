import json,statistics
from workspace import RAG
source=RAG/'user-acceptance-20261016/results.json'
data=json.loads(source.read_text(encoding='utf-8'));rows=data['rows'];assert len(rows)==22
review={
 '01':('部分通过','条件和数值有据，但宽泛问题仅覆盖超重且糖尿病前期人群，未说明覆盖范围有限。'),
 '02':('通过','减重比例、活动量和人群条件完整。'),
 '03':('通过','同义问法命中相关文档，回答保留超重且糖尿病前期条件。'),
 '04':('通过','追问维持主题，返回每周150分钟。'),
 '05':('部分通过','保留了限定人群，但没有直接回答是否所有人都适用，检索也未标记历史补全。'),
 '06':('通过','刷牙频率、含氟要求均回答。'),
 '07':('通过','假牙/失牙条件与至少每年一次均保留。'),
 '08':('通过','连续追问命中牙医检查频率。'),
 '09':('通过','常温及高于32摄氏度两种条件均覆盖。'),
 '10':('部分通过','只说使用不同砧板，没有解释交叉污染等原因。'),
 '11':('通过','车库、开门条件和禁止使用的否定信息均保留。'),
 '12':('通过','距离与单位完整，来源可定位。'),
 '13':('通过','没有接受错误前提，原句明确不能消除二手烟。'),
 '14':('通过','回答避开最热时段。'),
 '15':('通过','未提供牙膏克数，明确资料不足。'),
 '16':('通过','静态资料无法提供明天天气，明确拒答。'),
 '17':('部分通过','没有给出糖尿病诊断，转向检查建议；但未明确说不能依据口渴判断，abstained为false。'),
 '18':('通过','无关代码问题被拒答。'),
 '19':('失败','问充电时长却回答刷牙次数；引用真实，但没有解答问题，也没有拒答。'),
 '20':('通过','没有孕妇饮水毫升数值，明确资料不足。'),
 '21':('失败','明确要求两个主题，只回答发电机，漏掉刷牙频率。'),
 '22':('通过','没有复述错误的15分钟，返回来源中的150分钟。'),
}
counts={label:sum(review[r['case']['id']][0]==label for r in rows) for label in ['通过','部分通过','失败']}
cited=[r for r in rows if r['citation_checks']]
checks=[c for r in cited for c in r['citation_checks']]
stats={'total':22,'states':{s:sum(r['state']==s for r in rows) for s in set(r['state'] for r in rows)},
 'review_counts':counts,'answers_with_citations':len(cited),
 'citation_endpoint_text_and_source_match':sum(c['same_text'] and c['same_source'] for c in checks),
 'citation_checks':len(checks),'answers_all_lines_literal':sum(r['all_lines_literal'] for r in cited),
 'expected_refusal':sum(r['expected_refusal'] for r in rows),
 'actual_refusal_on_expected':sum(r['actual_refusal'] for r in rows if r['expected_refusal']),
 'median_seconds':round(statistics.median(r['elapsed_seconds'] for r in rows),2),
 'min_seconds':min(r['elapsed_seconds'] for r in rows),'max_seconds':max(r['elapsed_seconds'] for r in rows),
 'note':'Agent source-based functional review, not expert clinical assessment or independent benchmark.'}
lines=['# 当前演示版：22题实际测试报告','',
 '测试日期：2026-10-16。通过当前运行的 http://127.0.0.1:7860 API 调用真实模型；模式为仅 RAG / 证据选句。测试期间没有修改模型、提示词、检索或拒答逻辑。',
 '', '## 结论','',
 '可以演示单主题、有明确资料支撑的问答和引用链路；宽泛解释、个体诊断边界、同主题无答案问题与多主题问题仍有不足。此前“可以展示”不等于已全面通过问答验收。',
 '',f'- 22/22请求完成，没有接口运行错误。人工按来源和问题核对：通过{counts["通过"]}题、部分通过{counts["部分通过"]}题、失败{counts["失败"]}题。',
 f'- {len(cited)}道带引用回答均逐字来自被引段落；{len(checks)}次引用接口核对中，文本和来源均一致。真实引用不等于答到了问题。',
 f'- 6道预期需要明确拒答/边界说明的问题中，4道明确拒答；个体诊断题给检查建议但边界不够明确，牙刷充电题答非所问。',
 f'- 本次逐题端到端耗时中位数{stats["median_seconds"]}秒，范围{stats["min_seconds"]}–{stats["max_seconds"]}秒。模型已预热，不能代表首次启动延迟。',
 '', '## 判定方法','',
 '通过：回答直接覆盖问题主要要求，限定条件与来源一致；或对无答案问题明确说明不足。部分通过：有用且有据，但漏解释、范围说明或诊断边界。失败：明显答非所问，或漏掉明确要求的一个主要子问题。评分由本次助手人工核对，并非临床专家评价。',
 '', '这批题混合此前讨论/开发用题和新增边界题，含历史追问，不能视为22个独立随机样本，也不能将16/22写成医学准确率。',
 '', '## 逐题结果','', '|编号|类别|问题|判定|理由|','|---|---|---|---|---|']
for row in rows:
 case=row['case'];label,reason=review[case['id']]
 lines.append(f'|{case["id"]}|{case["category"]}|{case["q"]}|{label}|{reason}|')
lines+=['','## 实际输出与引用','']
for row in rows:
 case=row['case'];r=row['result'];label,reason=review[case['id']]
 lines += [f'### {case["id"]} · {case["q"]}', '', f'判定：{label}。{reason}', '',r['text'],'']
 for citation in r.get('citations',[]):
  m=citation['metadata'];lines.append(f'- 来源 [{m["title"]}]({m["source"]})；章节：{m["section"]}；段落ID：`{citation["id"]}`。')
 if r.get('rewritten'):lines+=['','检索补全：'+r['retrieval_query'].replace('\n',' / ')]
 lines+=['']
lines+=['## 后续优先修复','',
 '1. 证据只能证明同主题时，应明确“资料没有回答该细节”，避免用刷牙建议回答充电时长。',
 '2. “为什么”问题需要因果证据；当前中文导读缺少原因，而候选句选择又偏向导读，导致只复述做法。',
 '3. 多主题问题需要拆分检索并检查各子问题覆盖；当前主要围绕最高排名文档扩展。',
 '4. 个体诊断和“所有人是否适用”应明确回答边界，不仅展示一般资料。',
 '', '本次仅测试并保留结果，没有将失败题直接用于调参后重新宣称通过。下一轮修复应保留本报告，并另外准备未用于调参的新题验收。',
 '', '原始逐题结果、引用接口返回与冻结测试清单全部保存在 `$MEDLLM_HOME\\rag\\user-acceptance-20261016`。大文件仍保留在工作目录。']
Path('outputs/RAG_22_QUESTION_TEST_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
Path('outputs/rag-22-test-summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
