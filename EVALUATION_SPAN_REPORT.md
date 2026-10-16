# Span 级证据评测（v3，2026-10-16）

本报告把"能答"换成可复核的指标。之前的判据是关键词初筛（11/12 = 91.7%），项目文档自己写了"不能这么写"——所以那个数字等于零。本轮改为 **span 级金标准**：给每道可答题标注"哪些证据句算答对"，再算选句精确/召回、误拒率、漏拒率。

v3 相对 v2 新增三件事：**候选池从 20 句扩到 343 句的对照实验**、**bge-reranker-v2-m3 交叉编码重排序的对照实验（负面结果）**、**同配置重复跑标定出的噪声底**。§3 主表改用 gold v2 口径。

## 1. 判据与分母（每个数字都能追到定义）

三层集合，都是冻结语料的**句子子集**，不撰写任何新文本：

| 层 | 定义 | 用途 |
|---|---|---|
| `core` | 直接包含问题所需事实的句子 | 召回分母 |
| `acceptable` | `core` ∪ 允许展示且不算答非所问的句子 | 严格精确率分母 |
| `doc_scope` | 答案文档内的**全部**句子（v1 只放该文档的中文导读句） | 宽松精确率分母 |

派生指标：
- `precision` = 选中∩acceptable / 选中 —— **严格**：多引一句同文档但没回答问题的句子会被扣
- `lenient_precision` = 选中∩doc_scope / 选中 —— **宽松**：只要来自答案文档就不算错
- `off_topic` = 选中 − doc_scope —— 来自**别的文档**，属真错误
- `padding` = 选中 − acceptable（但仍在 doc_scope 内）—— 同文档无关句
- `recall` = 选中∩core / core；`exact` = core 全覆盖**且**无 padding
- 混淆矩阵：`over_refusal`（该答却拒）/`answerable`、`under_refusal`（该拒却答）/`refusable`
- 报 95% **Wilson 区间**：n=32 时点估计不可单独引用
- 出错用例单独计数并排除，不静默当 0

> **`lenient_precision` 在两个候选范围下都恒为 1.0，不能用它做配置对比。** 候选句取自检索到的答案文档，而 `doc_scope` 覆盖整篇答案文档，所以"宽松不扣分"是**候选构造的恒等式**，不是模型表现。v1 报告把它当成一列结果，本轮取消这个用法；有区分度的是严格 `precision`、`padding`、`exact`。

## 2. 语料、句子索引与金标准

- 语料：`chunks.json` 67 段，sha256 `e12a92186cf0c195…`（冻结未改；v1/v2 两个句子索引记录的 `corpus_sha256` 相同）。v2 报告此处曾写 `5b9d989c…`，核对产物时更正。
- **分句器修正**：旧 `SENTENCE_SPLIT` 只按 `。！？` 断句，英文段落整段被当成一句。改为中文标点 + 英文句点（带 `2.5`、`e.g. the`、`U.S. embassy` 守护，有单测）。语料句子数 **242 → 343**；旧索引留档为 `sentence-index-v1-legacy-splitter.json`。
- **候选宇宙是显式开关**（`RAG_CANDIDATE_SCOPE`，写错直接拒绝启动），两个口径的可达句子数由 `evidence_index.reachable_keys(chunks, scope)` 算出并写进索引产物：

  | 口径 | 规则 | 可达句子 |
  |---|---|---|
  | `summary_only`（线上默认） | 只有 top-1 文档的中文导读块进候选 | **20** |
  | `retrieved_sentences` | 检索并扩展后的每篇文档全部句子都进候选 | **343** |

- 金标准：`evidence-gold-v2.json`（`supersedes: evidence-gold-v1`，sha256 `455ddb3a…`），54 例（32 可答 / 22 拒答）。**`core` 与 `acceptable` 与 v1 逐字相同**，只把 `doc_scope` 从"该文档导读句"扩成"该文档全部句子"，因此召回与 `exact` 可与 v1 直接比较。
- 20 个句子标签由脚本按"唯一子串"解析成稳定 key（`chunk_id + sha1(规范化句子)`），标签不唯一命中即报错退出。**Gold 是语料的子集，由构造保证**，并由单测复核（`test_every_gold_key_exists_in_corpus`、`test_core_is_subset_of_acceptable`）。
- 标注者：AI 助手依据语料原文标注，**未经临床专家复核**；每个 key 都带原文可逐句审计。
- 追问题（dev22/04、05、08）按 `previous` 串会话上下文提问；不带会话会把追问测错（见 §8）。

## 3. 窄候选主结果（llama.cpp · gold v2 · `summary_only`，0 错误）

| 模式 | 严格 P | R | F1 | exact | exact 95% CI | 误拒 | 漏拒 | padding | 跨文档 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| rag（基座选句） | **0.8854** | 0.9688 | **0.9146** | **25/32** | [0.612, 0.890] | 0/32 | 1/22 | 7 | 0 |
| sft_rag（微调选句） | 0.6250 | 1.0000 | 0.7490 | 8/32 | [0.133, 0.421] | 0/32 | 0/22 | 26 | 0 |

产物 `span-eval-llamacpp-summary_only-v2base2.json`（`8b1e134f…`）。**与已发布的 v1 产物逐行一致**（108 行全等），说明 §2 的 `doc_scope` 变更没有动到召回与 `exact` 的口径。

Transformers 后端（gold v1 口径，"宽松 P"一列已按 §1 作废）：rag 严格 P 0.9115 / R 1.0000 / exact 26/32（CI [0.647, 0.911]）、sft_rag 严格 P 0.5156 / R 1.0000 / exact 2/32（CI [0.017, 0.202]）—— **微调换来召回、付出 padding** 跨后端一致。两后端的 rag 差 2.6 个百分点、`exact` 区间大幅重叠，只能说**差异低于这批样本的可分辨水平**。

## 4. 候选池实验：把 20 句扩到 343 句（本轮新增）

同一个服务、同一份 gold v2、同一批 54 例，只换 `RAG_CANDIDATE_SCOPE`：

| 口径 / 模式 | 严格 P | R | F1 | exact | exact 95% CI | padding | 窗口中位句数 |
|---|---:|---:|---:|---:|---|---:|---:|
| `summary_only` rag | 0.8854 | 0.9688 | 0.9146 | 25/32 | [0.612, 0.890] | 7 | 3 |
| `retrieved_sentences` rag | **0.5208** | 0.9531 | 0.6636 | **3/32** | [0.032, 0.242] | 32 | 12 |
| `summary_only` sft_rag | 0.6250 | 1.0000 | 0.7490 | 8/32 | [0.133, 0.421] | 26 | 3 |
| `retrieved_sentences` sft_rag | **0.4448** | 0.9219 | 0.5981 | **0/32** | [0.000, 0.107] | 41 | 12 |

机制核对（逐题统计，不是推测）：

| 统计 | `summary_only` | `retrieved_sentences` |
|---|---:|---:|
| 可答题里"窗口内至少有一句 core"的题数 | **30/32** | **30/32** |
| 窗口内 acceptable 句平均数 | 1.1 | 1.1 |
| core 句排在第 1 位的题数 | 30 | 28 |
| 平均选中句数（rag / sft_rag） | 1.25 / 1.97 | 2.00 / 2.25 |

**扩大候选没有多救回任何一道题。** 两档都是 30/32 题的 core 进了窗口（另外 2 题的 core 在未检索到的文档里，扩池也够不着）；新增的 323 句候选里 acceptable 句平均仍是 1.1 句——**扩池只增加了干扰项**。代价是严格精确率 0.885→0.521、`exact` 25/32→3/32，且这组差异在 n=32 上可分辨（`exact` 的 Wilson 区间 [0.612, 0.890] 与 [0.032, 0.242] 不重叠）。

所以此前那个 0.885 的精确率里，**有一部分来自"候选只有 20 句、答案句几乎必然在场且排在第 1"，不能全记在选择器能力上**。反过来，线上默认继续保留 `summary_only` 有了数据依据：当前抽取式选择器在宽候选下不占优。

## 5. 重排序实验：bge-reranker-v2-m3（本轮新增，负面结果）

做法（`reranker.py`，开关 `RAG_RERANKER`）：稠密召回 → 对**全部**候选句做 (问题, 句子) 交叉编码打分 → 再截断 top-16 进模型视野。截断放在打分之后，否则后到的块永远进不了候选（单测 `test_cross_encoder_can_rescue_a_low_dense_candidate` 锁住这条）。CPU 单线程，不占生成侧显存。

| 宽候选下的排序 | 模式 | 严格 P | R | F1 | exact | padding | 该轮 108 请求耗时 |
|---|---|---:|---:|---:|---:|---:|---:|
| 稠密相似度 | rag | 0.5208 | 0.9531 | 0.6636 | 3/32 | 32 | 185 s |
| 交叉编码重排 | rag | **0.4531** | 0.8906 | 0.6000 | 0/32 | 35 | 645 s |
| 稠密相似度 | sft_rag | 0.4448 | 0.9219 | 0.5981 | 0/32 | 41 | 185 s |
| 交叉编码重排 | sft_rag | 0.4479 | 0.8906 | 0.5907 | 1/32 | 39 | 645 s |

**重排序没有把精确率找回来**（rag 反而 0.521→0.453），代价是墙钟时间约 3.5 倍。

为什么没用——逐题比对两轮的窗口与原始选句：

- 窗口**集合**在 52/54 题上完全相同，但窗口**顺序**只有 23/54 题相同；
- 也就是 **29 题属于"同一批句子换了编号"**；全部 54 题里 rag 有 40 题、sft_rag 有 41 题的模型原始 JSON 不变，其余都跟着编号动了。典型：`sft_rag dev22/04` 从 `{"sentences":[1,2]}` 变成 `{"sentences":[1,11]}`。
- 排序器本身是有效的：`dev22/11`（车库发电机）上交叉编码把"不要在室内、地下室或车库使用便携发电机，即使门窗敞开也不可以"打到 **0.9965**，同批其余句子 ≤0.37；稠密分也把它排第 1。

判读：**选择器跟的是编号，不是句子内容**。所以瓶颈在抽取式选择器本身（训练时的候选分布就是 2–4 句的短列表），排序侧修不动。要修得从选择侧下手：用宽候选池重造训练数据，或对"选中句是否真的支撑回答"做内容级校验。

## 6. 可复现性与噪声底（本轮新增）

同一配置（llama.cpp / `summary_only` / gold v2）跑了 3 次：

| 轮次 | 服务状态 | rag | sft_rag | 与发布产物逐行比对 |
|---|---|---|---|---|
| 发布产物 `span-eval-llamacpp-v2` | 热 | 0.8854 / 25/32 | 0.6250 / 8/32 | — |
| `…-v2base` | **冷**（`/api/status` 报 `embedding_ready=false`，索引尚未构建） | 0.8854 / 25/32 | 0.6111 / 7/30 | 差 2 行（`dev22/05`、`dev22/11`：`[1,2]`→`[]`） |
| `…-v2base2` | 热 | 0.8854 / 25/32 | 0.6250 / 8/32 | **108 行全等** |

排查过程与撤回的假设：
- 两次的 `/api/status` 里 `adapter`、`model_served`、`retrieval_threshold`、两个 llama-server 地址**全同**，只差冷启动字段 → 排除配置漂移。
- 逐题比对 `question`、`retrieval_query`、`rewritten`、`selection_options`（文本与编号）**完全相同**，也没触发格式重试 → 排除提示词与候选差异。
- 曾怀疑 llama-server 的 prompt cache：同一 prompt 连发 8 次得 8/8 一致，再交错两种 prompt 制造缓存抖动得 6/6 一致 → **该假设不成立，已撤回**。
- 冷/热差异未彻底归因（能排除的都排除了），因此按测量限制处理。

**采信规则：单轮结果的 ±1~2/54 行是噪声底。** 本报告只引用超出噪声底的差异——§4 的候选池对照在 `exact` 上区间不重叠，成立；§5 的 `rag` P 差 0.068、`exact` 差 3/32、两个模式方向一致，但召回差 2 题接近噪声底，因此只据此说"重排序没有恢复精确率"，**不说"重排序显著有害"**。

## 7. 结论

1. **候选宇宙是指标的一部分**：可达句子从 20 扩到 343，召回一点没涨（30/32 不变），严格精确率与 `exact` 显著下降。窄候选下的好分数里含"题目被做简单了"的成分。
2. **`lenient_precision` 不能用于配置对比**：它在两个口径下都恒为 1.0，是候选构造的恒等式（v1 曾误当作结果）。
3. **微调换来召回、付出 padding**：`sft_rag` 在窄候选下把 R 拉到 1.0 但 padding 7→26，严格 P 低于基座；跨后端一致。
4. **重排序不是解**：交叉编码换了顺序但基本没换窗口集合，精确率未恢复；证据指向选择器对编号的依赖。
5. **拒答方向干净，但有 1 例漏拒**：`over_refusal = 0/32`，`under_refusal = 1/22`（`frozen16/a08` 电动牙刷充电/噪音问题被用刷牙句回答，属答非所问）。这是目前最值得修的一类错误。

## 8. 过程中修掉的方法缺陷（都影响过结论）

- **`confusion()` 把 answerable/refusable 的分母写反**，被新加的单测抓到（`test_errors_are_excluded_not_silently_zeroed`）。
- **第一版 runner 没给追问题带会话**，导致 sft_rag 2 次"空选"、rag 1 次跨文档选句——全是测量假象。按 `previous` 串会话后消失。
- **分句器只认中文标点**，把可达宇宙压成 20 句，并使 v1 的 `lenient_precision=1.0` 成为恒等式；本轮同时修分句器与判据（gold v2 的 `doc_scope`）。
- **`corpus.py` 把工作目录的 `DATA` 指错一层**（`<home>/data` 而不是 `<home>/rag`），真实问题报 `No such file or directory` 时 **34 项单测仍然全绿**。修正后加了 `WorkspaceTests.test_every_module_agrees_on_the_rag_root` 做回归锁。**单测绿 ≠ 系统能跑**，这是本轮最贵的一课。
- 报告里 `chunks.json` 的 sha256 曾写成 `5b9d989c…`，实际 `e12a9218…`（§2）。
- 第一版产物改名为 `span-eval-llamacpp-v1-SUPERSEDED-missing-sessions*.json` 保留留痕，**其数字不得引用**。

## 9. 单例

- 漏召（窄候选）1 例：`llamacpp/rag dev22/11` 选中距离句而非"不要在车库使用"。与既有根因一致：该步 token 概率近平局（f16 下 −0.754 vs −0.764，差 0.01 nat），transformers 后端在此例正确。
- 漏拒 1 例：`frozen16/a08`（见 §7.5），两后端一致。
- 编号敏感 1 例：`sft_rag dev22/04` 在窗口集合不变时从 `[1,2]` 跳到 `[1,11]`（§5）。

## 10. 限制（不得省略）

1. 样本 n=32 可答 / 22 拒答，区间很宽；只能支撑方向性判断，**不能宣称医学准确率**。
2. 金标准由 AI 依据语料标注，未经临床专家复核；`acceptable` 的宽严是我定的判据，报告把两种宽严都给出，避免用宽严制造结论。
3. `core` 常为 1 句，召回分辨率粗（只能取 0 / 0.5 / 1）。
4. 判据是"证据句选择"，不是医学正确性；系统是抽取式，不做自由生成。
5. 语料 7 份 / 67 段，体量与文档分布都不代表真实场景。
6. **§4 与 §5 的宽候选、重排序对照只在 llama.cpp 后端上跑过**（transformers 后端未在新开关下重跑），跨后端一致性只对窄候选的主结论成立。
7. 重排序跑在 CPU 上，108 请求 645 s；换 GPU 或改批量会改变耗时结论，但不改变"窗口集合基本不变"这一机制结论。
8. 冷启动下存在 ±1~2/54 行的不可复现（§6），单轮数字不得单独引用。

## 11. 简历口径

- ✅ 可写：设计并落地 **span 级证据评测**（core/acceptable/doc_scope 三层判据、Wilson 区间、误拒/漏拒混淆矩阵、出错用例独立计数），在 54 例冻结集上做 **候选范围 × 排序方式 × 选句模式 × 2 后端**对照；定位并量化了"窄候选池使精确率虚高"（P 0.885→0.521、exact 25/32→3/32 且区间不重叠）；实现 bge-reranker-v2-m3 交叉编码重排并**以负面结果排除排序侧优化**（窗口集合 52/54 不变、精确率未恢复），据此把瓶颈定位到抽取式选择器对编号的依赖；用重复跑标定噪声底，并撤回自己提出的 prompt-cache 归因。
- ❌ 不可写：任何形如"准确率 9x%"的整体准确率；把抽取式选句指标说成医学问答准确率；把 `lenient_precision=1.0` 说成"零无关引用"；把重排序说成"提升了精度"。

## 12. 复现

```powershell
# 0) 先设置工作目录（模型、语料与评测产物都在它下面）
$env:MEDLLM_HOME = 'D:\medllm'
# 1) 语料句子索引（稳定 key；--scope 决定可达宇宙口径）
python evidence_index.py --scope retrieved_sentences
# 2) 金标准（标签唯一命中校验；产出 evidence-gold-v2.json）
python gold_span.py
# 3) 单测（gold 完整性、后端契约、候选范围与重排序开关、指标方向）
python -m unittest test_rag -v
# 4) 跑评测：后端、候选范围、重排序三者都必须与运行中的服务一致，否则拒绝运行
#    服务侧开关：$env:RAG_CANDIDATE_SCOPE、$env:RAG_RERANKER
python run_span_eval.py --backend llamacpp --candidate-scope summary_only --reranker none --tag base
python run_span_eval.py --backend llamacpp --candidate-scope retrieved_sentences --reranker none --tag wide
$env:RAG_RERANKER = 'bge_reranker_v2_m3'    # 权重需自行下载，见 README 第五节
python run_span_eval.py --backend llamacpp --candidate-scope retrieved_sentences --reranker bge_reranker_v2_m3 --tag rsrr
```

产物 sha256（前 8 位）：gold v2 `455ddb3a`、句子索引 `f2724d91`、
窄候选 `span-eval-llamacpp-summary_only-v2base2` = `8b1e134f`、
宽候选 `span-eval-llamacpp-retrieved_sentences-rs-warm` = `1cbe274a`、
重排序 `span-eval-llamacpp-retrieved_sentences-bge_reranker_v2_m3-rsrr`。
每个产物都含逐题原始响应与当时的 `/api/status` 快照。
