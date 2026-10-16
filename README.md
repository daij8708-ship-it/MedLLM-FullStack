# MedLLM-FullStack

医疗健康科普场景下的**抽取式证据问答**全链路实践：LoRA 微调 → BGE-M3 + ChromaDB 检索 →
合并 / GGUF 量化 / llama.cpp 本地推理，外加一套**可复核的 span 级证据评测**。

系统不做自由生成。模型只输出“选中了哪几句证据”的 JSON，引用文案由程序按语料原文**逐字绑定**，
所以每一条引用都能回查到原文句子，也正因如此评测可以把“答对”拆成“句子选对没有”。

## 一、链路

| 阶段 | 做法 |
|---|---|
| 微调 | LLaMA-Factory，LoRA（r=8 / alpha=16 / dropout=0.05 / 7 个 target_modules），基座 `DeepSeek-R1-Distill-Qwen-1.5B`；训练目标是**证据选句行为**，不是自由问答 |
| 检索 | BGE-M3（1024 维）+ ChromaDB 持久集合（cosine，相似度阈值 0.48 —— 由 110 例阈值扫描选定，见评测报告 §6.2） |
| 部署 | PEFT 适配器在 CPU 上以 bf16 从**非量化**权重合并 → GGUF f16 → Q4_K_M（5.00 BPW，1.04 GiB）→ llama.cpp `llama-server`（官方 CUDA 12.4 版，release b11681） |
| 服务 | FastAPI；llama.cpp 走 OpenAI 兼容的 `/v1/chat/completions` 流式接口，Transformers + PEFT 作为回退后端 |
| 约束 | 选句 JSON 用 GBNF 语法约束解码（transformers 侧用等价的 token-trie），程序侧校验不放宽：**最多 3 句、不重复、编号不越界**；首轮非法才受约束重试一次 |
| 多轮 | 追问按会话历史改写检索问题；资料不足时分层拒答（`personal_diagnosis` / `missing_requested_evidence` / `low_retrieval_score` 等） |

## 二、评测：把“能答”换成可复核的指标

关键词初筛那种“12 题过 11 题 = 91.7%”不构成证据，本仓库改成 **span 级金标准**：
给每道可答题标注三层句子集合，全部是冻结语料的句子子集，**不撰写任何新文本**。

| 层 | 定义 | 用途 |
|---|---|---|
| `core` | 直接包含问题所需事实的句子（一题需要几个事实就标几句） | 召回分母 |
| `acceptable` | `core` ∪ 允许展示且不算答非所问的句子 | 严格精确率分母 |
| `doc_scope` | 答案文档内的全部句子 | 宽松精确率分母 |

派生指标：严格 `precision`、宽松 `lenient_precision`、`off_topic`（引到别的文档 = 真错误）、
`recall`、`exact`、误拒/漏拒混淆矩阵、95% Wilson 区间；出错用例单独计数并排除，不静默当 0。
`lenient_precision` 在两种候选口径下都接近 1.0（0.976~0.979），是候选构造的恒等式，**不用来做配置对比**；
有区分度的是严格 P、padding、exact、off_topic。

**金标准 110 例**（70 可答 / 40 拒答）＝ 历史三个套件 54 例 + **本轮新写的 `expansion` 56 例**
（问题清单随仓库分发，写完才第一次跑评测）。`core` 由 2~3 句组成的题有 10 道，
召回不再只能取 0 / 0.5 / 1。

**主结果**（llama.cpp，110 例，0 错误；`RAG_MAX_OPTIONS=16`）：

| 候选 / 排序 | 模式 | 严格 P | R | F1 | exact | 误拒 | 漏拒 | padding | 跨域 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 句导读 · 稠密 | rag | **0.8024** | 0.9238 | **0.8372** | **41/70** | 0/70 | **10/40** | 28 | 3 |
| 20 句导读 · 稠密 | sft_rag | 0.5974 | 0.9462 | 0.7077 | 14/65 | 5/70 | **2/40** | 57 | 4 |
| 343 句 · 稠密 | rag | 0.4631 | 0.8524 | 0.5905 | 3/70 | 0/70 | **10/40** | 78 | 4 |
| 343 句 · 稠密 | sft_rag | 0.4355 | 0.8595 | 0.5720 | 1/70 | 0/70 | **7/40** | 91 | 4 |
| 343 句 · 交叉编码重排 | rag | 0.4333 | 0.7952 | 0.5524 | 2/70 | 0/70 | **10/40** | 81 | 4 |
| 343 句 · 交叉编码重排 | sft_rag | 0.4469 | 0.8406 | 0.5754 | 3/69 | 1/70 | **7/40** | 85 | 4 |

同一批题在**回退后端（Transformers + 4-bit PEFT）**上补跑了同样三组：窄候选 rag 严格 P **0.8214**（`exact` 45/70）、
扩候选 **0.5929**（20/70）、再重排 **0.5571**（17/70）。**方向与 llama.cpp 完全一致**（窄候选最好、扩候选掉、重排不恢复），
但**幅度小得多**——所以"扩候选使精确率腰斩"应读成 llama.cpp 量化路径上的量级，不是普遍规律。
两后端**绝对值不可互换**（宽候选 rag 档 0.463 vs 0.593，两者的 `exact` Wilson 区间不重叠），
可跨后端引用的是**方向**与**拒答计数**（rag 档漏拒两边都是 10/40、误拒都是 0/70）。
完整两表见评测报告 §3.1 / §3.2。

六条结论：

1. **默认口径仍然最好**：把候选从 20 句扩到 343 句，严格精确率 0.802→0.463、`exact` 41/70→3/70，
   召回也从 0.924 掉到 0.852（以上为 llama.cpp；transformers 同向，0.821→0.593）。
   窄候选下的高分里含"题目被做简单了"的成分——这一点用
   **"旧 54 例子集在本轮产物里逐字复现（P=0.8854 / exact=25/32 / padding=7）"** 做过对照，
   差异全部来自新增题，不是环境漂移（旧子集也不受阈值调整影响：被挡住的 3 例都在新增套件里）。
   另外扩候选**并没有多给一句能救回召回的句子**：可答题中 core 句可达的题数在两种口径下都是 **69/70**，
   变的只是送进选择器的窗口（中位 3 句 → 14 句）——**是干扰项变多，不是可选内容变多**。
2. **扩样本推翻了上一版三条结论**：跨文档引用不再是 0（旧集 0，新集出现，全量 3~4 句）；
   漏拒从 1/22 涨到 13/40（阈值调到 0.48 后为 10/40，见结论 3）；宽松精确率不再是 1.0。旧题集**测不出**这些错法。
3. **拒答短板是可量化的，最省的一档已经修掉**：对检索阈值做扫描后把缺省从 0.45 调到 **0.48**，
   漏拒 13/40 → **10/40** 且**零新增误拒**（六个 arm 一致）。被挡住的正是"语料外主题"那 3 例
   （维生素C剂量、新冠疫苗、抑郁症）；**剩下 7 例是个人判断/药物/商品这类语义问题**，
   继续抬阈值要付 12~41 例误拒，得换规则或分类器。详见评测报告 §6.2（含两条被数据否掉的特征假设）。
4. **交叉编码重排仍不是解**：110 例上 rag 精确率 0.463→0.433（transformers 0.593→0.557），耗时约 2.6 倍。
   机制是重排几乎不换候选（实际送进模型的窗口集合 **99/110 不变**）却改了编号顺序（顺序仅 **34/110 不变**），
   而**选择器跟编号不跟内容**；`selection_options` 的两组计数在两个后端上逐字相同。
5. **微调的价值在拒答与召回**：`sft_rag` 漏拒 2/40（rag 是 10/40，因为它会输出空选），
   召回 0.946 也更高；代价是严格精确率 0.597、padding 57、5 例误拒。
6. **噪声底要分区间说**：窄候选档**逐字复现**（本轮四个 arm 全部满足），
   llama.cpp 的 16 句候选档会漂 1~5/110（逐题对拍得出，与阈值无关），**成因未隔离**；
   报告只引用远超对应量级的差异，并撤回曾提出的 prompt-cache 归因（同 prompt 连发 8/8、交错 6/6 一致）。

完整判据、失败模式清单、机制分析与方法缺陷记录见 [`EVALUATION_SPAN_REPORT.md`](EVALUATION_SPAN_REPORT.md)。

## 三、目录

```
corpus.py                 CDC 语料抓取 / 解析 / 切块
rag_core.py               检索 + 选句 + 引用绑定 + 拒答判定
rag_api.py                FastAPI 服务（任务、会话、SSE 流式）
llama_backend.py          llama.cpp 生成后端（含流式取消、GBNF 约束、think 剥离）
selection_constraints.py  选句 JSON 的 token-trie 与 GBNF 约束
query_plan.py             问题拆分、个人诊断/尺寸缺失等边界判定
evidence_contract.py      选句提示词契约
evidence_index.py         语料句子稳定索引（chunk_id + sha1(规范化句子)）
reranker.py               候选句交叉编码重排序（可选，决定哪些句子进模型视野）
errors.py                 跨后端共用错误类型（上下文超限，由选句路径裁剪后重试）
gold_span.py              span 级金标准构建（标签唯一命中校验）
evidence_expansion_cases.json  新增 56 道题的清单（随仓库分发，可离线重建金标准）
span_metrics.py           指标与 Wilson 区间
run_span_eval.py          span 级评测 runner（真实 HTTP）
test_rag.py               单测（含 gold 完整性、开关契约、超限裁剪、指标方向）
requirements-ocr.txt      扫描件 PDF 的可选 OCR 依赖（默认不装）
static/                   前端页面
training/                 微调数据构建、训练与验收脚本
```

## 四、跑起来

### 1. 准备目录与依赖

工作目录由环境变量 `MEDLLM_HOME` 指定（见 `workspace.py`）；没设置会直接报错，不会去猜路径。
把大文件都放在它下面，仓库本身可以放在任意位置：

```
$MEDLLM_HOME/
  models/DeepSeek-R1-Distill-Qwen-1.5B/   基座模型（原始，勿改动）
  models/bge-m3/                          向量模型（离线加载）
  runs/<adapter>/                         LoRA 适配器
  merged/                                 CPU 合并后的完整模型
  gguf/                                   f16 与 Q4_K_M 量化文件
  rag/                                    runtime.json、chroma/、评测产物
  bin/llama-b11681/                       llama-server 与 CUDA 运行时
  env/                                    推理用 Python 环境
```

```powershell
$env:MEDLLM_HOME = 'D:\medllm'          # 换成你自己的工作目录
pip install fastapi uvicorn pydantic httpx beautifulsoup4 pypdf chromadb sentence-transformers
pip install torch transformers peft
```

### 2. 抓语料并建索引

```powershell
python corpus.py                       # 从 CDC 公网页面抓取并切成 chunks.json
python evidence_index.py               # 句子稳定索引
```

### 3. 合并适配器与量化

合并必须从**非量化**权重出发（不要拿 4-bit 运行时权重当合并源）。本项目用官方脚本转换：

```powershell
# 1) CPU 上合并 LoRA → merged/<name>
python convert_hf_to_gguf.py <merged_dir> --outfile gguf/model-f16.gguf --outtype f16
# 2) 量化
llama-quantize.exe gguf/model-f16.gguf gguf/model-Q4_K_M.gguf Q4_K_M
```

合并脚本会把**训练实际使用的聊天模板**写进输出目录与 `tokenizer_config.json`——
该模板在 user 之后预填一个空的 `<think>\n</think>\n\n`，模型因此直接输出答案/JSON，
不会只输出思考内容。转换日志里应能看到 `Setting chat_template to ...` 与
`Setting add_bos_token to False`。

### 4. 启动

```powershell
# 合并模型（sft / sft_rag / 证据选择）
llama-server.exe -m gguf/model-Q4_K_M.gguf  --host 127.0.0.1 --port 8080 -c 4096 -ngl 99 -np 1 --jinja
# 基座模型（rag 模式）
llama-server.exe -m gguf/base-Q4_K_M.gguf   --host 127.0.0.1 --port 8081 -c 4096 -ngl 99 -np 1 --jinja

$env:RAG_GENERATOR_BACKEND = 'llamacpp'
$env:RAG_LLAMA_BASE_URL    = 'http://127.0.0.1:8081'
$env:RAG_LLAMA_SFT_URL     = 'http://127.0.0.1:8080'
python rag_api.py --port 7860 --open
```

打开 <http://127.0.0.1:7860>。只监听本机。回退到 Transformers 后端：设
`RAG_GENERATOR_BACKEND=transformers` 后同样启动即可，不需要 llama-server。

扫描件 PDF 要入库时，额外装 OCR 依赖并显式开启（默认关闭，不装则上传扫描件会明确报错并提示做法）：

```powershell
pip install -r requirements-ocr.txt
$env:RAG_PDF_OCR = 'rapidocr'      # off（默认）/ rapidocr，写错直接拒绝启动
```

### 5. 复现评测

```powershell
python evidence_index.py --scope retrieved_sentences  # 句子索引（--scope 决定可达宇宙口径）
python gold_span.py                                   # 金标准（标签唯一命中校验）
python -m unittest test_rag -v                        # 指标、gold 完整性、开关契约
# 服务侧开关必须先对上，否则 runner 拒绝运行：
#   $env:RAG_CANDIDATE_SCOPE = 'summary_only' | 'retrieved_sentences'
#   $env:RAG_RERANKER        = 'none' | 'bge_reranker_v2_m3'
#   $env:RAG_MAX_OPTIONS     = 1..16（默认 16）
#   $env:RAG_MIN_SCORE       = 0.30..0.90（默认 0.48；写错或超范围会拒绝启动）
# 产物里会存下当时的 /api/status 快照（含 retrieval_threshold），保证数字能追到阈值。
python run_span_eval.py --backend llamacpp --candidate-scope summary_only --reranker none --tag base
python run_span_eval.py --backend llamacpp --candidate-scope retrieved_sentences --reranker none --tag wide
python run_span_eval.py --backend llamacpp --candidate-scope retrieved_sentences --reranker bge_reranker_v2_m3 --tag rsrr
```

`run_span_eval.py` 会逐一核对运行中服务的 `--backend` / `--candidate-scope` / `--reranker`
与实际 `/api/status` 是否一致，任何一项不符就拒绝运行，避免产物被误标。

## 五、需要自己下载的依赖（不在本仓库）

| 依赖 | 用途 | 链接 |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 基座模型 | <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B> · <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B> |
| LLaMA-Factory | SFT 训练框架 | <https://github.com/hiyouga/LLaMA-Factory> |
| BAAI/bge-m3 | 检索向量模型 | <https://huggingface.co/BAAI/bge-m3> |
| BAAI/bge-reranker-v2-m3 | 候选句重排序（可选，`RAG_RERANKER=bge_reranker_v2_m3` 时才加载） | <https://huggingface.co/BAAI/bge-reranker-v2-m3> · <https://modelscope.cn/models/BAAI/bge-reranker-v2-m3> |
| llama.cpp | GGUF 转换 / 量化 / `llama-server` | <https://github.com/ggml-org/llama.cpp>（本项目用 release **b11681** 的 `llama-b11681-bin-win-cuda-12.4-x64.zip` + `cudart-llama-bin-win-cuda-12.4-x64.zip`） |
| CDC 语料 | 评测语料来源 | 由 `corpus.py` 按 `SOURCES` 里的 6 个官方页面自动抓取 |
| 扫描件 OCR（可选） | 扫描件 PDF 入库 | `pip install -r requirements-ocr.txt`（rapidocr-onnxruntime 1.4.4 + pypdfium2 5.14.0，识别权重随 wheel 自带约 16MB，运行时不联网） |

## 六、仓库里没有的东西

- 模型权重、LoRA 适配器、合并模型、GGUF、Chroma 持久库、原始 HTML、日志与缓存等大文件/产物，
  一律不入库，按上面链接与步骤自行准备。
- 语料原文由脚本从 CDC 公网页面抓取，仓库只保留抓取与解析代码。

## 七、已知限制

1. 样本 110 例（70 可答 / 40 拒答），比 54 例宽，但仍只够支撑方向性判断，**不能宣称医学准确率**。
2. 金标准由 AI 助手依据语料原文标注，**未经临床专家复核**；每个句子 key 都带原文可逐句审计。
   新增的 56 道题同样是 AI 所写——"未参与开发"指没有用它们调过系统，**不等于题目对抗性强**。
3. `core` 仍有 60/70 题只有 1 句，召回分辨率粗（0 / 0.33 / 0.67 / 1）。
4. 判据是“证据句选择”，不是医学正确性；系统本身是抽取式，不做自由生成。
5. **拒答仍是最弱的一环**：漏拒 10/40（阈值已从 0.45 调到 0.48，零新增误拒换掉 3 例），
   剩下 7 例是个人判断/药物/商品这类**语义**问题（剂量、疫苗、高血压、肠镜等），
   与候选范围、生成后端都无关。修法在拒答判定（规则或分类器），不在排序。
   另外 0.48 是贴 110 例样本取的（与可答题最小余弦只差 0.011），**换语料要重扫**。
6. 默认候选只覆盖 6 篇中文导读的 20 句（全语料 343 句）；`RAG_CANDIDATE_SCOPE=retrieved_sentences`
   可扩到检索文档全部句子，但实测严格精确率 0.802→0.463、`exact` 41/70→3/70 且召回也降；
   transformers 后端同向但幅度小得多（0.821→0.593）。因此默认仍是窄候选。
   交叉编码重排不能补救（两个后端都不恢复精确率），已作为负面结果记录。
7. 宽候选与重排序对照已在两个后端各跑完三组（llama.cpp 与 Transformers）。**两后端方向一致，
   但绝对指标不可互换**（宽候选 rag 档 0.463 vs 0.593，`exact` 区间不重叠），可跨后端引用的只有方向与拒答计数。
8. **噪声底分区间**：窄候选档逐字复现，llama.cpp 的 16 句候选档有 1~5/110 的选句漂移
   （与阈值无关，成因未隔离）；单轮数字必须在带波动的口径下引用。
9. 上下文超限不再抛错：选句路径按相关度裁剪候选并在 `warnings` 里声明丢弃句数；
   自由生成路径没有可裁剪的候选，仍会报错。默认口径（≤16 句）不会触发裁剪，
   六份产物共 1320 条记录的 `context_trimmed` 全部为 0；该路径由单测与定向验证覆盖。
10. 评测在服务刚重启、索引未就绪时就发第一个请求，可能撞上连接超时（本轮 1 例，已单独重问成功）；
    接入时建议先等 `/api/status` 的 `ready`。
10. 扫描件 PDF 需显式开启 OCR（`RAG_PDF_OCR=rapidocr`）并安装 `requirements-ocr.txt`；默认关闭。
    实测中文识别可用，但**英文词间空格会丢**（`Reduce salt`→`Reducesalt`），而引用是逐字展示的，
    所以 OCR 页会带 `ocr=true` 标记并在引用弹窗提示对照原件；未做 CER 评测。

## 八、更多文档

- [`README_DELIVERY.md`](README_DELIVERY.md) —— 交付说明：入口、路径、验收结果、回滚
- [`LLAMACPP_GGUF_REPORT.md`](LLAMACPP_GGUF_REPORT.md) —— 合并 / 量化 / llama.cpp 接入的版本、参数、显存实测与模板一致性核对
- [`EVALUATION_SPAN_REPORT.md`](EVALUATION_SPAN_REPORT.md) —— span 级评测判据、主表、单例与方法缺陷
- [`EVIDENCE_ALIGNMENT_AND_RETRY_REPORT.md`](EVIDENCE_ALIGNMENT_AND_RETRY_REPORT.md)、
  [`LLAMA_FACTORY_V2_EVIDENCE_REPORT.md`](LLAMA_FACTORY_V2_EVIDENCE_REPORT.md) —— 证据对齐与受约束重试
