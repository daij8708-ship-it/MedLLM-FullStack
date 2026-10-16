# MedLLM-FullStack

医疗健康科普场景下的**抽取式证据问答**全链路实践：LoRA 微调 → BGE-M3 + ChromaDB 检索 →
合并 / GGUF 量化 / llama.cpp 本地推理，外加一套**可复核的 span 级证据评测**。

系统不做自由生成。模型只输出“选中了哪几句证据”的 JSON，引用文案由程序按语料原文**逐字绑定**，
所以每一条引用都能回查到原文句子，也正因如此评测可以把“答对”拆成“句子选对没有”。

## 一、链路

| 阶段 | 做法 |
|---|---|
| 微调 | LLaMA-Factory，LoRA（r=8 / alpha=16 / dropout=0.05 / 7 个 target_modules），基座 `DeepSeek-R1-Distill-Qwen-1.5B`；训练目标是**证据选句行为**，不是自由问答 |
| 检索 | BGE-M3（1024 维）+ ChromaDB 持久集合（cosine，相似度阈值 0.45） |
| 部署 | PEFT 适配器在 CPU 上以 bf16 从**非量化**权重合并 → GGUF f16 → Q4_K_M（5.00 BPW，1.04 GiB）→ llama.cpp `llama-server`（官方 CUDA 12.4 版，release b11681） |
| 服务 | FastAPI；llama.cpp 走 OpenAI 兼容的 `/v1/chat/completions` 流式接口，Transformers + PEFT 作为回退后端 |
| 约束 | 选句 JSON 用 GBNF 语法约束解码（transformers 侧用等价的 token-trie），程序侧校验不放宽：**最多 3 句、不重复、编号不越界**；首轮非法才受约束重试一次 |
| 多轮 | 追问按会话历史改写检索问题；资料不足时分层拒答（`personal_diagnosis` / `missing_requested_evidence` / `low_retrieval_score` 等） |

## 二、评测：把“能答”换成可复核的指标

关键词初筛那种“12 题过 11 题 = 91.7%”不构成证据，本仓库改成 **span 级金标准**：
给每道可答题标注三层句子集合，全部是冻结语料的句子子集，**不撰写任何新文本**。

| 层 | 定义 | 用途 |
|---|---|---|
| `core` | 直接包含问题所需事实的句子 | 召回分母 |
| `acceptable` | `core` ∪ 允许展示且不算答非所问的句子 | 严格精确率分母 |
| `doc_scope` | 答案文档内的全部句子 | 宽松精确率分母 |

派生指标：严格 `precision`、宽松 `lenient_precision`、`off_topic`（引到别的文档 = 真错误）、
`recall`、`exact`、误拒/漏拒混淆矩阵、95% Wilson 区间；出错用例单独计数并排除，不静默当 0。
`lenient_precision` 在下面的两个候选口径下都恒为 1.0（候选取自答案文档，宽松分母覆盖整篇文档），
**是候选构造的恒等式，不用来做配置对比**；有区分度的是严格 P、padding、exact。

**主结果 A**（54 例 × 2 选句模式 × 2 后端，0 错误；候选 = 线上默认 `summary_only`）：

| 后端 | 模式 | 严格 P | R | F1 | exact | 误拒 | 漏拒 | padding | 跨文档 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama.cpp | rag（基座选句） | 0.8854 | 0.9688 | 0.9146 | 25/32 | 0/32 | 1/22 | 7 | 0 |
| llama.cpp | sft_rag（微调选句） | 0.6250 | 1.0000 | 0.7490 | 8/32 | 0/32 | 0/22 | 26 | 0 |
| Transformers | rag | 0.9115 | 1.0000 | 0.9406 | 26/32 | 0/32 | 1/22 | 7 | 0 |
| Transformers | sft_rag | 0.5156 | 1.0000 | 0.6709 | 2/32 | 0/32 | 0/22 | 35 | 0 |

**主结果 B**（同一批 54 例，只换候选范围与排序方式；llama.cpp，`RAG_MAX_OPTIONS=16`）：

| 候选 / 排序 | 模式 | 严格 P | R | F1 | exact | padding |
|---|---|---:|---:|---:|---:|---:|
| 20 句导读 · 稠密 | rag | 0.8854 | 0.9688 | 0.9146 | 25/32 | 7 |
| **343 句 · 稠密** | rag | **0.5208** | 0.9531 | 0.6636 | **3/32** | 32 |
| **343 句 · 交叉编码重排** | rag | **0.4531** | 0.8906 | 0.6000 | 0/32 | 35 |
| 20 句导读 · 稠密 | sft_rag | 0.6250 | 1.0000 | 0.7490 | 8/32 | 26 |
| **343 句 · 稠密** | sft_rag | **0.4448** | 0.9219 | 0.5981 | **0/32** | 41 |
| **343 句 · 交叉编码重排** | sft_rag | 0.4479 | 0.8906 | 0.5907 | 1/32 | 39 |

六条结论：

1. **候选宇宙是指标的一部分**：把可达句子从 20 扩到 343，召回没多救回任何一题
   （两档都是 30/32 题的 `core` 进了窗口），严格精确率与 `exact` 显著下降
   （`exact` 的 Wilson 区间 [0.612, 0.890] 与 [0.032, 0.242] 不重叠）。窄候选下的好分数含"题目被做简单了"的成分。
2. **微调换来召回、付出 padding**：微调把 `recall` 拉到 1.0，但 padding 从 7 涨到 26/35 句——
   它学会了"多选几句"，严格精确率反而低于基座。两个后端一致，不是某个运行时的副作用。
3. **交叉编码重排序不是解**（负面结果）：bge-reranker-v2-m3 把窗口**顺序**改了（只有 23/54 题顺序不变），
   但窗口**集合**在 52/54 题上不变，精确率没恢复；反而有 29 题"同一批句子换了编号"就改变了模型的选句
   （`sft_rag dev22/04`：`[1,2]`→`[1,11]`）。**瓶颈在选择器对编号的依赖，不在排序侧。**
4. **两个后端在窄候选主表上不可区分**：rag 的 F1 差 2.6 个百分点，`exact` 区间大幅重叠。
5. **拒答方向干净但有 1 例漏拒**：`over_refusal = 0/32`，`under_refusal = 1/22`
   （"电动牙刷每次充电时间及噪音分贝是多少？"被用刷牙句回答了，属答非所问）。
6. **单轮结果有噪声底**：同配置重复跑，冷启动那一轮与热轮差 2/54 行；报告只引用超出噪声底的差异，
   并撤回了曾提出的 prompt-cache 归因（同 prompt 连发 8/8、交错 6/6 一致）。

完整判据、机制统计、单例分析与方法缺陷记录见 [`EVALUATION_SPAN_REPORT.md`](EVALUATION_SPAN_REPORT.md)。

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
gold_span.py              span 级金标准构建（标签唯一命中校验）
span_metrics.py           指标与 Wilson 区间
run_span_eval.py          span 级评测 runner（真实 HTTP）
test_rag.py               单测（含 gold 完整性、后端契约、指标方向）
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

### 5. 复现评测

```powershell
python evidence_index.py --scope retrieved_sentences  # 句子索引（--scope 决定可达宇宙口径）
python gold_span.py                                   # 金标准（标签唯一命中校验）
python -m unittest test_rag -v                        # 指标、gold 完整性、开关契约
# 服务侧开关必须先对上，否则 runner 拒绝运行：
#   $env:RAG_CANDIDATE_SCOPE = 'summary_only' | 'retrieved_sentences'
#   $env:RAG_RERANKER        = 'none' | 'bge_reranker_v2_m3'
#   $env:RAG_MAX_OPTIONS     = 1..16（默认 16）
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

## 六、仓库里没有的东西

- 模型权重、LoRA 适配器、合并模型、GGUF、Chroma 持久库、原始 HTML、日志与缓存等大文件/产物，
  一律不入库，按上面链接与步骤自行准备。
- 语料原文由脚本从 CDC 公网页面抓取，仓库只保留抓取与解析代码。

## 七、已知限制

1. 样本量小：54 例（32 可答 / 22 拒答），置信区间很宽，只能支撑方向性判断，
   **不能宣称医学准确率**。
2. 金标准由 AI 助手依据语料原文标注，**未经临床专家复核**；每个句子 key 都带原文可逐句审计。
3. `core` 常为 1 句，召回分辨率粗（只能取 0 / 0.5 / 1）。
4. 判据是“证据句选择”，不是医学正确性；系统本身是抽取式，不做自由生成。
5. 默认候选只覆盖 6 篇中文导读的 20 句（全语料 343 句）；`RAG_CANDIDATE_SCOPE=retrieved_sentences`
   可扩到检索文档全部句子，但实测严格精确率 0.885→0.521、`exact` 25/32→3/32 且召不回（见主结果 B），
   因此默认仍是窄候选。交叉编码重排也不能补救，已在报告里作为负面结果记录。
6. 主结果 B 的宽候选与重排序对照只在 llama.cpp 后端上跑过，transformers 后端未在新开关下重跑。
7. 同配置重复跑存在 ±1~2/54 行的冷启动不可复现；单轮数字不单独引用。
8. 上下文上限 4096 token，超限抛错而不是静默截断。
9. 扫描件 PDF 无 OCR。

## 八、更多文档

- [`README_DELIVERY.md`](README_DELIVERY.md) —— 交付说明：入口、路径、验收结果、回滚
- [`LLAMACPP_GGUF_REPORT.md`](LLAMACPP_GGUF_REPORT.md) —— 合并 / 量化 / llama.cpp 接入的版本、参数、显存实测与模板一致性核对
- [`EVALUATION_SPAN_REPORT.md`](EVALUATION_SPAN_REPORT.md) —— span 级评测判据、主表、单例与方法缺陷
- [`EVIDENCE_ALIGNMENT_AND_RETRY_REPORT.md`](EVIDENCE_ALIGNMENT_AND_RETRY_REPORT.md)、
  [`LLAMA_FACTORY_V2_EVIDENCE_REPORT.md`](LLAMA_FACTORY_V2_EVIDENCE_REPORT.md) —— 证据对齐与受约束重试
