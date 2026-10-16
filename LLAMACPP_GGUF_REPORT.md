# LoRA 合并 → GGUF 转换 → llama.cpp 接入报告

本报告只记录本轮实际执行并可复核的事实。所有大文件在 `$MEDLLM_HOME`，原始权重与适配器未被改动。

## 1. 版本与产物

| 项目 | 值 |
|---|---|
| 基础模型（原始，未改动） | `$MEDLLM_HOME\models\DeepSeek-R1-Distill-Qwen-1.5B`（Qwen2ForCausalLM，bf16，1777M 参数） |
| LoRA 适配器 | `$MEDLLM_HOME\runs\evidence-lf-v2`（PEFT LoRA，r=8，alpha=16，7 个 target_modules） |
| 合并模型 | `$MEDLLM_HOME\merged\evidence-lf-v2-merged`（bf16 safetensors，3.31 GiB） |
| GGUF f16 | `$MEDLLM_HOME\gguf\base-f16.gguf`、`evidence-lf-v2-f16.gguf`（各 3.32 GiB） |
| GGUF Q4_K_M | `$MEDLLM_HOME\gguf\base-Q4_K_M.gguf`、`evidence-lf-v2-Q4_K_M.gguf`（各 1.04 GiB，5.00 BPW） |
| llama.cpp | 官方仓库 ggml-org/llama.cpp，release **b11681**，commit **5e03a1ff450cae37a774684de55d14fb48b1f90b** |
| llama.cpp 二进制 | `$MEDLLM_HOME\bin\llama-b11681\`（`llama-b11681-bin-win-cuda-12.4-x64.zip` + `cudart-llama-bin-win-cuda-12.4-x64.zip`，CUDA 12.4 版） |
| 转换脚本 | 同 commit 的 `convert_hf_to_gguf.py` + `conversion/` + `gguf-py/`（仓库浅克隆在 `$MEDLLM_HOME\llama.cpp`） |
| 推理环境 | Python 3.11.17，torch 2.11.0+cu128，transformers 5.20.0，peft 0.21.2 |

文件哈希见 `$MEDLLM_HOME\deliverables\llamacpp-artifacts-manifest.json`。

## 2. 合并（CPU，未用量化权重）

- 合并来源是**原始 bf16 safetensors**（`models\DeepSeek-R1-Distill-Qwen-1.5B`），不是运行时 4-bit 权重。
- `peft_config.base_model_name_or_path` 与基础模型路径逐字比对一致，否则脚本直接退出。
- 7 个 target_modules 全部存在于基础模型 state_dict（缺失即退出）。
- 合并后探针核对 `model.layers.0.self_attn.q_proj.weight`：‖Δ‖ = 0.116021，相对变化 1.42e-03（≠0，证明适配器已写入）。
- 重新从磁盘加载合并模型做 CPU 贪心生成，输出 `{"sentences":[1,2]}`，与适配器训练行为一致。
- 保存：完整模型、tokenizer、config、generation_config，以及**训练实际使用的聊天模板**（`runs\evidence-lf-v2\chat_template.jinja`，同时写入 `tokenizer_config.json` 的 `chat_template`）。
- 脚本：`$MEDLLM_HOME\build\merge_lora_cpu.py`；日志：`logs\merge-lora.log`。

## 3. 转换与量化

- 两个 f16 GGUF 均由同一 commit 的官方脚本转换：合并模型用 `merged\evidence-lf-v2-merged`；基础模型用 `merged\base-for-gguf`（权重为硬链接，不额外占盘；只把 `tokenizer_config.json` 的 chat_template 设为同一训练模板）。
  - 基础模型也必须用同一模板，否则它会走官方模板的 `<think>\n`（未闭合），与运行时既有行为不一致（现有 `rag_core` 就是从适配器目录加载模板）。
- 量化：`llama-quantize.exe <in> <out> Q4_K_M`，两个模型都是 5.00 BPW / 1059.89 MiB。
- 日志：`logs\convert-gguf-f16.log`、`logs\convert-gguf-base-f16.log`、`logs\quantize-sft-q4km.log`、`logs\quantize-base-q4km.log`。
- 转换日志中 `Setting chat_template to ...<think>\n</think>\n\n` 与 `Setting add_bos_token to False` 均已确认，模板进了 GGUF 元数据。

## 4. llama-server 接入

- 两个实例，只监听本机，端口冲突时脚本报错而**不结束他人进程**：
  - `:8080` 合并模型 `evidence-lf-v2-Q4_K_M.gguf`（sft / sft_rag / 证据选择）
  - `:8081` 基础模型 `base-Q4_K_M.gguf`（base / rag 无适配器）
- 运行参数：`-c 4096 -ngl 99 -np 1 --jinja --host 127.0.0.1`；管理脚本 `Start-LlamaServers.ps1` / `Stop-LlamaServers.ps1`（pid 文件 `tmp\llama-servers.json`，停止前校验进程名）。
- **显存实测**：两个服务 + 工作台进程共 4438 MiB；停掉两个服务后 1914 MiB ⇒ 两个 llama-server（Q4_K_M 权重 + 4096 上下文 + CUDA 上下文）合计约 **2.5 GiB，约 1.26 GiB/个**。8 GiB 卡上留有明显余量。
- 真实请求已验证（`/v1/chat/completions`、流式与 `stream_options.include_usage`、`timings` 均可解析）。

## 5. 模板与思考标记核对

- 训练模板在 user 之后写入 `<｜Assistant｜><think>\n</think>\n\n`，即**预填一个空的思考块**。因此模型直接输出答案/JSON，不会只输出思考内容。
- 后端发送 `messages`，由 llama-server 套一次模板；Python 侧不重复套模板。
- 逐字核对：同一 prompt 下 transformers 侧 `apply_chat_template` 计数与 llama-server 返回的 `prompt_tokens` 完全一致（简例 65 = 65；真实检索题 226 = 226）。若重复套模板或 BOS 重复，该数字会对不上。
- 代码仍保留对 `<think>...</think>` 的剥离作为防御（`llama_backend._strip_think`），并有单测覆盖。

## 6. 受约束重试（JSON 语法）

- transformers 后端用 token trie 枚举有限 JSON 语言；llama.cpp 后端提供等价的 **GBNF 语法**（`selection_constraints.selection_gbnf`），枚举 `{"sentences":[...]}` 的全部合法串（0–3 个、无重复、编号在 1..N 内），只对双引号做转义（已核对 llama.cpp `parse_char` 只接受 `\x \u \U \t \r \n \\ \" \[ \] \-`）。
- **程序侧校验不放宽**：`selected_evidence()` 仍拒绝越界、重复、非整数、超过 3 个或在格式外的输出；首轮非法才用语法受约束重试一次。
- 真实触发：`frozen16 / sft_rag` 的食品安全四步骤题，首轮返回 `{"sentences":[1,2,3,4]}`（4 个，超上限），被拒绝后用 GBNF 重试得到 `{"sentences":[1,2,3]}` 并通过。界面上可见“首轮选句格式不合法，已受约束重试一次”。

## 7. 验收结果

### 7.1 冻结对齐用例（与 transformers 版本逐格对照）

同一批冻结用例、同一评分方式，llama.cpp 后端结果与冻结的 transformers 结果**完全一致**：

| suite | mode | transformers(ans/ref) | llama.cpp(ans/ref) | 重试 |
|---|---|---|---|---|
| dev12 | rag | 6/6 · 6/6 | 6/6 · 6/6 | 0 |
| dev12 | sft_rag | 6/6 · 6/6 | 6/6 · 6/6 | 1（合法） |
| regression16 | rag | 8/8 · 8/8 | 8/8 · 8/8 | 0 |
| regression16 | sft_rag | 8/8 · 8/8 | 8/8 · 8/8 | 0 |
| frozen16 | rag | 8/8 · 7/8 | 8/8 · 7/8 | 0 |
| frozen16 | sft_rag | 8/8 · 8/8 | 8/8 · 8/8 | 1（合法） |

产物：`$MEDLLM_HOME\rag\llamacpp-alignment-results.json`、`llamacpp-alignment-summary.json`（未覆盖任何历史文件）。脚本：`$MEDLLM_HOME\build\acceptance_llamacpp.py`。

### 7.2 22 题开发回归（HTTP 全链路）

`$MEDLLM_HOME\rag\user-acceptance-llamacpp\results.json`：16 道可答题全部给出带编号引用（16/16），6 道预期拒答全部拒答（6/6），引用回查（段落文本与来源一致）17 处全部通过，多轮追问（第 4、8 题）命中历史改写。

### 7.3 单项核对

- **排气管长度**（`发电机的排气管应该有多长？`）：rag 与 sft_rag 均拒答，理由 `missing_requested_evidence`，文本“现有资料未提供"排气管"这一对象的尺寸证据，不能用其他对象的距离代替。”——不再被答成离门窗距离。
- **食品安全四步骤**：答出清洁、生熟分开、彻底煮熟、及时冷藏并绑定引用；sft_rag 触发一次受约束重试后格式合法。
- **取消 / 超时 / 错误**：取消为真实中断（流式连接断开），实测在 0.51 s / 90 token 处停止（完整生成为 1.36 s / 200 token），之后服务仍可正常回答；服务不可达时抛 `LlamaServerError` 并在界面显示错误。
- **一键重启**：本地一键入口按“启动/复用两个 llama-server → `/health` 就绪检测 → 启动工作台”的顺序执行，实测通过；等价命令见 [`README.md`](README.md) 第四节。

## 8. 已知限制（不隐藏）

1. **基础模型在“否定信息”题上的边界翻转**（22 题中的第 11 题：“在车库里开着门使用便携发电机可以吗？”）。llama.cpp 下基础模型选中“离门窗 20 英尺”，transformers 下选中“不要在车库使用”。
   根因已定位，**与合并、转换、量化无关**：
   - 同一 prompt（token 数与内容一致，226=226）、同一候选句与次序；
   - 未量化的 `base-f16`（CPU 运行）同样选中 [2]，与 `base-Q4_K_M` 一致 ⇒ 不是量化导致；
   - 该步的 token 概率是**近乎平局**：f16 下 `2` = −0.754 与 `1` = −0.764 只差 0.01 nat；任何数值实现差异（PyTorch bf16/GPU 与 ggml f16/f32）都会翻转。
   - 微调模型对同一题给出 `1` 置信度 −0.034，对比 `2` −3.417，判断稳定且正确。
   处理方式：保留原模式语义不做静默修补；该题建议用 `微调 + RAG`。此为未微调基础模型在平局点上的数值敏感性，不是医学结论差异。
2. `frozen16 / rag` 有 1 道拒答未过（与 transformers 基线相同，属既有问题）。
3. 关键字与拒答判据只是初筛，不等于医学正确率；本批资料同源、样本很小。
4. 上下文上限 4096 token（`RAG_LLAMA_CTX`）；超限时抛错而不是静默截断。
5. 扫描件 PDF 无 OCR（既有边界，本轮未变）。

## 9. 回退方法

- 回退到 Transformers 后端：设 `$env:RAG_GENERATOR_BACKEND='transformers'` 后启动即可，不需要 llama-server。
- 配置备份：`$MEDLLM_HOME\rag\runtime-transformers-backup.json`。
- llama.cpp 相关开关都在 `$MEDLLM_HOME\rag\runtime.json`；合并/GGUF 产物可整体删除，原始模型与适配器从未被修改。
