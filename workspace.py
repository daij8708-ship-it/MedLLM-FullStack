"""工作目录根路径。

工作目录由环境变量 MEDLLM_HOME 指定：模型、适配器、GGUF、语料与评测产物都放在它下面，
仓库本身可以放在任意位置。缺省不猜路径——没设置就直接报错，避免静默读到别处的数据。
"""
import os
from pathlib import Path

_root = os.environ.get('MEDLLM_HOME')
if not _root:
    raise RuntimeError(
        '未设置环境变量 MEDLLM_HOME。请先把它指向你的工作目录，例如：\n'
        '  PowerShell:  $env:MEDLLM_HOME = "D:\\medllm"\n'
        '  bash:        export MEDLLM_HOME=/data/medllm\n'
        '目录结构见 README 第四节。'
    )

HOME = Path(_root)
RAG = HOME / 'rag'
DATA = HOME / 'data'
RUNS = HOME / 'runs'
