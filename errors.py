"""跨后端共用的错误类型。

单独放一个模块是为了避免 rag_core 与 llama_backend 互相 import 成环。
"""


class ContextOverflowError(ValueError):
    """提示超出后端上下文上限。

    调用方（选句路径）应当按相关度裁剪候选后重试，并把裁剪情况写进 warnings，
    而不是把异常直接抛给用户。
    """
