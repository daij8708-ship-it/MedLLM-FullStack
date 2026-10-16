"""候选句重排序：用 (问题, 句子) 交叉编码分数替换纯稠密相似度的截断顺序。

候选池扩到整篇文档后，稠密相似度截断会把同文档的无关句留在窗口里，
交叉编码只看这一对文本，用来决定"哪 16 句进模型视野"。
"""
import os

from workspace import HOME

RERANKERS = ('none', 'bge_reranker_v2_m3')
MODEL_DIRS = {'bge_reranker_v2_m3': 'bge-reranker-v2-m3'}


def reranker_from_env():
    name = os.environ.get('RAG_RERANKER', 'none')
    if name not in RERANKERS:
        raise ValueError('RAG_RERANKER 只支持 ' + ' / '.join(RERANKERS) + '。')
    return SentenceReranker(name)


class SentenceReranker:
    """惰性加载：没有权重时服务仍可构造，只有真正打分才要求模型目录存在。"""

    def __init__(self, name):
        if name not in RERANKERS:
            raise ValueError('RAG_RERANKER 只支持 ' + ' / '.join(RERANKERS) + '。')
        self.name = name
        self.model = None

    @property
    def active(self):
        return self.name != 'none'

    @property
    def path(self):
        return HOME / 'models' / MODEL_DIRS[self.name]

    def _load(self):
        if not self.path.exists():
            raise RuntimeError('未找到重排序模型目录 ' + str(self.path) + '，请先按 README 的下载表获取。')
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(str(self.path), device='cpu')
        return self.model

    def scores(self, query, texts):
        if not texts:
            return []
        return list(self._load().predict([[query, t] for t in texts], batch_size=32, show_progress_bar=False))


def rank_options(options, reranker, query, max_options):
    """先给全部候选打分，再截断：截断在打分前做会让后到的块永远进不了候选。"""
    if reranker.active:
        for option, value in zip(options, reranker.scores(query, [o['text'] for o in options])):
            option['rerank_score'] = float(value)
        scored = 'rerank_score'
    else:
        scored = 'sentence_score'
    return sorted(options, key=lambda o: o[scored], reverse=True)[:max_options]
