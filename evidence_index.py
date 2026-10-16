"""Stable sentence index over the frozen corpus.

Gold evidence annotations must reference corpus sentences by a stable key, never by
the model-facing option numbers (those are re-assigned per query by embedding
re-rank, so they are not reproducible). The key is the sha1 of the whitespace-
normalised sentence text, prefixed by the chunk id: the same string tokenises to
the same key in the index, in a model answer, and in an audited citation.
"""
import hashlib
import json
import re
from pathlib import Path

from workspace import RAG as DATA

# 中英混切：中文靠句末标点，英文靠「句点 + 空白 + 大写/引号/括号开头」。
# 只写 `(?<=[.!?])\s+` 会把 2.5、e.g. the、U.S. embassy 误切，所以要求下一句以大写起头。
SENTENCE_SPLIT = r'(?<=[。！？])\s*|(?<=[.!?])[ \t]+(?=[A-Z"\'(“‘])|\n+'


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text):
    return [s.strip() for s in re.split(SENTENCE_SPLIT, text)]


def sentence_key(chunk_id, text):
    digest = hashlib.sha1(normalize(text).encode('utf-8')).hexdigest()[:16]
    return f'{chunk_id}:{digest}'


def index_sentences(chunks, min_length=8):
    """Every corpus sentence with provenance. Gold is a subset of this list."""
    out = []
    for chunk in chunks:
        meta = chunk['metadata']
        for position, sentence in enumerate(split_sentences(chunk['text'])):
            sentence = sentence.strip()
            if len(sentence) < min_length:
                continue
            out.append({'key': sentence_key(chunk['id'], sentence), 'text': sentence,
                        'chunk_id': chunk['id'], 'doc_id': meta['doc_id'], 'title': meta['title'],
                        'section': meta['section'], 'kind': meta.get('kind', 'original'),
                        'source': meta['source'], 'position': position})
    return out


def load_chunks():
    return json.loads((DATA / 'chunks.json').read_text(encoding='utf-8'))


def reachable_keys(chunks, scope='summary_only'):
    """Keys the extractive selector can ever offer, per candidate scope.

    ``summary_only`` reproduces the historical behaviour of ``evidence_sentences``: the
    first chunk's sentences are taken and every later chunk is skipped unless it is an
    editorial summary, so only the top-1 document's overview is selectable.
    ``retrieved_sentences`` lets every retrieved chunk contribute sentences, so any chunk
    that can surface in the top-3 becomes reachable — i.e. the whole corpus.
    Recorded here so the report states the candidate universe instead of assuming it.
    """
    if scope == 'retrieved_sentences':
        return {e['key'] for e in index_sentences(chunks)}
    reachable = set()
    for chunk in chunks:
        if chunk['metadata'].get('kind') == 'editorial_summary':
            for sentence in split_sentences(chunk['text']):
                if len(sentence.strip()) >= 8:
                    reachable.add(sentence_key(chunk['id'], sentence.strip()))
    return reachable


def dump_index(path=None, scope='summary_only'):
    chunks = load_chunks()
    index = index_sentences(chunks)
    reachable = reachable_keys(chunks, scope)
    for entry in index:
        entry['in_editorial_summary'] = entry['kind'] == 'editorial_summary'
    payload = {'corpus_sha256': hashlib.sha256((DATA / 'chunks.json').read_bytes()).hexdigest(),
               'candidate_scope': scope,
               'chunks': len(chunks), 'sentences': len(index),
               'editorial_summary_sentences': sum(e['in_editorial_summary'] for e in index),
               'reachable_by_selector': len(reachable), 'index': index}
    target = Path(path) if path else DATA / 'gold' / 'sentence-index.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return target, payload


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build the corpus sentence index.')
    parser.add_argument('--scope', choices=('summary_only', 'retrieved_sentences'),
                        default='summary_only', help='选择器候选范围，只影响 reachable_by_selector 的口径。')
    args = parser.parse_args()
    target, payload = dump_index(scope=args.scope)
    print(f"wrote {target}")
    print(f"chunks={payload['chunks']} sentences={payload['sentences']} "
          f"editorial={payload['editorial_summary_sentences']} "
          f"reachable={payload['reachable_by_selector']} scope={payload['candidate_scope']}")
    for entry in payload['index']:
        if entry['in_editorial_summary']:
            print(' ', entry['key'], '|', entry['doc_id'], '|', entry['text'])
