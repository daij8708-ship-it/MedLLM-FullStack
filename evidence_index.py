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
SENTENCE_SPLIT = r'(?<=[。！？])\s*|\n+'


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


def reachable_keys(chunks):
    """Keys the extractive selector can ever offer.

    ``evidence_sentences`` appends the first chunk's sentences and then skips every
    later chunk unless it is an editorial summary, so in practice only the top-1
    document's chunk is selectable. Recorded here so the report can state the
    candidate universe instead of assuming the whole corpus is reachable.
    """
    reachable = set()
    for chunk in chunks:
        if chunk['metadata'].get('kind') == 'editorial_summary':
            for sentence in split_sentences(chunk['text']):
                if len(sentence.strip()) >= 8:
                    reachable.add(sentence_key(chunk['id'], sentence.strip()))
    return reachable


def dump_index(path=None):
    chunks = load_chunks()
    index = index_sentences(chunks)
    reachable = reachable_keys(chunks)
    for entry in index:
        entry['in_editorial_summary'] = entry['kind'] == 'editorial_summary'
    payload = {'corpus_sha256': hashlib.sha256((DATA / 'chunks.json').read_bytes()).hexdigest(),
               'chunks': len(chunks), 'sentences': len(index),
               'editorial_summary_sentences': sum(e['in_editorial_summary'] for e in index),
               'reachable_by_selector': len(reachable), 'index': index}
    target = Path(path) if path else DATA / 'gold' / 'sentence-index.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return target, payload


if __name__ == '__main__':
    target, payload = dump_index()
    print(f"wrote {target}")
    print(f"chunks={payload['chunks']} sentences={payload['sentences']} "
          f"editorial={payload['editorial_summary_sentences']} reachable={payload['reachable_by_selector']}")
    for entry in payload['index']:
        if entry['in_editorial_summary']:
            print(' ', entry['key'], '|', entry['doc_id'], '|', entry['text'])
