"""Span-level evaluation runner: real HTTP, real model, versioned outputs.

Scoring uses the service's own artefacts, not a re-implementation:
``selection_options`` gives the candidate set the selector actually saw, and
``selection_raw`` gives the JSON it actually returned, so a case can be separated
into "gold was never offered" (retrieval/architecture) vs "gold offered but not
chosen" (selection).

Usage:
    python run_span_eval.py --backend llamacpp --modes rag,sft_rag --tag v1
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx

from evidence_index import normalize
from reranker import RERANKERS
from span_metrics import aggregate, candidate_reachability, selection_metrics
from workspace import RAG as DATA

GOLD = DATA / 'gold' / 'evidence-gold-v3.json'
BASE_URL = 'http://127.0.0.1:7860'


def sentence_key(chunk_id, text):
    digest = hashlib.sha1(normalize(text).encode('utf-8')).hexdigest()[:16]
    return f'{chunk_id}:{digest}'


def options_to_keys(result):
    """option number -> stable key, using the retrieved chunks for provenance.

    ``selection_options`` only carries ``chunk_index`` (an index into the result's
    ``retrieved`` list), so the chunk id has to come from there.
    """
    retrieved = result.get('retrieved') or []
    mapping, keys = {}, []
    for opt in result.get('selection_options') or []:
        index = int(opt['chunk_index'])
        chunk_id = retrieved[index]['id'] if 0 <= index < len(retrieved) else 'unknown'
        key = sentence_key(chunk_id, opt['text'])
        mapping[int(opt['option'])] = key
        keys.append(key)
    return mapping, keys


def parse_selection_ids(raw):
    text = (raw or '').strip()
    if text.startswith('```'):
        text = text.strip('`')
        text = text.split('\n', 1)[1] if '\n' in text else text
    try:
        ids = json.loads(text.strip())['sentences']
    except Exception:
        return []
    return [int(n) for n in ids if isinstance(n, int)]


def selected_keys(result):
    """Selected sentence keys for one result (recursing into multi-part answers)."""
    mode = result.get('delivery_mode')
    if mode == 'multi_part_evidence':
        keys, offered = [], []
        for sub in result.get('subresults', []):
            sub_keys, sub_offered = selected_keys(sub)
            keys += sub_keys
            offered += sub_offered
        return keys, offered
    mapping, offered = options_to_keys(result)
    if result.get('abstained') or mode in ('evidence_selection_rejected', 'cancelled'):
        return [], offered
    return [mapping[n] for n in parse_selection_ids(result.get('selection_raw')) if n in mapping], offered


def wait_job(client, job_id, timeout=240):
    started = time.monotonic()
    while True:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] != 'running':
            return job
        if time.monotonic() - started > timeout:
            raise TimeoutError(f'job {job_id} exceeded {timeout}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=['llamacpp', 'transformers'])
    ap.add_argument('--candidate-scope', required=True, choices=['summary_only', 'retrieved_sentences'],
                    help='必须与服务实际使用的候选范围一致，否则拒绝运行。')
    ap.add_argument('--reranker', required=True, choices=list(RERANKERS),
                    help='必须与服务实际使用的重排序器一致，否则拒绝运行。')
    ap.add_argument('--modes', default='rag,sft_rag')
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--allow-errors', action='store_true')
    args = ap.parse_args()
    modes = [m for m in args.modes.split(',') if m]

    gold_bytes = GOLD.read_bytes()
    gold = json.loads(gold_bytes)
    client = httpx.Client(base_url=BASE_URL, timeout=120)
    status = client.get('/api/status').json()
    if status.get('busy'):
        raise SystemExit('服务正忙，未运行评测（不干扰进行中的任务）。')
    if status.get('generator_backend') != args.backend:
        raise SystemExit(f"服务后端是 {status.get('generator_backend')}，与 --backend {args.backend} 不符，拒绝误标。")
    if status.get('candidate_scope') != args.candidate_scope:
        raise SystemExit(f"服务候选范围是 {status.get('candidate_scope')}，与 --candidate-scope "
                         f"{args.candidate_scope} 不符，拒绝误标。")
    if status.get('reranker') != args.reranker:
        raise SystemExit(f"服务重排序器是 {status.get('reranker')}，与 --reranker {args.reranker} 不符，拒绝误标。")

    started = time.strftime('%Y-%m-%dT%H:%M:%S')
    rows, summaries = [], {}
    for mode in modes:
        cases, sessions = [], {}
        for case in gold['cases']:
            row = {'suite': case['suite'], 'id': case['id'], 'question': case['question'],
                   'mode': mode, 'expected': case['expected'], 'error': None,
                   'followup_of': case.get('previous')}
            try:
                payload = {'prompt': case['question'], 'mode': mode}
                # Follow-ups are asked inside the session that produced their parent,
                # otherwise the history rewrite cannot fire and the case is measured wrongly.
                if case.get('previous'):
                    if case['previous'] not in sessions:
                        raise RuntimeError(f"parent case {case['previous']} has no session")
                    payload['session_id'] = sessions[case['previous']]
                created = client.post('/api/jobs', json=payload).json()
                sessions[case['id']] = created['session_id']
                job = wait_job(client, created['id'])
                if job['state'] == 'error':
                    raise RuntimeError(job.get('error') or 'unknown service error')
                result = job['results'][0]
                selected, offered = selected_keys(result)
                reach = candidate_reachability(case['core'], offered)
                metrics = (selection_metrics(case['core'], case['acceptable'], selected,
                                            case['doc_scope'])
                           if case['expected'] == 'answer' and not result.get('abstained') else None)
                row.update(result=result, selected=selected, offered=offered, reachability=reach,
                           metrics=metrics, state=job['state'], session_id=created['session_id'])
                cases.append({'expected': case['expected'], 'abstained': bool(result.get('abstained')),
                              'core': case['core'], 'acceptable': case['acceptable'],
                              'doc_scope': case['doc_scope'], 'selected': selected, 'error': False,
                              'id': case['id'], 'reachable': reach['reachable']})
            except Exception as exc:
                row.update(error=f'{type(exc).__name__}: {exc}')
                cases.append({'expected': case['expected'], 'abstained': False, 'core': case['core'],
                              'acceptable': case['acceptable'], 'doc_scope': case['doc_scope'],
                              'selected': [], 'error': True, 'id': case['id'], 'reachable': None})
            rows.append(row)
        block = aggregate(cases)
        unreachable = [r['id'] for r in rows if r['mode'] == mode and r.get('reachability')
                       and not r['reachability']['reachable'] and r['expected'] == 'answer']
        block['gold_never_offered'] = unreachable
        summaries[mode] = block

    slug = f"{args.backend}-{args.candidate_scope}-{args.reranker}"
    artefact = {'tag': args.tag, 'backend': args.backend, 'candidate_scope': args.candidate_scope,
                'reranker': args.reranker,
                'modes': modes, 'started': started,
                'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'gold_sha256': hashlib.sha256(gold_bytes).hexdigest(),
                'gold_name': gold['name'], 'corpus_sha256': gold['corpus_sha256'],
                'status': status, 'summaries': summaries, 'rows': rows}
    target = DATA / f"span-eval-{slug}-{args.tag}.json"
    target.write_text(json.dumps(artefact, ensure_ascii=False, indent=2), encoding='utf-8')
    summary_target = DATA / f"span-eval-{slug}-{args.tag}-summary.json"
    summary_target.write_text(json.dumps({k: v for k, v in artefact.items() if k != 'rows'},
                                        ensure_ascii=False, indent=2), encoding='utf-8')
    errors = sum(1 for r in rows if r['error'])
    for mode, block in summaries.items():
        print(f"{args.backend}/{mode}: P={block['precision']} Plen={block['lenient_precision']} "
              f"R={block['recall']} F1={block['f1']} exact={block['exact']}/{block['answered_with_evidence']} "
              f"over-refusal={block['over_refusal']}/{block['answerable']} "
              f"under-refusal={block['under_refusal']}/{block['refusable']} "
              f"padding={block['padding_sentences']} off-topic={block['off_topic_sentences']} "
              f"errors={block['errors']}")
    print(f'wrote {target} (errors={errors})')
    if errors and not args.allow_errors:
        raise SystemExit(f'{errors} case(s) errored; artefacts written, treat rates as incomplete.')


if __name__ == '__main__':
    main()
