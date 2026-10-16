"""Build the span-level gold file.

Annotations are written against summary-sentence LABELS, never against hashes or raw
ids, and the builder resolves each label to a corpus sentence by unique substring.
A label that does not resolve to exactly one sentence is a hard error, so the gold
set cannot silently drift from the corpus.

Protocol (also stored in the artefact):
* core       = sentence(s) that carry the fact the question asks for.
* acceptable = core plus sentences that may legitimately be shown for this question.
* Gold is a subset of the frozen corpus by construction: no free text is authored here.
* Annotated by an AI assistant against the corpus text, NOT reviewed by a clinician.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from evidence_index import DATA, dump_index, load_chunks, normalize, reachable_keys, split_sentences

LABELS = [
    ('F1', 'food', '预防食源性疾病的四个步骤'),
    ('F2', 'food', '处理生肉与即食食品应使用不同的砧板'),
    ('F3', 'food', '易腐食品通常应在两小时内冷藏'),
    ('F4', 'food', '冰箱冷藏室应保持'),
    ('O1', 'oral', '成人应每天使用含氟牙膏'),
    ('O2', 'oral', '也应至少每年看一次牙医'),
    ('O3', 'oral', '尽量减少添加糖'),
    ('P1', 'prediabetes', '糖尿病前期可通过血糖检查发现'),
    ('P2', 'prediabetes', '减轻约5%到7%的体重'),
    ('P3', 'prediabetes', '这不是个人诊断或用药建议'),
    ('C1', 'co', '一氧化碳是无色无味的有毒气体'),
    ('C2', 'co', '不要在室内、地下室或车库使用便携发电机'),
    ('C3', 'co', '至少20英尺'),
    ('H1', 'heat', '保持凉爽、保持水分'),
    ('H2', 'heat', '不要只依赖电风扇'),
    ('H3', 'heat', '尽量避开最热时段'),
    ('H4', 'heat', '具体饮水限制应咨询医疗专业人员'),
    ('S1', 'smoke', '不存在安全量'),
    ('S2', 'smoke', '都不能消除二手烟'),
    ('S3', 'smoke', '应让室内和车辆完全无烟'),
]

# suite -> {case id: (core labels, acceptable labels, expected doc or docs)}
ANNOTATIONS = {
    'dev22': {
        '01': (['P2'], ['P2'], 'prediabetes'),
        '02': (['P2'], ['P2'], 'prediabetes'),
        '03': (['P2'], ['P2'], 'prediabetes'),
        '04': (['P2'], ['P2'], 'prediabetes'),
        '05': (['P2'], ['P2', 'P3'], 'prediabetes'),
        '06': (['O1'], ['O1'], 'oral'),
        '07': (['O2'], ['O2'], 'oral'),
        '08': (['O2'], ['O2'], 'oral'),
        '09': (['F3'], ['F3'], 'food'),
        '10': (['F2'], ['F2'], 'food'),
        '11': (['C2'], ['C2'], 'co'),
        '12': (['C3'], ['C3'], 'co'),
        '13': (['S2'], ['S1', 'S2'], 'smoke'),
        '14': (['H3'], ['H3'], 'heat'),
        '21': (['O1', 'C3'], ['O1', 'C3'], ['oral', 'co']),
        '22': (['P2'], ['P2'], 'prediabetes'),
    },
    'frozen16': {
        'a09': (['C3'], ['C3'], 'co'),
        'a10': (['C2'], ['C2'], 'co'),
        'a11': (['F2'], ['F2'], 'food'),
        'a12': (['F3'], ['F3'], 'food'),
        'a13': (['O1', 'O3'], ['O1', 'O3'], 'oral'),
        'a14': (['P2'], ['P2'], 'prediabetes'),
        'a15': (['S2'], ['S1', 'S2'], 'smoke'),
        'a16': (['F1'], ['F1'], 'food'),
    },
    'regression16': {
        'n01': (['O1'], ['O1'], 'oral'),
        'n02': (['O2'], ['O2'], 'oral'),
        'n03': (['F2'], ['F2'], 'food'),
        'n04': (['F3'], ['F3'], 'food'),
        'n05': (['P2'], ['P2'], 'prediabetes'),
        'n06': (['C2'], ['C2'], 'co'),
        'n07': (['C3'], ['C3'], 'co'),
        'n08': (['S2'], ['S1', 'S2'], 'smoke'),
    },
    # 扩充集：问题只写一次，标签仍全部取自冻结语料句子；本题集在写完之后才第一次跑评测。
    'expansion': {
        'e01': (['F1'], ['F1'], 'food'),
        'e02': (['F2'], ['F2'], 'food'),
        'e03': (['F3'], ['F3'], 'food'),
        'e04': (['F3'], ['F3'], 'food'),
        'e05': (['F4'], ['F4'], 'food'),
        'e06': (['F2', 'F3'], ['F2', 'F3'], 'food'),
        'e07': (['F1'], ['F1', 'F2', 'F3'], 'food'),
        'e08': (['F3', 'F4'], ['F3', 'F4'], 'food'),
        'e09': (['O1'], ['O1'], 'oral'),
        'e10': (['O1'], ['O1'], 'oral'),
        'e11': (['O2'], ['O2'], 'oral'),
        'e12': (['O2'], ['O2'], 'oral'),
        'e13': (['O3'], ['O3'], 'oral'),
        'e14': (['O1', 'O3'], ['O1', 'O3'], 'oral'),
        'e15': (['O1', 'O3'], ['O1', 'O2', 'O3'], 'oral'),
        'e16': (['P1'], ['P1'], 'prediabetes'),
        'e17': (['P1'], ['P1'], 'prediabetes'),
        'e18': (['P2'], ['P2'], 'prediabetes'),
        'e19': (['P2'], ['P2'], 'prediabetes'),
        'e20': (['P2'], ['P1', 'P2'], 'prediabetes'),
        'e21': (['P3'], ['P3'], 'prediabetes'),
        'e22': (['P1', 'P2'], ['P1', 'P2'], 'prediabetes'),
        'e23': (['C1'], ['C1'], 'co'),
        'e24': (['C3'], ['C3'], 'co'),
        'e25': (['C1', 'C2'], ['C1', 'C2'], 'co'),
        'e26': (['C2', 'C3'], ['C2', 'C3'], 'co'),
        'e27': (['C1'], ['C1'], 'co'),
        'e28': (['H1'], ['H1'], 'heat'),
        'e29': (['H2'], ['H2'], 'heat'),
        'e30': (['H3'], ['H3'], 'heat'),
        'e31': (['H4'], ['H4'], 'heat'),
        'e32': (['H1', 'H2', 'H3'], ['H1', 'H2', 'H3', 'H4'], 'heat'),
        'e33': (['H2'], ['H2'], 'heat'),
        'e34': (['H1'], ['H1'], 'heat'),
        'e35': (['S1'], ['S1'], 'smoke'),
        'e36': (['S2'], ['S1', 'S2'], 'smoke'),
        'e37': (['S3'], ['S3'], 'smoke'),
        'e38': (['S2'], ['S2'], 'smoke'),
    },
}

SUITES = ('dev22', 'frozen16', 'regression16', 'expansion')

# 扩充集的问题清单随仓库分发，第三方不必依赖任何运行产物就能重建金标准。
CASE_SOURCES = {
    'dev22': DATA / 'user-acceptance-llamacpp' / 'results.json',
    'frozen16': DATA / 'alignment-final-frozen.json',
    'regression16': DATA / 'final-health-v2-frozen.json',
    'expansion': Path(__file__).resolve().parent / 'evidence_expansion_cases.json',
}


def resolve_labels(index):
    """label -> {key, text, doc_id, chunk_id}; unique substring match is enforced."""
    summary = [e for e in index if e['in_editorial_summary']]
    resolved, used = {}, set()
    for label, doc, needle in LABELS:
        matches = [e for e in summary if e['doc_id'] == doc and needle in e['text']]
        if len(matches) != 1:
            raise SystemExit(f'label {label}: expected 1 match for {needle!r}, got {len(matches)}')
        if matches[0]['key'] in used:
            raise SystemExit(f'label {label}: sentence already claimed by another label')
        used.add(matches[0]['key'])
        resolved[label] = matches[0]
    return resolved


def load_cases(suite):
    raw = json.loads(CASE_SOURCES[suite].read_text(encoding='utf-8'))
    rows = raw['rows'] if isinstance(raw, dict) and 'rows' in raw else raw
    out = {}
    for row in rows:
        case = row['case'] if 'case' in row else row
        out[case['id']] = case
    return out


def case_kind(suite, ident, case):
    """answer / refuse；既没标注又不带 refuse 标记的案例直接报错，不允许静默跳过。"""
    if ident in ANNOTATIONS[suite]:
        return 'answer'
    if case.get('refuse'):
        return 'refuse'
    raise SystemExit(f'{suite}/{ident}: 没有标注也不带 refuse 标记，会被静默跳过，拒绝构建金标准。')


def build():
    _, payload = dump_index(scope='retrieved_sentences')
    index, resolved = payload['index'], resolve_labels(payload['index'])
    doc_keys = {}
    for entry in index:
        doc_keys.setdefault(entry['doc_id'], []).append(entry['key'])
    cases, counts = [], {'answer': 0, 'refuse': 0}
    for suite in SUITES:
        source = load_cases(suite)
        for ident in sorted(source):
            case = source[ident]
            if case_kind(suite, ident, case) == 'refuse':
                counts['refuse'] += 1
                cases.append({'suite': suite, 'id': ident, 'question': case['q'],
                              'expected': 'refuse', 'core_labels': [], 'acceptable_labels': [],
                              'core': [], 'acceptable': [], 'doc_scope': [],
                              'expected_doc': None, 'previous': case.get('previous')})
                continue
            core_labels, acceptable_labels, doc = ANNOTATIONS[suite][ident]
            assert set(core_labels) <= set(acceptable_labels), f'{suite}/{ident}: core not subset'
            docs = doc if isinstance(doc, list) else [doc]
            scope = sorted({k for d in docs for k in doc_keys.get(d, [])})
            counts['answer'] += 1
            cases.append({'suite': suite, 'id': ident, 'question': case['q'], 'expected': 'answer',
                          'core_labels': core_labels, 'acceptable_labels': acceptable_labels,
                          'core': [resolved[l]['key'] for l in core_labels],
                          'acceptable': [resolved[l]['key'] for l in acceptable_labels],
                          'doc_scope': scope, 'expected_doc': doc,
                          'previous': case.get('previous')})
    reachable_narrow = len(reachable_keys(load_chunks(), 'summary_only'))
    answers = [c for c in cases if c['expected'] == 'answer']
    core_sizes = Counter(len(c['core']) for c in answers)
    gold = {
        'name': 'evidence-gold-v3',
        'supersedes': 'evidence-gold-v2',
        'corpus_sha256': payload['corpus_sha256'],
        'sentence_index': {'chunks': payload['chunks'], 'sentences': payload['sentences'],
                           'editorial_summary_sentences': payload['editorial_summary_sentences'],
                           'reachable_summary_only': reachable_narrow,
                           'reachable_retrieved_sentences': payload['reachable_by_selector']},
        'protocol': {
            'core': '句子直接包含回答问题所需事实（召回分母）；一题需要几个事实就标几句，不强行凑成一句',
            'acceptable': '允许展示且不算答非所问的句子，core 的超集（严格精确率分母）',
            'doc_scope': '答案文档的全部句子（宽松精确率分母）；v1 只算了该文档的中文导读句',
            'padding': '选中但在 acceptable 内、仍在 doc_scope 内 = 同文档冗余引用',
            'off_topic': '选中但不在 doc_scope 内 = 引到了别的文档',
            'gold_source': '只从冻结语料句子中挑选，builder 校验每个标签唯一命中；未撰写任何新文本',
            'annotator': 'AI 助手依据语料原文标注，未经临床专家复核；判据可逐句审计',
            'why_v2': 'v1 的候选只来自 top-1 文档的导读句，因此 lenient_precision 恒为 1.0——那是候选集'
                      '构造出来的恒等式而不是模型能力；同时旧切分器不切英文句点，英文段落大多整块成一句。'
                      'v2 修正切分并把 doc_scope 扩到答案文档全部句子，core/acceptable 保持 v1 的导读句不变，'
                      '因此召回与 exact 仍可与 v1 直接比较。',
            'why_v3': '扩充集（suite=expansion）的 56 道题在写完后才第一次跑评测，因此可以用它检验'
                      '在开发回归集上得到的结论；问题清单随仓库分发（evidence_expansion_cases.json），'
                      '不依赖任何运行产物即可重建。core 允许由 2~3 句事实单元组成，'
                      '以改善"core 常为 1 句、召回只能取 0/0.5/1"的分辨率问题。'
                      '本版同时把"未标注又不带 refuse 标记"的案例从静默跳过改为构建期报错。',
        },
        'label_map': {k: {'key': v['key'], 'doc_id': v['doc_id'], 'text': v['text']} for k, v in resolved.items()},
        'counts': {'cases': len(cases), **counts,
                   'core_sentences_per_answer_case': {str(k): v for k, v in sorted(core_sizes.items())},
                   'answer_cases_needing_multiple_sentences': sum(v for k, v in core_sizes.items() if k > 1)},
        'cases': cases,
    }
    target = DATA / 'gold' / 'evidence-gold-v3.json'
    target.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding='utf-8')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return target, gold, digest


if __name__ == '__main__':
    target, gold, digest = build()
    print(f'wrote {target}')
    print(f"sha256 {digest}")
    print(f"cases={gold['counts']['cases']} answer={gold['counts']['answer']} refuse={gold['counts']['refuse']}")
    print(f"core 句数分布={gold['counts']['core_sentences_per_answer_case']} "
          f"需要多句事实的题数={gold['counts']['answer_cases_needing_multiple_sentences']}")
    for suite in SUITES:
        sub = [c for c in gold['cases'] if c['suite'] == suite]
        print(f"  {suite:14s} {len(sub):3d} case  answer={sum(c['expected']=='answer' for c in sub)} "
              f"refuse={sum(c['expected']=='refuse' for c in sub)}")
    print('label 覆盖检查：', 'OK' if len(gold['label_map']) == len(LABELS) else 'FAIL')
