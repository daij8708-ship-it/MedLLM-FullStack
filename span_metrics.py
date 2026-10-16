"""Span-level evaluation metrics.

Definitions are deliberately explicit because every denominator is a claim:

* ``core``       sentences that carry the fact the question asks for (recall denominator).
* ``acceptable`` superset of ``core``: sentences that may be shown without being off-topic
                 (precision denominator). Anything selected outside it is unsupported padding.
* ``hit``        selected AND core  -> the answer actually rests on required evidence.
* ``supported``  selected AND acceptable -> not off-topic.
* ``unsupported`` selected minus acceptable -> padding the model added.

An abstention is scored in the confusion matrix, never mixed into selection metrics.
Cases that error are excluded from every rate and reported separately.
"""
import math

ANSWER = 'answer'
REFUSE = 'refuse'


def wilson(successes, total, z=1.96):
    """95% Wilson score interval; with n of this size an interval matters more than a point."""
    if total <= 0:
        return (None, None)
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def selection_metrics(core, acceptable, selected, doc_scope=None):
    """Three tiers, because one precision number would hide the interesting part.

    * ``acceptable``  -> strict precision: only sentences that address the asked fact.
    * ``doc_scope``   -> lenient precision: any sentence of the answer document's summary.
    * ``off_topic``   -> selected outside ``doc_scope``: a genuinely wrong source.
    """
    core, acceptable, selected = set(core), set(acceptable), set(selected)
    scope = set(doc_scope) if doc_scope is not None else acceptable
    hit, supported = selected & core, selected & acceptable
    precision = len(supported) / len(selected) if selected else 0.0
    lenient = len(selected & scope) / len(selected) if selected else 0.0
    recall = len(hit) / len(core) if core else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'selected': len(selected), 'core': len(core), 'hit': len(hit),
            'padding': len(selected - acceptable), 'off_topic': len(selected - scope),
            'missed_core': len(core - selected),
            'precision': round(precision, 4), 'lenient_precision': round(lenient, 4),
            'recall': round(recall, 4), 'f1': round(f1, 4),
            'exact': bool(core) and core <= selected and not (selected - acceptable),
            'any_hit': bool(hit), 'full_recall': bool(core) and core <= selected}


def confusion(rows):
    """rows: [{expected: answer|refuse, abstained: bool}] -> counts and two error rates."""
    tp = sum(1 for r in rows if r['expected'] == REFUSE and r['abstained'])
    fp = sum(1 for r in rows if r['expected'] == ANSWER and r['abstained'])
    fn = sum(1 for r in rows if r['expected'] == REFUSE and not r['abstained'])
    tn = sum(1 for r in rows if r['expected'] == ANSWER and not r['abstained'])
    answerable, refusable = fp + tn, tp + fn
    return {'correct_refusal': tp, 'over_refusal': fp, 'under_refusal': fn,
            'answered_when_answerable': tn, 'refusable': refusable, 'answerable': answerable,
            'over_refusal_rate': round(fp / answerable, 4) if answerable else None,
            'over_refusal_ci': wilson(fp, answerable),
            'under_refusal_rate': round(fn / refusable, 4) if refusable else None,
            'under_refusal_ci': wilson(fn, refusable)}


def aggregate(cases):
    """cases: [{expected, abstained, core, acceptable, selected, error?}] -> one report block."""
    usable = [c for c in cases if not c.get('error')]
    errors = [c for c in cases if c.get('error')]
    answered = [c for c in usable if c['expected'] == ANSWER and not c['abstained']]
    metrics = [selection_metrics(c['core'], c['acceptable'], c['selected'], c.get('doc_scope'))
               for c in answered]
    n = len(metrics) or 1
    report = confusion(usable)
    report.update({
        'cases': len(cases), 'errors': len(errors),
        'answerable': report['answerable'], 'refusable': report['refusable'],
        'answered_with_evidence': len(metrics),
        'coverage': round(len(metrics) / report['answerable'], 4) if report['answerable'] else None,
        'precision': round(sum(m['precision'] for m in metrics) / n, 4),
        'lenient_precision': round(sum(m['lenient_precision'] for m in metrics) / n, 4),
        'recall': round(sum(m['recall'] for m in metrics) / n, 4),
        'f1': round(sum(m['f1'] for m in metrics) / n, 4),
        'exact': sum(m['exact'] for m in metrics), 'any_hit': sum(m['any_hit'] for m in metrics),
        'full_recall': sum(m['full_recall'] for m in metrics),
        'padding_sentences': sum(m['padding'] for m in metrics),
        'off_topic_sentences': sum(m['off_topic'] for m in metrics),
        'per_case': metrics})
    report['exact_ci'] = wilson(report['exact'], len(metrics))
    report['any_hit_ci'] = wilson(report['any_hit'], len(metrics))
    return report


def candidate_reachability(core, offered):
    """Did the candidate set the selector was shown even contain a core sentence?

    Separates a retrieval/architecture failure (gold never offered) from a selection
    failure (gold offered but not chosen).
    """
    core, offered = set(core), set(offered)
    return {'core_offered': len(core & offered) if core else 0,
            'reachable': bool(core) and bool(core & offered) if core else True}
