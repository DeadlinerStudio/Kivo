"""Selective full-proposal metrics; no partial-field successes count as correct."""
from .data import FIELDS


def accepted(proposal):
    return bool(proposal and proposal.get('ready') and not proposal.get('needsAI'))


def correct(expected, proposal):
    return expected['decision'] == 'accept' and accepted(proposal) and all(
        proposal.get(key) == expected['proposal'][key] for key in FIELDS)


def summarize(rows, predictions, mode):
    total = len(rows)
    eligible = sum(r['expected']['decision'] == 'accept' for r in rows)
    accepted_count = wrong = right = negative_wrong = 0
    failures, latencies, sources = [], [], {}
    for row in rows:
        prediction = predictions[row['id']]
        proposal = prediction[mode]
        is_accepted = accepted(proposal)
        is_correct = correct(row['expected'], proposal)
        accepted_count += is_accepted
        right += is_correct
        wrong += is_accepted and not is_correct
        negative_wrong += is_accepted and row['expected']['decision'] == 'defer'
        sources[proposal.get('source', 'DEFER') if proposal else 'DEFER'] = sources.get(proposal.get('source', 'DEFER') if proposal else 'DEFER', 0) + 1
        latencies.append(prediction[mode + 'Ms'])
        if is_accepted and not is_correct or row['expected']['decision'] == 'accept' and not is_correct:
            failures.append(dict(id=row['id'], text=row['text'], expected=row['expected'], actual=proposal))
    ordered = sorted(latencies)
    return dict(total=total, eligible=eligible, accepted=accepted_count, correct=right,
                wrongAccepted=wrong, negativeWrongAccepted=negative_wrong,
                wrongAcceptanceRate=wrong / accepted_count if accepted_count else None,
                correctCoverage=right / eligible if eligible else None,
                deferredRate=(total - accepted_count) / total if total else None,
                p95Ms=ordered[max(0, __import__('math').ceil(.95 * len(ordered)) - 1)] if ordered else None,
                sources=sources, failures=failures)


def streaming(rows, predictions, mode):
    by_parent = {}
    for row in rows:
        if 'parentId' in row:
            by_parent.setdefault(row['parentId'], []).append(row)
    changes = early_accepts = 0
    for stream in by_parent.values():
        previous = None
        for row in sorted(stream, key=lambda r: r['prefixLength']):
            proposal = predictions[row['id']][mode]
            current = tuple(proposal.get(k) for k in FIELDS) if accepted(proposal) else None
            changes += previous is not None and current != previous
            early_accepts += current is not None and not row['isFinal']
            previous = current
    return dict(streams=len(by_parent), acceptedProposalChanges=changes, acceptedBeforeFinal=early_accepts,
                note='Early acceptance is descriptive, not necessarily wrong: some prefixes are complete requests.')
