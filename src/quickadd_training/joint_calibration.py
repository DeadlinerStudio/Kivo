"""Independent monotone calibration and finite-grid risk bounds; no test tuning."""
import math
import numpy as np
import torch

from .joint_decode import apply_gate
from .joint_heads import exact

GRID = (.90, .95, .98, .99, .995, .999)


def upper_bound(wrong, accepted, alpha=.05):
    if not accepted or wrong == accepted:
        return 1.
    if wrong == 0:
        return 1 - alpha ** (1 / accepted)
    low, high = wrong / accepted, 1.
    for _ in range(55):
        p = (low + high) / 2
        logs = [math.lgamma(accepted + 1) - math.lgamma(k + 1) - math.lgamma(accepted - k + 1)
                + k * math.log(p) + (accepted - k) * math.log1p(-p) for k in range(wrong + 1)]
        maximum = max(logs)
        cdf = math.exp(maximum) * sum(math.exp(v - maximum) for v in logs)
        if cdf > alpha:
            low = p
        else:
            high = p
    return high


def fit(rows, selected, logits):
    result = {}
    for kind in ['TASK', 'HABIT']:
        ids = [i for i, v in enumerate(selected) if v['proposal']['kind'] == kind]
        z = torch.tensor(np.asarray(logits)[ids], dtype=torch.float64)
        y = torch.tensor([float(exact(rows[i], selected[i])) for i in ids], dtype=torch.float64)
        if not ids or y.min() == y.max():
            result[kind] = dict(available=False, reason='insufficient-label-diversity', samples=len(ids))
            continue
        a, b = torch.tensor(.5413248546, dtype=torch.float64, requires_grad=True), torch.tensor(0., dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([a, b], max_iter=150, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.nn.functional.softplus(a) * z + b, y)
            loss.backward()
            return loss
        optimizer.step(closure)
        slope, bias = float(torch.nn.functional.softplus(a).detach()), float(b.detach())
        result[kind] = dict(available=True, slope=slope, bias=bias, samples=len(ids), positives=int(y.sum()),
                            NLL=float(torch.nn.functional.binary_cross_entropy_with_logits(slope * z + bias, y)))
    return result


def confidence(selected, logits, calibration):
    values = []
    for value, logit in zip(selected, logits):
        setting = calibration[value['proposal']['kind']]
        values.append(1 / (1 + math.exp(-max(-700, min(700, setting['slope'] * float(logit) + setting['bias']))))
                      if setting['available'] else None)
    return values


def evaluate(rows, values, alpha=.025):
    result = {}
    for kind in ['TASK', 'HABIT']:
        eligible = sum(r['expected']['decision'] == 'accept' and r['expected']['proposal']['kind'] == kind for r in rows)
        predicted = [(r, v) for r, v in zip(rows, values) if v['proposal']['kind'] == kind]
        accepted = [(r, v) for r, v in predicted if v['proposal']['ready'] and not v['proposal']['needsAI']]
        ready = [(r, v) for r, v in predicted if v['proposal']['ready']]
        correct = sum(exact(r, v) for r, v in accepted)
        ready_correct = sum(exact(r, v) for r, v in ready)
        wrong = len(accepted) - correct
        result[kind] = dict(eligible=eligible, accepted=len(accepted), correct=correct, wrongAccepted=wrong,
                            correctCoverage=correct / eligible if eligible else None,
                            wrongAcceptanceRate=wrong / len(accepted) if accepted else None,
                            wrongAcceptanceUpper=upper_bound(wrong, len(accepted), alpha), alpha=alpha,
                            negativeWrongAccepted=sum(r['expected']['decision'] == 'defer' for r, _ in accepted),
                            ready=len(ready), readyCorrect=ready_correct, readyWrong=len(ready) - ready_correct,
                            readyCorrectCoverage=ready_correct / eligible if eligible else None,
                            readyWithAI=sum(v['proposal']['needsAI'] for _, v in ready))
    return result


def select(rows, selected, probabilities):
    grid, choices = [], {}
    for threshold in GRID:
        values = [apply_gate(v, c, threshold) for v, c in zip(selected, probabilities)]
        metrics = evaluate(rows, values, alpha=.05 / 12)
        for kind, m in metrics.items():
            grid.append(dict(kind=kind, threshold=threshold, **m))
    for kind in ['TASK', 'HABIT']:
        feasible = [v for v in grid if v['kind'] == kind and v['accepted'] and v['correctCoverage'] >= .7
                    and v['wrongAcceptanceUpper'] <= .005]
        best = max(feasible, key=lambda v: (v['correctCoverage'], -v['wrongAccepted'], v['threshold'])) if feasible else None
        choices[kind] = dict(threshold=best['threshold'] if best else None, screeningTargetMet=bool(best))
    return dict(choices=choices, grid=grid, formalGatePassed=False,
                reason='Synthetic screening only; no independent human gold. No feasible cutoff means keep AI visible, never lower threshold on test.')


def reliability(rows, values, probabilities):
    report = {}
    for kind in ['TASK', 'HABIT']:
        examples = [(float(p), float(exact(r, v))) for r, v, p in zip(rows, values, probabilities)
                    if v['proposal']['kind'] == kind and p is not None]
        bins = []
        for index in range(10):
            points = [(p, y) for p, y in examples if index / 10 <= p <= 1 and (p < (index + 1) / 10 or index == 9)]
            bins.append(dict(lower=index / 10, count=len(points), meanConfidence=float(np.mean([p for p, _ in points])) if points else None,
                             accuracy=float(np.mean([y for _, y in points])) if points else None))
        report[kind] = dict(samples=len(examples), bins=bins,
                            Brier=float(np.mean([(p - y) ** 2 for p, y in examples])) if examples else None)
    return report
