"""Raw-text production rule baseline, without gold or model fields in its input."""
import numpy as np

from .native_adapter import evaluate as native_evaluate
from .joint_calibration import evaluate
from .joint_diagnostics import diagnose


def baseline(rows, binary, directory):
    inputs = [{k: row[k] for k in ['id', 'text', 'preferredKind', 'now', 'timezone']} for row in rows]
    raw = native_evaluate(binary, None, inputs, directory)
    values = [dict(proposal=raw[r['id']]['rules'], hardReasons=[]) for r in rows]
    times = [raw[r['id']]['rulesMs'] for r in rows]
    return dict(metrics=evaluate(rows, values), diagnostics=diagnose(rows, values),
                hostRuleP95Milliseconds=float(np.percentile(times, 95)),
                scope='Unchanged rule acceptance policy on raw text, with current Core numeral/clock fixes'), values
