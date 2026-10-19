"""Fail closed: an authored dev benchmark can never authorize model release."""
import argparse
import json
from pathlib import Path


def reasons(report, evidence, max_wrong_rate=.01, min_coverage_gain=.05, max_p95_ms=50):
    failures=[]
    if report.get('split')!='test' or not evidence.get('independentlyReviewed'):
        failures.append('Independent reviewed test set required')
    if report.get('datasetSha256')!=evidence.get('frozenTestSha256'):
        failures.append('Frozen test hash mismatch')
    if not report.get('modelSha256') or report.get('modelSha256')!=evidence.get('modelSha256'):
        failures.append('Model evidence hash mismatch')
    if not report.get('manifestSha256') or report.get('manifestSha256')!=evidence.get('manifestSha256'):
        failures.append('Manifest evidence hash mismatch')
    baseline=report.get('modes',{}).get('rules',{})
    candidate=report.get('modes',{}).get('hybrid',{})
    if candidate.get('total',0)<200 or candidate.get('eligible',0)<100:
        failures.append('At least 200 test examples / 100 accepted ground truths required')
    candidate_rate=candidate.get('wrongAcceptanceRate')
    baseline_rate=baseline.get('wrongAcceptanceRate')
    if candidate_rate is None or baseline_rate is None or candidate_rate>min(max_wrong_rate,baseline_rate):
        failures.append('Wrong acceptance must be <= rules and <= 1%')
    if (candidate.get('correctCoverage') or 0)-(baseline.get('correctCoverage') or 0)<min_coverage_gain:
        failures.append('Correct coverage gain must be >= 5 percentage points')
    devices=evidence.get('devices',{})
    for platform in ('android','ios','ohos'):
        measurement=devices.get(platform,{})
        if not measurement.get('device') or not measurement.get('endToEndPassed') or measurement.get('samples',0)<100 or (measurement.get('p95Ms') is None or measurement['p95Ms']>max_p95_ms):
            failures.append(platform+': require real-device end-to-end proof, >=100 samples, P95 <=50ms')
    if not evidence.get('exportParityPassed') or not evidence.get('streamingReviewed'):
        failures.append('Three-format parity and streaming review required')
    return failures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path);parser.add_argument('evidence',type=Path)
    args=parser.parse_args()
    failures=reasons(json.loads(args.report.read_text()),json.loads(args.evidence.read_text()))
    print(json.dumps(dict(releaseAllowed=not failures,reasons=failures),ensure_ascii=False,indent=2))
    raise SystemExit(1 if failures else 0)

if __name__=='__main__':
    main()
