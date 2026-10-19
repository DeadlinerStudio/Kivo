"""Small counterfactual checks for the MVP review; no production code changes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from quickadd_training.data import digest
from quickadd_training.native_adapter import evaluate
from quickadd_training.gate import reasons


def run(core, output):
    output.mkdir(parents=True, exist_ok=True)
    model_dir = Path('artifacts/zh-v3')
    manifest = json.loads((model_dir / 'quick-add-model.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model_path = model_dir / 'quick-add-zh-v3.onnx'
    assert digest(model_path) == manifest['artifacts']['android']['sha256']
    runtime = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    titles = ['检查商品库存', '检查商品有没有库存', '退掉衣服', '退掉不合适的衣服',
              '确认租车油量', '确认租的车还有多少油', '记录原因', '记录取消原因']
    pairs = []
    for title in titles:
        text = '明天下午3点' + title
        encoded = tokenizer(text, padding='max_length', max_length=128, return_tensors='np')
        il, _ = runtime.run(None, {k: encoded[k].astype(np.int32) for k in ['input_ids', 'attention_mask']})
        p = np.exp(il[0] - il[0].max())
        p /= p.sum()
        pairs.append(dict(text=text, intentProbabilities=dict(zip(['TASK', 'HABIT', 'OTHER'], p.tolist()))))

    # Inject perfect TOKEN labels to test projection independently of learned errors.
    rows = []
    for i, time in enumerate(['明天下午 3 点', '明天下午  3  点']):
        text = time + '打电话'
        encoded = tokenizer(text, padding='max_length', max_length=128, return_offsets_mapping=True)
        sl = np.full((128, 9), -20.)
        sl[:, 0] = 20.
        for j, (a, b) in enumerate(encoded['offset_mapping']):
            if a == b:
                continue
            label = (3 if a == 0 else 4) if a < len(time) else (1 if a == len(time) else 2)
            sl[j] = -20.
            sl[j, label] = 20.
        rows.append(dict(id=f'space-{i}', text=text, preferredKind='TASK', now='2026-10-19T04:00:00Z',
                         timezone='Asia/Shanghai', inputIds=encoded['input_ids'], attentionMask=encoded['attention_mask'],
                         intent=[20., -20., -20.], slots=sl.reshape(-1).tolist()))
    binary = core / 'shared/build/bin/macosArm64/debugTest/test.kexe'
    result = evaluate(binary, manifest, rows, output / 'contracts-native')
    whitespace = dict(sameInputIds=rows[0]['inputIds'] == rows[1]['inputIds'],
                      sameTokenLogits=rows[0]['slots'] == rows[1]['slots'],
                      proposals=[dict(text=r['text'], model=result[r['id']]['model']) for r in rows])
    text = '明天下午3点给Zoë打电话'
    vocabulary = set(manifest['codec']['vocabulary'])
    unknown = dict(text=text, unknownCharacters=[c for c in text if c not in vocabulary],
                   characterUnknownRatio=sum(c not in vocabulary for c in text) / len(text),
                   wordPieceUnknownTokens=tokenizer(text)['input_ids'].count(tokenizer.unk_token_id))
    # Gate logic fixture only: this is not release evidence or a release decision.
    report = dict(split='test', datasetSha256='a', modelSha256='b', manifestSha256='c', modes=dict(
        rules=dict(wrongAcceptanceRate=.1, correctCoverage=0),
        hybrid=dict(total=300, eligible=200, wrongAcceptanceRate=0, correctCoverage=.06)))
    evidence = dict(independentlyReviewed=True, frozenTestSha256='a', modelSha256='b', manifestSha256='c',
                    exportParityPassed=True, streamingReviewed=True,
                    devices={p: dict(device='fixture', endToEndPassed=True, samples=100, p95Ms=20)
                             for p in ['android', 'ios', 'ohos']})
    output_data = dict(scriptSha256=digest(Path(__file__)), modelSha256=digest(model_path),
                       adapterBinarySha256=digest(binary), intentMinimalPairs=pairs,
                       perfectTokenWhitespace=whitespace, wordPieceVersusCharacterGate=unknown,
                       syntheticGateFixture=dict(coverage=.06, failureReasons=reasons(report, evidence)))
    (output / 'contracts.json').write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
    print(json.dumps(output_data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    parser.add_argument('--output', type=Path, default=Path('reports/mvp-review-2026-10-19'))
    args = parser.parse_args()
    run(args.core.resolve(), args.output.resolve())
