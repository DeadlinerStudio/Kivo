"""Interactive copy-normalization -> fixed rules -> production KMP prototype."""
import argparse
import datetime as dt
import json
from pathlib import Path
import tempfile

import torch
from transformers import AutoConfig, AutoTokenizer

from .data import digest
from .normalizer_model import Normalizer, predict
from .normalizer_rules import parse_normalized
from .normalizer_fidelity import guard_normalized, policy_sha256
from .span_evaluate import normalize_drafts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    parser.add_argument('--model', type=Path, default=Path('artifacts/zh-normalizer-v1'))
    parser.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    parser.add_argument('--now', default=None)
    parser.add_argument('--timezone', default='Asia/Shanghai')
    parser.add_argument('--fidelity-settings', type=Path,
                        help='Enable the source fidelity guard with its separate dev-calibrated prototype.json')
    args = parser.parse_args()
    torch.set_num_threads(3)
    settings = json.loads((args.fidelity_settings or args.model / 'prototype.json').read_text())
    if digest(args.model / 'normalizer.pt') != settings['modelSha256']:
        raise ValueError('Normalization checkpoint differs from calibrated model')
    if args.fidelity_settings:
        if (settings.get('policySha256') != policy_sha256() or
                settings.get('fidelitySourceSha256') != digest(Path(__file__).with_name('normalizer_fidelity.py'))):
            raise ValueError('Fidelity policy differs from calibrated policy')
    row = dict(id='normalizer-interactive', text=args.text, preferredKind='TASK',
               now=args.now or dt.datetime.now(dt.timezone.utc).isoformat(), timezone=args.timezone)
    moment = dt.datetime.fromisoformat(row['now'].replace('Z', '+00:00'))
    if moment.tzinfo is None:
        parser.error('--now must have UTC offset')
    from zoneinfo import ZoneInfo
    ZoneInfo(args.timezone)
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'tokenizer', local_files_only=True)
    if not args.text.strip() or len(args.text) > 126 or len(tokenizer(args.text)['input_ids']) > 96:
        print(json.dumps(dict(state='CONFIRM', rawText=args.text, needsConfirmation=True, reasons=['empty-or-context-exceeded']), ensure_ascii=False))
        return
    network = Normalizer(AutoConfig.from_pretrained(args.model / 'encoder-config', local_files_only=True))
    network.load_state_dict(torch.load(args.model / 'normalizer.pt', weights_only=True))
    prediction = predict(network, tokenizer, [row], batch_size=1)[0]
    draft = parse_normalized(row, prediction, settings['minimumProbability'])
    if args.fidelity_settings:
        draft = guard_normalized(row, prediction, draft, tokenizer)
    with tempfile.TemporaryDirectory(prefix='quickadd-normalizer-') as temporary:
        result = normalize_drafts((args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve(),
                                  [row], [draft], Path(temporary))[0]
    print(json.dumps(dict(normalization=prediction, result=result,
                          fidelityEnabled=bool(args.fidelity_settings),
                          devQualityTargetMet=settings.get('devQualityTargetMet', False)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
