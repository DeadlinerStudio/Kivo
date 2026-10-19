"""Interactive offline shared-BERT TASK/HABIT prototype; no save or cloud calls."""
import argparse
import datetime as dt
import json
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import torch
from transformers import AutoConfig, AutoTokenizer

from .data import digest
from .joint_model import JointModel, predict
from .joint_adapter import network
from .joint_heads import load_heads, prepare, choose, quality_logits
from .joint_native import NativeFields
from .joint_calibration import confidence
from .joint_decode import apply_gate
from .joint_evaluate import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    parser.add_argument('--prototype', type=Path, required=True)
    parser.add_argument('--kind', choices=['TASK', 'HABIT'], default='TASK')
    parser.add_argument('--now')
    parser.add_argument('--timezone', default='Asia/Shanghai')
    parser.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    args = parser.parse_args()
    torch.set_num_threads(3)
    config = json.loads(args.prototype.read_text())
    directory, head_dir = Path(config['modelDirectory']), args.prototype.parent
    binary = (args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve()
    if digest(directory / 'joint.pt') != config['modelSha256'] or digest(binary) != config['nativeSha256']:
        raise ValueError('Model/native version differs from calibrated prototype')
    for name, checksum in config['headSha256'].items():
        if digest(head_dir / name) != checksum:
            raise ValueError('Calibrated proposal head differs: ' + name)
    for name, checksum in config['policySources'].items():
        if digest(Path(__file__).parent / name) != checksum:
            raise ValueError('Calibrated feature/policy code differs: ' + name)
    for name, checksum in config['modelAssetSha256'].items():
        if digest(directory / name) != checksum:
            raise ValueError('Calibrated tokenizer/config differs: ' + name)
    row = dict(id='interactive-joint', text=args.text, preferredKind=args.kind,
               now=args.now or dt.datetime.now(dt.timezone.utc).isoformat(), timezone=args.timezone)
    if dt.datetime.fromisoformat(row['now'].replace('Z', '+00:00')).tzinfo is None:
        parser.error('--now must contain UTC offset')
    ZoneInfo(args.timezone)
    tokenizer = AutoTokenizer.from_pretrained(directory / 'tokenizer', local_files_only=True)
    if not args.text.strip() or len(args.text) > 126 or len(tokenizer(args.text)['input_ids']) > 96:
        print(json.dumps(dict(rawText=args.text, ready=False, needsAI=bool(args.text.strip()),
                              reasons=['empty-or-context-exceeded']), ensure_ascii=False))
        return
    model = network(AutoConfig.from_pretrained(directory / 'encoder-config', local_files_only=True))
    model.load_state_dict(torch.load(directory / 'joint.pt', weights_only=True))
    rankers, quality = load_heads(head_dir / 'rank.pt'), load_heads(head_dir / 'quality.pt')
    with tempfile.TemporaryDirectory(prefix='joint-fields-') as temporary:
        native = NativeFields(binary, temporary)
        selected = choose(prepare([row], predict(model, tokenizer, [row]), native), rankers)
    logits = quality_logits(quality, selected)
    probability = confidence(selected, logits, config['calibration'])[0]
    kind = selected[0]['proposal']['kind']
    result = apply_gate(selected[0], probability, config['thresholds'][kind]['threshold'])
    print(json.dumps(dict(result=plain(result), formalGatePassed=False,
                          scope='offline synthetic prototype; no automatic save'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
