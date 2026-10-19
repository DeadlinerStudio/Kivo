"""Run the isolated ONNX core-field prototype and real KMP date normalization."""
import argparse
import datetime as dt
import json
from pathlib import Path
import tempfile

import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from .data import digest
from .span_decode import decode
from .span_evaluate import normalize_drafts
from .span_model import predict_onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    parser.add_argument('--model', type=Path, default=Path('artifacts/zh-core-spans-v1'))
    parser.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    parser.add_argument('--timezone', default='Asia/Shanghai')
    parser.add_argument('--now', default=None, help='ISO instant with UTC offset; default current UTC instant')
    parser.add_argument('--kind', choices=['TASK', 'HABIT'], default='TASK')
    args = parser.parse_args()
    torch.set_num_threads(3)
    config = json.loads((args.model / 'prototype.json').read_text())
    if digest(args.model / 'core-spans.onnx') != config['onnxSha256']:
        raise ValueError('ONNX hash differs from calibrated prototype configuration')
    row = dict(id='interactive', text=args.text, preferredKind=args.kind,
               now=args.now or dt.datetime.now(dt.timezone.utc).isoformat(), timezone=args.timezone)
    # Validation of clock/zone is delegated to standard libraries before any native call.
    moment = dt.datetime.fromisoformat(row['now'].replace('Z', '+00:00'))
    if moment.tzinfo is None:
        parser.error('--now must include a UTC offset')
    from zoneinfo import ZoneInfo
    ZoneInfo(args.timezone)
    if not args.text.strip():
        print(json.dumps(dict(state='CONFIRM', rawText=args.text, reasons=['empty-input'], needsConfirmation=True), ensure_ascii=False))
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'tokenizer', local_files_only=True)
    if len(args.text) > 126 or len(tokenizer(args.text)['input_ids']) > 96:
        print(json.dumps(dict(state='CONFIRM', rawText=args.text, reasons=['context-exceeded'], needsConfirmation=True), ensure_ascii=False))
        return
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    runtime = ort.InferenceSession(str(args.model / 'core-spans.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    prediction = predict_onnx(runtime, tokenizer, [row])[0]
    draft = decode(row, prediction, config['thresholds']['span'], config['thresholds']['request'])
    with tempfile.TemporaryDirectory(prefix='quickadd-span-') as temporary:
        result = normalize_drafts((args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve(),
                                  [row], [draft], Path(temporary))[0]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
