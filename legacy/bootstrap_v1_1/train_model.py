"""Reproducible bootstrap training/export. Synthetic accuracy is not real-world accuracy."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from character_model import CharacterModel, LENGTH
from training_data import generate, ACCEPTANCE


def tensorize(rows, vocabulary, dropout=False):
    lookup = {s: i for i, s in enumerate(vocabulary)}
    ids = np.zeros((len(rows), LENGTH), dtype=np.int32)
    mask = np.zeros_like(ids)
    slots = np.full_like(ids, -100)
    for index, row in enumerate(rows):
        for pos, char in enumerate(row['text'][:LENGTH]):
            ids[index, pos] = lookup.get(char, 1)
            mask[index, pos] = 1
            slots[index, pos] = row['slots'][pos]
    return tuple(torch.from_numpy(a) for a in [ids, mask, slots]), torch.tensor([r['intent'] for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--resume', type=Path, help='Warm-start weights from a recorded training stage')
    parser.add_argument('--learning-rate', type=float, default=0.0015)
    parser.add_argument('--slot-loss-weight', type=float, default=1.0)
    parser.add_argument('--repeat-loss-weight', type=float, default=1.0)
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(6)
    rows = generate()
    vocabulary = ['[PAD]', '[UNK]'] + sorted({c for row in rows + ACCEPTANCE for c in row['text']} | set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -_,.:：，。？！一二三四五六七八九十两零'))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'vocabulary.json').write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2))
    (args.output / 'acceptance.json').write_text(json.dumps(ACCEPTANCE, ensure_ascii=False, indent=2))
    # Random split within generated families; this is a bootstrap regression set, not independent evaluation.
    train, validation = rows[:17000], rows[17000:]
    (x, masks, y), intents = tensorize(train, vocabulary)
    model = CharacterModel(len(vocabulary))
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    slot_weights = torch.ones(9)
    slot_weights[5:7] = args.repeat_loss_weight
    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=slot_weights)
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train))
        total = 0
        for batch in order.split(128):
            # Unknown-token training prevents a rare title character from forcing a wrong boundary.
            ids = x[batch].clone()
            ids[(torch.rand(ids.shape) < 0.025) & (masks[batch] == 1)] = 1
            intent_logits, slot_logits = model(ids, masks[batch])
            loss = nn.functional.cross_entropy(intent_logits, intents[batch]) + args.slot_loss_weight * criterion(slot_logits.reshape(-1, 9), y[batch].reshape(-1).long())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print('Epoch', epoch + 1, 'loss', round(total / len(order.split(128)), 4), 'elapsed', round(time.time() - started), flush=True)
    model.eval()
    torch.save(model.state_dict(), args.output / 'weights.pt')
    (vx, vm, vs), vi = tensorize(validation, vocabulary)
    correct_intents = correct_spans = 0
    with torch.no_grad():
        for start in range(0, len(validation), 128):
            il, sl = model(vx[start:start+128], vm[start:start+128])
            correct_intents += (il.argmax(-1) == vi[start:start+128]).sum().item()
            correct_spans += ((sl.argmax(-1) == vs[start:start+128]) | (vs[start:start+128] == -100)).all(-1).sum().item()
    report = dict(training='synthetic bootstrap, random split within generated families; not an independent real-world benchmark',
                  examples=len(train), validationExamples=len(validation), parameters=sum(p.numel() for p in model.parameters()),
                  intentAccuracy=correct_intents/len(validation), exactSlots=correct_spans/len(validation), elapsedSeconds=time.time()-started)
    report['epochs'] = args.epochs
    report['learningRate'] = args.learning_rate
    report['slotLossWeight'] = args.slot_loss_weight
    report['repeatLossWeight'] = args.repeat_loss_weight
    if args.resume:
        import hashlib
        report['warmStartSha256'] = hashlib.sha256(args.resume.read_bytes()).hexdigest()
    (ax, am, _), _ = tensorize(ACCEPTANCE, vocabulary)
    with torch.no_grad():
        il, sl = model(ax, am)
        report['manualExamples'] = [dict(text=row['text'], expectedIntent=row['intent'], predictedIntent=int(il[i].argmax()),
                                        intentConfidence=float(il[i].softmax(-1).max()), exactSlots=bool(torch.equal(sl[i, :len(row['text'])].argmax(-1), torch.tensor(row['slots']))),
                                        predictedSlots=sl[i, :len(row['text'])].argmax(-1).tolist(), minSlotConfidence=float(sl[i, :len(row['text'])].softmax(-1).max(-1).values.min())) for i,row in enumerate(ACCEPTANCE)]
    (args.output / 'training-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    examples = (ax[:1], am[:1])
    torch.onnx.export(model, examples, str(args.output / 'quick-add-zh-v1.onnx'), input_names=['input_ids','attention_mask'],
                      output_names=['intent_logits','slot_logits'], opset_version=13, do_constant_folding=True, dynamo=False)
    import onnxruntime as ort
    runtime = ort.InferenceSession(str(args.output / 'quick-add-zh-v1.onnx'), providers=['CPUExecutionProvider'])
    expected = model(*examples)
    actual = runtime.run(None, dict(input_ids=examples[0].numpy(), attention_mask=examples[1].numpy()))
    report['onnxMaxAbsoluteError'] = max(float(np.max(np.abs(a-b.detach().numpy()))) for a,b in zip(actual,expected))
    import coremltools as ct
    traced = torch.jit.trace(model, examples)
    ml = ct.convert(traced, inputs=[ct.TensorType(name='input_ids', shape=(1,LENGTH),dtype=np.int32),
                                   ct.TensorType(name='attention_mask',shape=(1,LENGTH),dtype=np.int32)],
                    outputs=[ct.TensorType(name='intent_logits'),ct.TensorType(name='slot_logits')],
                    convert_to='neuralnetwork', minimum_deployment_target=ct.target.iOS14)
    ml.save(str(args.output / 'quick-add-zh-v1.mlmodel'))
    prediction = ml.predict(dict(input_ids=examples[0].numpy(),attention_mask=examples[1].numpy()))
    report['coremlShapes'] = {k:list(v.shape) for k,v in prediction.items()}
    report['coremlMaxAbsoluteError'] = max(float(np.max(np.abs(prediction[k]-b.detach().numpy()))) for k,b in zip(['intent_logits','slot_logits'],expected))
    (args.output / 'training-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
