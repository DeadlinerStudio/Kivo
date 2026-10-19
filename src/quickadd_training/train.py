"""Supervised training: reads train/dev only; no acceptance/test-derived vocabulary."""
import argparse
import json
import random
from pathlib import Path
from .data import load, validate, supervised, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--coreml', action='store_true')
    parser.add_argument('--public-data', type=Path)
    parser.add_argument('--public-epochs', type=int, default=3)
    parser.add_argument('--locale', choices=['zh', 'en'], default='zh')
    parser.add_argument('--stem', help='Export filename stem, default quick-add-<locale>-v2')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--allow-synthetic', action='store_true', help='Explicit experimental training; never makes a model release eligible')
    parser.add_argument('--slot-loss-weight', type=float, default=2.0)
    parser.add_argument('--repeat-loss-weight', type=float, default=3.0)
    args = parser.parse_args()
    # Deliberately do not read test. Validate all splits separately before any run.
    hashes = {s:digest(args.data_dir/(s+'.jsonl')) for s in ('train','dev')}
    splits = {s: load(args.data_dir / (s + '.jsonl')) for s in ('train', 'dev')}
    validate(splits)
    if not splits['train'] or not splits['dev']:
        raise SystemExit('Reviewed train/dev data required; bootstrap challenge examples are not training labels')
    if any(r.get('locale', args.locale) != args.locale for rows in splits.values() for r in rows):
        raise SystemExit('Corpus/model locale mismatch')
    if any(r.get('annotationStatus') != 'human-reviewed' for r in splits['train']) and not args.allow_synthetic:
        raise SystemExit('Human-reviewed training required, or explicitly opt into --allow-synthetic experiments')
    stem = args.stem or f'quick-add-{args.locale}-v2'
    import re
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', stem) or args.epochs < 1 or args.threads < 1:
        raise SystemExit('Invalid export stem, epochs or thread count')
    train = [supervised(r) for r in splits['train']]
    dev = [supervised(r) for r in splits['dev'] if 'spans' in r]
    if not dev:
        raise SystemExit('At least one reviewed development span annotation is required')
    import numpy as np
    import torch
    from .character_model import CharacterModel, LENGTH
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    characters={c for r in train for c in r['text']}
    if args.public_data:
        from .public_pretrain import public_rows
        characters.update(c for r in public_rows(args.public_data) for c in r['text'])
    vocabulary = ['[PAD]', '[UNK]'] + sorted(characters)
    lookup = {c:i for i,c in enumerate(vocabulary)}
    def tensorize(rows):
        ids = np.zeros((len(rows), LENGTH), dtype=np.int32)
        mask = np.zeros_like(ids); slots = np.full_like(ids, -100)
        for i,row in enumerate(rows):
            for j,c in enumerate(row['text']):
                ids[i,j] = lookup.get(c,1); mask[i,j] = 1; slots[i,j] = row['slots'][j]
        return tuple(torch.from_numpy(a) for a in (ids,mask,slots)), torch.tensor([r['intent'] for r in rows])
    (ids, mask, slots), intents = tensorize(train)
    import shutil
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot=args.output/'training-source';snapshot.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):shutil.copy(source,snapshot/source.name)
    model = CharacterModel(len(vocabulary))
    public_report=None
    if args.public_data:
        from .public_pretrain import warmstart
        public_report=warmstart(model,lookup,args.public_data,args.public_epochs,args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.01)
    slot_weights = torch.ones(9); slot_weights[5:7] = args.repeat_loss_weight
    history = []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for batch in torch.randperm(len(train)).split(128):
            x = ids[batch].clone()
            x[(torch.rand(x.shape) < .025) & (mask[batch] == 1)] = 1
            intent_logits, slot_logits = model(x, mask[batch])
            loss = torch.nn.functional.cross_entropy(intent_logits, intents[batch]) + args.slot_loss_weight * torch.nn.functional.cross_entropy(slot_logits.reshape(-1,9),slots[batch].reshape(-1).long(),ignore_index=-100,weight=slot_weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        history.append(sum(losses)/len(losses))
        print(f'Epoch {epoch+1}: loss={history[-1]:.4f}', flush=True)
    model.eval(); args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / 'weights.pt')
    (dx,dm,ds),di = tensorize(dev)
    with torch.no_grad():
        il,sl = model(dx,dm)
        metrics=dict(intentAccuracy=float((il.argmax(-1)==di).float().mean()),
                     exactSlots=float(((sl.argmax(-1)==ds)|(ds==-100)).all(-1).float().mean()))
    example=(ids[:1],mask[:1])
    torch.onnx.export(model,example,str(args.output/(stem+'.onnx')),input_names=['input_ids','attention_mask'],output_names=['intent_logits','slot_logits'],opset_version=13,dynamo=False)
    if args.coreml:
        import coremltools as ct
        ml=ct.convert(torch.jit.trace(model,example),inputs=[ct.TensorType(name='input_ids',shape=(1,LENGTH),dtype=np.int32),ct.TensorType(name='attention_mask',shape=(1,LENGTH),dtype=np.int32)],outputs=[ct.TensorType(name='intent_logits'),ct.TensorType(name='slot_logits')],convert_to='neuralnetwork',minimum_deployment_target=ct.target.iOS14)
        ml.save(str(args.output/(stem+'.mlmodel')))
    (args.output/'vocabulary.json').write_text(json.dumps(vocabulary,ensure_ascii=False))
    report=dict(publicAuxiliary=public_report,architecture='character-cnn-receptive125-global-slots',trainExamples=len(train),devExamples=len(dev),parameters=sum(p.numel() for p in model.parameters()),locale=args.locale,modelStem=stem,allowSynthetic=args.allow_synthetic,slotLossWeight=args.slot_loss_weight,repeatLossWeight=args.repeat_loss_weight,seed=args.seed,epochs=args.epochs,loss=history,trainSha256=hashes['train'],devSha256=hashes['dev'],vocabularySource='business train + public official train only' if args.public_data else 'train only',devMetrics=metrics,releaseEligible=False,note='Intent/slot scores are diagnostics. Full production proposal benchmark and native checks required.')
    if any(digest(args.data_dir/(s+'.jsonl')) != hashes[s] for s in hashes):raise RuntimeError('Corpus changed during training')
    (args.output/'training-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
