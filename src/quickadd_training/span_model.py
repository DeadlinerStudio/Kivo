"""One whole-sentence tiny BERT encoder and joint token-span role scores."""
import math

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .span_data import ROLES

LENGTH = 96


class SpanModel(nn.Module):
    def __init__(self, config, inner_size=32):
        super().__init__()
        self.encoder = AutoModel.from_config(config, attn_implementation='eager')
        self.inner_size = inner_size
        self.query = nn.Linear(config.hidden_size, len(ROLES) * inner_size)
        self.key = nn.Linear(config.hidden_size, len(ROLES) * inner_size)
        self.start_bias = nn.Linear(config.hidden_size, len(ROLES))
        self.end_bias = nn.Linear(config.hidden_size, len(ROLES))
        self.request = nn.Linear(config.hidden_size, 2)  # supported single request / confirm

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        batch, length, _ = hidden.shape
        q = self.query(hidden).view(batch, length, len(ROLES), self.inner_size).permute(0, 2, 1, 3)
        k = self.key(hidden).view(batch, length, len(ROLES), self.inner_size).permute(0, 2, 1, 3)
        spans = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.inner_size)
        spans = spans + self.start_bias(hidden).permute(0, 2, 1).unsqueeze(-1)
        spans = spans + self.end_bias(hidden).permute(0, 2, 1).unsqueeze(-2)
        return self.request(hidden[:, 0]), spans


def initialize(teacher_dir):
    """Reuse v3 encoder weights exactly; fresh heads, no old intent/slot heads."""
    import json
    from pathlib import Path
    path = Path(teacher_dir)
    report = json.loads((path / 'training-report.json').read_text())
    config = AutoConfig.from_pretrained(report['pretrainedModel'], revision=report['revision'],
                                        local_files_only=True, trust_remote_code=False)
    model = SpanModel(config)
    weights = torch.load(path / 'teacher.pt', map_location='cpu', weights_only=True)
    encoder = {k.removeprefix('encoder.'): v for k, v in weights.items() if k.startswith('encoder.')}
    model.encoder.load_state_dict(encoder, strict=True)
    return model, report


def encode_rows(tokenizer, rows, labels=True):
    encoded = tokenizer([r['text'] for r in rows], padding='max_length', max_length=LENGTH,
                        truncation=False, return_offsets_mapping=True, return_tensors='pt')
    offsets = encoded.pop('offset_mapping')
    if encoded['input_ids'].shape[1] != LENGTH:
        raise ValueError('WordPiece context exceeded; training never truncates')
    encoded.pop('token_type_ids', None)
    targets = torch.zeros((len(rows), len(ROLES), LENGTH, LENGTH)) if labels else None
    valid_token = offsets[:, :, 1] > offsets[:, :, 0]
    valid = valid_token[:, :, None] & valid_token[:, None, :]
    valid &= torch.ones((LENGTH, LENGTH), dtype=torch.bool).triu()[None]
    # All bounded token pairs are candidates; no dictionary of title/location names.
    valid &= (torch.arange(LENGTH)[None, :] - torch.arange(LENGTH)[:, None] < 64)[None]
    if labels:
        for i, row in enumerate(rows):
            starts = {a: j for j, (a, b) in enumerate(offsets[i].tolist()) if b > a}
            ends = {b: j for j, (a, b) in enumerate(offsets[i].tolist()) if b > a}
            for span in row['spans']:
                if span['start'] not in starts or span['end'] not in ends:
                    raise ValueError(f"Unrepresentable gold token boundary: {row['id']} {span}")
                a, b = starts[span['start']], ends[span['end']]
                if not valid[i, a, b]:
                    raise ValueError('Gold exceeds candidate span bounds')
                targets[i, ROLES.index(span['label']), a, b] = 1.
    return encoded, offsets, valid, targets


def span_loss(logits, target, valid):
    """Balance all negative candidates against positives per role, including empty roles."""
    mask = valid[:, None].expand_as(target)
    negative = logits.masked_fill(~(mask & (target == 0)), -1e4).flatten(2)
    positive = (-logits).masked_fill(~(mask & (target == 1)), -1e4).flatten(2)
    zeros = torch.zeros((*negative.shape[:2], 1), device=logits.device)
    return (torch.logsumexp(torch.cat([negative, zeros], -1), -1)
            + torch.logsumexp(torch.cat([positive, zeros], -1), -1)).mean()


def predict(model, tokenizer, rows, batch_size=32):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded, offsets, valid, _ = encode_rows(tokenizer, batch, labels=False)
            request, spans = model(**encoded)
            predictions.extend(candidates_from_logits(request, spans, offsets, valid))
    return predictions


def candidates_from_logits(request, spans, offsets, valid):
    """Identical candidate extraction for PyTorch training checks and ONNX execution."""
    spans = spans.masked_fill(~valid[:, None], -1e4)
    predictions = []
    for i in range(request.shape[0]):
        candidates = []
        for role in range(len(ROLES)):
            scores, indexes = spans[i, role].flatten().topk(12)
            for score, index in zip(scores.tolist(), indexes.tolist()):
                if score < -2:
                    continue
                a, b = divmod(index, LENGTH)
                candidates.append(dict(role=ROLES[role], start=offsets[i, a, 0].item(),
                                       end=offsets[i, b, 1].item(), score=score))
        predictions.append(dict(requestProbabilities=request[i].softmax(-1).tolist(), candidates=candidates))
    return predictions


def predict_onnx(runtime, tokenizer, rows):
    import numpy as np
    predictions = []
    for row in rows:
        encoded, offsets, valid, _ = encode_rows(tokenizer, [row], labels=False)
        outputs = runtime.run(None, {k: v.numpy().astype(np.int32) for k, v in encoded.items()})
        predictions.extend(candidates_from_logits(*(torch.from_numpy(v) for v in outputs), offsets, valid))
    return predictions
