"""One cached tiny BERT encoding, three-way intent and eight-role joint span heads."""
import math
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from .joint_data import ROLES, KINDS
from .span_model import LENGTH, encode_rows as encode_base, span_loss


class JointModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_config(config, attn_implementation='eager')
        h, r, d = config.hidden_size, len(ROLES), 32
        self.query, self.key = nn.Linear(h, r * d), nn.Linear(h, r * d)
        self.start_bias, self.end_bias = nn.Linear(h, r), nn.Linear(h, r)
        self.intent = nn.Linear(h + 2, len(KINDS))

    def forward(self, input_ids, attention_mask, preferred_kind):
        hidden = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        b, length, _ = hidden.shape
        q = self.query(hidden).view(b, length, len(ROLES), 32).permute(0, 2, 1, 3)
        k = self.key(hidden).view(b, length, len(ROLES), 32).permute(0, 2, 1, 3)
        spans = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(32)
        spans = spans + self.start_bias(hidden).permute(0, 2, 1).unsqueeze(-1)
        spans = spans + self.end_bias(hidden).permute(0, 2, 1).unsqueeze(-2)
        hint = torch.nn.functional.one_hot(preferred_kind.long(), 2).to(hidden.dtype)
        return self.intent(torch.cat([hidden[:, 0], hint], -1)), spans, hidden


def encode(tokenizer, rows):
    encoded, offsets, valid, _ = encode_base(tokenizer, rows, labels=False)
    encoded['preferred_kind'] = torch.tensor([int(r['preferredKind'] == 'HABIT') for r in rows])
    return encoded, offsets, valid


def targets(rows, offsets):
    values = torch.zeros((len(rows), len(ROLES), LENGTH, LENGTH))
    for i, row in enumerate(rows):
        starts = {a: j for j, (a, b) in enumerate(offsets[i].tolist()) if b > a}
        ends = {b: j for j, (a, b) in enumerate(offsets[i].tolist()) if b > a}
        for span in row['spans']:
            a, b = starts.get(span['start']), ends.get(span['end'])
            if a is None or b is None or b - a >= 64:
                raise ValueError('Joint label crosses tokenizer candidate boundary: ' + row['id'])
            values[i, ROLES.index(span['label']), a, b] = 1
    intents = torch.tensor([KINDS.index(r['expected']['proposal']['kind']) if r['expected']['decision'] == 'accept'
                            else 2 for r in rows])
    return intents, values


def from_logits(intent, spans, hidden, offsets, valid, with_vectors=True):
    spans = spans.masked_fill(~valid[:, None], -1e4)
    results = []
    for i in range(len(intent)):
        candidates = []
        for role in range(len(ROLES)):
            scores, ids = spans[i, role].flatten().topk(12)
            for score, index in zip(scores.tolist(), ids.tolist()):
                if score < -2:
                    continue
                a, b = divmod(index, LENGTH)
                value = dict(role=ROLES[role], start=int(offsets[i, a, 0]), end=int(offsets[i, b, 1]), score=score)
                if with_vectors:
                    value['vector'] = hidden[i, a:b + 1].mean(0).numpy().copy()
                    value['edge'] = torch.cat([hidden[i, max(0, a - 1), :16], hidden[i, a, :16],
                                               hidden[i, b, :16], hidden[i, min(LENGTH - 1, b + 1), :16]]).numpy().copy()
                candidates.append(value)
        result = dict(intentProbabilities=intent[i].softmax(-1).tolist(), candidates=candidates)
        if with_vectors:
            result['vector'] = hidden[i, 0].numpy().copy()
        results.append(result)
    return results


def predict(model, tokenizer, rows, batch_size=32, with_vectors=True):
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            inputs, offsets, valid = encode(tokenizer, batch)
            results += from_logits(*model(**inputs), offsets, valid, with_vectors)
    return results


def span_quality(rows, raw):
    result = {}
    for role in ROLES:
        if role == 'CONNECTOR':
            continue
        tp = predicted = gold = 0
        for row, prediction in zip(rows, raw):
            expected = {(s['start'], s['end']) for s in row['spans'] if s['label'] == role}
            actual = {(s['start'], s['end']) for s in prediction['candidates'] if s['role'] == role and s['score'] >= 0}
            tp += len(expected & actual)
            predicted += len(actual)
            gold += len(expected)
        result[role] = 2 * tp / (predicted + gold) if predicted + gold else None
    return result
