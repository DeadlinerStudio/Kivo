"""Frozen-encoder proposal ranking and per-predicted-kind correctness heads."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import FIELDS, digest
from .joint_decode import candidates, quality_features
from .joint_model import predict
from .joint_native import attach


def exact(row, value):
    return row['expected']['decision'] == 'accept' and all(
        value['proposal'].get(k) == row['expected']['proposal'][k] for k in FIELDS)


class Head(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('scale', torch.ones(size))
        self.network = nn.Sequential(nn.Linear(size, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))

    def standardize(self, values):
        self.mean.copy_(values.mean(0))
        self.scale.copy_(values.std(0, correction=0).clamp_min(.1))

    def forward(self, x):
        return self.network((x - self.mean) / self.scale).squeeze(-1)


def prepare(rows, raw, native):
    groups = [candidates(r, p) for r, p in zip(rows, raw)]
    flat_rows = [r for r, values in zip(rows, groups) for _ in values]
    flat = [v for values in groups for v in values]
    normalized = native.normalize(flat_rows, flat)
    offset = 0
    for i, values in enumerate(groups):
        groups[i] = [attach(v, n) for v, n in zip(values, normalized[offset:offset + len(values)])]
        offset += len(values)
    return groups


def scores(heads, values):
    kind = values[0]['proposal']['kind']
    if not heads:
        return np.asarray([v['candidateScore'] for v in values], dtype=np.float32)
    head = heads[kind]
    head.eval()
    with torch.inference_mode():
        return head(torch.from_numpy(np.stack([v['feature'] for v in values]))).numpy()


def choose(groups, heads):
    selected = []
    for values in groups:
        logits = scores(heads, values)
        order = np.argsort(-logits, kind='stable')
        result = dict(values[int(order[0])])
        result['rankingMargin'] = float(logits[order[0]] - logits[order[1]]) if len(order) > 1 else 0.
        result['qualityFeature'] = quality_features(result, result['rankingMargin'])
        selected.append(result)
    return selected


def train_rank(rows, groups, dev_rows, dev_groups, epochs=25):
    heads, history = {}, {}
    for kind in ['TASK', 'HABIT']:
        train = [(values, np.asarray([v['draftValid'] and exact(r, v) for v in values]))
                 for r, values in zip(rows, groups) if values[0]['proposal']['kind'] == kind]
        train = [(values, y) for values, y in train if y.any() and len(values) > 1]
        head = Head(len(groups[0][0]['feature']))
        head.standardize(torch.from_numpy(np.stack([v['feature'] for values, _ in train for v in values])))
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01)
        best, records = -1, []
        for epoch in range(epochs):
            head.train()
            losses = []
            for batch in torch.randperm(len(train)).split(32):
                examples = [train[i] for i in batch.tolist()]
                x = torch.zeros((len(examples), 32, head.mean.numel()))
                valid, good = torch.zeros((len(examples), 32), dtype=torch.bool), torch.zeros((len(examples), 32), dtype=torch.bool)
                for i, (values, y) in enumerate(examples):
                    n = len(values)
                    x[i, :n] = torch.from_numpy(np.stack([v['feature'] for v in values]))
                    valid[i, :n], good[i, :n] = True, torch.from_numpy(y)
                logits = head(x)
                loss = (torch.logsumexp(logits.masked_fill(~valid, -1e4), -1)
                        - torch.logsumexp(logits.masked_fill(~good, -1e4), -1)).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
            head.eval()
            correct = 0
            for r, values in zip(dev_rows, dev_groups):
                if values[0]['proposal']['kind'] != kind:
                    continue
                index = int(np.argmax(scores({kind: head}, values)))
                correct += values[index]['draftValid'] and exact(r, values[index])
            records.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), devCorrect=int(correct)))
            if correct > best:
                best, state = correct, copy.deepcopy(head.state_dict())
        head.load_state_dict(state)
        heads[kind], history[kind] = head, dict(eligibleTrainingGroups=len(train), epochs=records, devBestCorrect=int(best))
    return heads, history


def train_quality(rows, values, dev_rows, dev_values, epochs=40):
    heads, history = {}, {}
    for kind in ['TASK', 'HABIT']:
        examples = [(v['qualityFeature'], float(exact(r, v))) for r, v in zip(rows, values) if v['proposal']['kind'] == kind]
        validation = [(v['qualityFeature'], float(exact(r, v))) for r, v in zip(dev_rows, dev_values) if v['proposal']['kind'] == kind]
        x = torch.from_numpy(np.stack([v for v, _ in examples]))
        y = torch.tensor([label for _, label in examples])
        dx = torch.from_numpy(np.stack([v for v, _ in validation]))
        dy = torch.tensor([label for _, label in validation])
        head = Head(x.shape[-1]); head.standardize(x)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01)
        pos_weight = (len(y) - y.sum()) / y.sum().clamp_min(1)
        records, best = [], float('inf')
        for epoch in range(epochs):
            head.train()
            for batch in torch.randperm(len(y)).split(64):
                loss = nn.functional.binary_cross_entropy_with_logits(head(x[batch]), y[batch], pos_weight=pos_weight)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            head.eval()
            with torch.inference_mode():
                loss = float(nn.functional.binary_cross_entropy_with_logits(head(dx), dy))
            records.append(dict(epoch=epoch + 1, devNLL=loss))
            if loss < best:
                best, state = loss, copy.deepcopy(head.state_dict())
        head.load_state_dict(state)
        heads[kind], history[kind] = head, dict(trainingRows=len(y), positives=int(y.sum()), realPredictionsOnly=True,
                                              history=records, bestDevNLL=best)
    return heads, history


def quality_logits(heads, selected):
    values = np.zeros(len(selected), dtype=np.float64)
    for kind in ['TASK', 'HABIT']:
        ids = [i for i, v in enumerate(selected) if v['proposal']['kind'] == kind]
        heads[kind].eval()
        if ids:
            with torch.inference_mode():
                output = heads[kind](torch.from_numpy(np.stack([selected[i]['qualityFeature'] for i in ids]))).numpy()
            values[ids] = output
    return values


def save_heads(path, heads):
    torch.save({k: dict(size=h.mean.numel(), state=h.state_dict()) for k, h in heads.items()}, path)


def load_heads(path):
    result = {}
    for kind, value in torch.load(path, weights_only=True).items():
        head = Head(value['size']); head.load_state_dict(value['state']); head.eval(); result[kind] = head
    return result
