"""Tiny BERT + autoregressive copy decoder. No span-model prediction is consumed."""
import math
import re

import torch
from torch import nn
from transformers import AutoModel

from .span_model import LENGTH, encode_rows

CONTROL = ('BOS', 'EOS', 'CONFIRM', 'TIME', 'TITLE', 'LOCATION', 'NOTE')
IDS = {name: LENGTH + i for i, name in enumerate(CONTROL)}
ORDER = ['TIME', 'TITLE', 'LOCATION', 'NOTE']
HEADERS = dict(TIME='时间', TITLE='事项', LOCATION='地点', NOTE='备注')
MAX_OUTPUT = 112


class Normalizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_config(config, attn_implementation='eager')
        h = config.hidden_size
        self.control = nn.Embedding(len(CONTROL), h)
        self.positions = nn.Embedding(MAX_OUTPUT, h)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=h, nhead=2, dim_feedforward=h * 4, dropout=.1, batch_first=True), num_layers=2)
        self.query = nn.Linear(h, h)
        self.keys = nn.Linear(h, h)
        self.special = nn.Linear(h, len(CONTROL))

    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state

    def decode(self, memory, attention_mask, copy_valid, actions):
        copied = memory.gather(1, actions.clamp(0, LENGTH - 1).unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
        control = self.control((actions - LENGTH).clamp(0, len(CONTROL) - 1))
        embeddings = torch.where((actions < LENGTH).unsqueeze(-1), copied, control)
        embeddings = embeddings + self.positions(torch.arange(actions.shape[1], device=actions.device))[None]
        causal = torch.ones((actions.shape[1], actions.shape[1]), dtype=torch.bool, device=actions.device).triu(1)
        hidden = self.decoder(embeddings, memory, tgt_mask=causal,
                              memory_key_padding_mask=attention_mask == 0)
        copies = torch.matmul(self.query(hidden), self.keys(memory).transpose(-1, -2)) / math.sqrt(memory.shape[-1])
        copies = copies.masked_fill(~copy_valid[:, None], -1e4)
        special = self.special(hidden)
        special[:, :, CONTROL.index('BOS')] = -1e4
        return torch.cat([copies, special], -1)

    def forward(self, input_ids, attention_mask, copy_valid, actions):
        return self.decode(self.encode(input_ids, attention_mask), attention_mask, copy_valid, actions)


def targets(rows, offsets):
    sequences = []
    for i, row in enumerate(rows):
        sequence = []
        if row['expected']['decision'] == 'defer':
            sequence = [IDS['CONFIRM']]
        else:
            for role in ORDER:
                spans = [s for s in row['spans'] if s['label'] == role]
                if not spans:
                    continue
                sequence.append(IDS[role])
                for span in spans:
                    pieces = [j for j, (a, b) in enumerate(offsets[i].tolist())
                              if b > a and a >= span['start'] and b <= span['end']]
                    if not pieces or offsets[i, pieces[0], 0] != span['start'] or offsets[i, pieces[-1], 1] != span['end']:
                        raise ValueError('Copy target crosses tokenizer boundary')
                    sequence.extend(pieces)
        sequence.append(IDS['EOS'])
        if len(sequence) >= MAX_OUTPUT:
            raise ValueError('Normalization target too long')
        sequences.append(sequence)
    length = max(map(len, sequences))
    labels = torch.full((len(rows), length), -100)
    actions = torch.full_like(labels, IDS['EOS'])
    for i, sequence in enumerate(sequences):
        labels[i, :len(sequence)] = torch.tensor(sequence)
        actions[i, :len(sequence)] = torch.tensor([IDS['BOS']] + sequence[:-1])
    return actions, labels


def grammar(actions, copy_valid):
    """Only fixed headers plus source copies; source text is never freely generated."""
    mask = torch.zeros(LENGTH + len(CONTROL), dtype=torch.bool)
    if not actions:
        mask[[IDS['TIME'], IDS['TITLE'], IDS['CONFIRM']]] = True
        return mask
    if actions[0] == IDS['CONFIRM']:
        mask[IDS['EOS']] = True
        return mask
    controls = [v for v in actions if v >= LENGTH]
    role = CONTROL[controls[-1] - LENGTH]
    last_control = max(i for i, v in enumerate(actions) if v >= LENGTH)
    field = actions[last_control + 1:]
    mask[:LENGTH] = copy_valid
    used = [v for v in actions if v < LENGTH]
    if used:
        mask[used] = False
    if field:
        # Reorder fields, while each copied field keeps its source order.
        mask[:field[-1] + 1] = False
        if role != 'TIME':
            mask[:LENGTH] = False
            if field[-1] + 1 < LENGTH and copy_valid[field[-1] + 1] and field[-1] + 1 not in used:
                mask[field[-1] + 1] = True
        if role == 'TIME':
            mask[IDS['TITLE']] = True
        elif role == 'TITLE':
            mask[[IDS['LOCATION'], IDS['NOTE'], IDS['EOS']]] = True
        elif role == 'LOCATION':
            mask[[IDS['NOTE'], IDS['EOS']]] = True
        else:
            mask[IDS['EOS']] = True
    return mask


def render(text, actions, offsets):
    if actions == [IDS['CONFIRM'], IDS['EOS']]:
        return '需要确认'
    if not actions or actions[-1] != IDS['EOS']:
        return None
    fields, role = {}, None
    for action in actions[:-1]:
        if action >= LENGTH:
            name = CONTROL[action - LENGTH]
            if name not in ORDER or name in fields:
                return None
            role = name
            fields[role] = []
        elif role:
            fields[role].append(action)
        else:
            return None
    if not fields.get('TITLE'):
        return None
    values = {}
    for role, indexes in fields.items():
        if not indexes or indexes != sorted(set(indexes)):
            return None
        chunks, first, last = [], indexes[0], indexes[0]
        for index in indexes[1:]:
            if index == last + 1:
                last = index
            else:
                chunks.append(text[offsets[first][0]:offsets[last][1]])
                first = last = index
        chunks.append(text[offsets[first][0]:offsets[last][1]])
        if len(chunks) > (2 if role == 'TIME' else 1):
            return None
        values[role] = re.sub(r'\s+', ' ', ' '.join(chunks)).strip()
    return '；'.join(HEADERS[role] + '：' + values[role] for role in ORDER if role in values)


def canonical(row):
    """Gold normalization target for training/evaluation only."""
    if row['expected']['decision'] == 'defer':
        return '需要确认'
    values = {role: ' '.join(row['text'][s['start']:s['end']] for s in row['spans'] if s['label'] == role)
              for role in ORDER}
    return '；'.join(HEADERS[role] + '：' + re.sub(r'\s+', ' ', values[role]).strip() for role in ORDER if values[role])


def predict(model, tokenizer, rows, batch_size=32):
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded, offsets, _, _ = encode_rows(tokenizer, batch, labels=False)
            copy_valid = offsets[:, :, 1] > offsets[:, :, 0]
            memory = model.encode(**encoded)
            actions = torch.full((len(batch), 1), IDS['BOS'], dtype=torch.long)
            sequences, scores, ended = [[] for _ in batch], [[] for _ in batch], [False] * len(batch)
            for step in range(MAX_OUTPUT - 1):
                logits = model.decode(memory, encoded['attention_mask'], copy_valid, actions)[:, -1]
                masks = torch.stack([grammar(seq, copy_valid[i]) if not ended[i] else torch.arange(logits.shape[-1]) == IDS['EOS']
                                     for i, seq in enumerate(sequences)])
                logits = logits.masked_fill(~masks, -1e4)
                probability = logits.softmax(-1)
                chosen = probability.argmax(-1)
                for i, value in enumerate(chosen.tolist()):
                    if not ended[i]:
                        sequences[i].append(value)
                        scores[i].append(float(probability[i, value]))
                        ended[i] = value == IDS['EOS']
                actions = torch.cat([actions, chosen[:, None]], 1)
                if all(ended):
                    break
            for i, row in enumerate(batch):
                results.append(dict(normalized=render(row['text'], sequences[i], offsets[i].tolist()),
                                    minimumProbability=min(scores[i]) if scores[i] else 0.,
                                    meanLogProbability=sum(math.log(max(v, 1e-9)) for v in scores[i]) / max(1, len(scores[i])),
                                    actions=sequences[i], rawText=row['text']))
    return results
