"""Dev-triggered small type-specific residuals; still exactly one BERT encoder."""
import math
import torch
from torch import nn

from .joint_model import JointModel
from .joint_data import ROLES


class AdapterJointModel(JointModel):
    def __init__(self, config):
        super().__init__(config)
        h = config.hidden_size
        self.adapters = nn.ModuleList([nn.Sequential(nn.Linear(h, 32), nn.GELU(), nn.Linear(32, h)) for _ in range(2)])
        for layer in self.adapters:
            nn.init.zeros_(layer[-1].weight)
            nn.init.zeros_(layer[-1].bias)

    def forward(self, input_ids, attention_mask, preferred_kind):
        hidden = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        hint = nn.functional.one_hot(preferred_kind.long(), 2).to(hidden.dtype)
        intent = self.intent(torch.cat([hidden[:, 0], hint], -1))
        # The same predicted route is used during training and inference; never a gold kind input.
        route = intent.argmax(-1)
        hidden = hidden + sum(layer(hidden) * (route == kind).to(hidden.dtype)[:, None, None]
                              for kind, layer in enumerate(self.adapters))
        b, length, _ = hidden.shape
        q = self.query(hidden).view(b, length, len(ROLES), 32).permute(0, 2, 1, 3)
        k = self.key(hidden).view(b, length, len(ROLES), 32).permute(0, 2, 1, 3)
        spans = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(32)
        spans = spans + self.start_bias(hidden).permute(0, 2, 1).unsqueeze(-1)
        spans = spans + self.end_bias(hidden).permute(0, 2, 1).unsqueeze(-2)
        return intent, spans, hidden


def network(config):
    if getattr(config, 'quickadd_type_heads', False):
        return TypeHeadJointModel(config)
    return AdapterJointModel(config) if getattr(config, 'quickadd_adapter', False) else JointModel(config)


class TypeHeadJointModel(JointModel):
    """One encoder, separate small boundary heads; supervised domain only during training."""
    def __init__(self, config):
        super().__init__(config)
        h = config.hidden_size
        self.habit_query, self.habit_key = nn.Linear(h, len(ROLES) * 32), nn.Linear(h, len(ROLES) * 32)
        self.habit_start, self.habit_end = nn.Linear(h, len(ROLES)), nn.Linear(h, len(ROLES))

    def span_scores(self, hidden, query, key, start, end):
        b, length, _ = hidden.shape
        q = query(hidden).view(b, length, len(ROLES), 32).permute(0, 2, 1, 3)
        k = key(hidden).view(b, length, len(ROLES), 32).permute(0, 2, 1, 3)
        return (torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(32)
                + start(hidden).permute(0, 2, 1).unsqueeze(-1)
                + end(hidden).permute(0, 2, 1).unsqueeze(-2))

    def forward(self, input_ids, attention_mask, preferred_kind, training_kind=None):
        hidden = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        hint = nn.functional.one_hot(preferred_kind.long(), 2).to(hidden.dtype)
        intent = self.intent(torch.cat([hidden[:, 0], hint], -1))
        route = intent.argmax(-1) if training_kind is None else training_kind
        task = self.span_scores(hidden, self.query, self.key, self.start_bias, self.end_bias)
        habit = self.span_scores(hidden, self.habit_query, self.habit_key, self.habit_start, self.habit_end)
        return intent, torch.where((route == 1)[:, None, None, None], habit, task), hidden
