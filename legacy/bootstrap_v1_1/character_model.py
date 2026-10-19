import torch
from torch import nn

LENGTH = 128


class CharacterModel(nn.Module):
    """Compact contextual classifier trained from scratch; no transformer/runtime generation loop."""
    def __init__(self, vocabulary_size, width=64):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=0)
        layers = []
        for dilation in (1, 1, 2, 2, 4):
            layers += [nn.Conv2d(width, width, (1, 5), padding=(0, 2 * dilation), dilation=(1, dilation)), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.intent = nn.Linear(width, 3)
        self.slots = nn.Conv2d(width, 9, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids.long()).transpose(1, 2).unsqueeze(2)
        hidden = self.encoder(hidden)
        mask = attention_mask.to(hidden.dtype).unsqueeze(1).unsqueeze(2)
        pooled = (hidden * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
        intent = self.intent(pooled)
        slots = self.slots(hidden).squeeze(2).transpose(1, 2)
        return intent, slots
