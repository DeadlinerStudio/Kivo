import torch
from torch import nn

LENGTH = 128


class CharacterModel(nn.Module):
    """Character CNN with 125-character receptive field and global slot context; trained from scratch."""
    def __init__(self, vocabulary_size, width=64):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=0)
        layers = []
        for dilation in (1, 2, 4, 8, 16):
            layers += [nn.Conv2d(width, width, (1, 5), padding=(0, 2 * dilation), dilation=(1, dilation)), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.intent = nn.Linear(width, 3)
        self.slots = nn.Conv2d(width * 2, 9, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids.long()).transpose(1, 2).unsqueeze(2)
        hidden = self.encoder(hidden)
        mask = attention_mask.to(hidden.dtype).unsqueeze(1).unsqueeze(2)
        pooled = (hidden * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
        intent = self.intent(pooled)
        global_context = pooled.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, LENGTH)
        slots = self.slots(torch.cat([hidden, global_context], dim=1)).squeeze(2).transpose(1, 2)
        return intent, slots
