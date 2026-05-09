import torch
import torch.nn as nn
import torch.nn.functional as F


class XSSClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_filters: int = 64, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        emb = self.embedding(x).permute(0, 2, 1)
        h3 = F.relu(self.conv3(emb)).max(dim=2).values
        h5 = F.relu(self.conv5(emb)).max(dim=2).values
        h = torch.cat([h3, h5], dim=1)
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] trainable parameters: {n:,}")
    return n
