import torch
import torch.nn as nn

class GPT1(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embedding_dim, num_heads, num_layers, hidden_dim):
        super(GPT1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                activation="gelu",
            ) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)
