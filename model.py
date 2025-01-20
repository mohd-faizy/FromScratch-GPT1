import torch
import torch.nn as nn


class GPT1(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embedding_dim, num_heads, num_layers, hidden_dim):
        super(GPT1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.positional_embedding(positions)

        for layer in self.transformer_layers:
            x = layer(x)

        logits = self.fc_out(x)
        return logits
