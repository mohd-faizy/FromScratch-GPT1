import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.drop = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_head,
                dim_feedforward=config.d_ff,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, x, attention_mask=None):
        B, T = x.size()
        device = x.device
        
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x, src_mask=mask, src_key_padding_mask=attention_mask)
            
        x = self.ln_f(x)
        return self.head(x)