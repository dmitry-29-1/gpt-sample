import torch
from torch import nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, attn_mask):
        attn_output, _ = self.attention(query, key, value, attn_mask=attn_mask)
        attn_output = self.dropout(self.norm1(attn_output + value))
        ff_output = self.feed_forward(attn_output)
        output = self.dropout(self.norm2(ff_output + attn_output))
        return output

class GPT(nn.Module):
    def __init__(
            self,
            vocab_size,
            max_len,
            num_layers,
            embed_dim,
            num_heads,
            forward_expansion,
            dropout,
            pad_idx
    ):
        super(GPT, self).__init__()
        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def make_attn_mask(self, idx):
        seq_len = idx.shape[1]
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        mask = mask.masked_fill(idx[:, :, None] == self.pad_idx, value=True)
        return mask.to(idx.device)

    def forward(self, x):
        N, seq_length = x.size()
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        attn_mask = self.make_attn_mask(x)

        for layer in self.layers:
            out = layer(out, out, out, attn_mask=attn_mask)

        out = self.fc_out(out)

        return out
