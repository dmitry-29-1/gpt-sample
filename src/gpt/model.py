import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, forward_expansion: int):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            # nn.RELU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, attn_mask):
        attention_output, _ = self.attention(value, value, value, attn_mask=attn_mask)
        norm1_output = self.dropout(self.norm1(attention_output + value))
        feed_forward_output = self.feed_forward(norm1_output)
        norm2_output = self.dropout(self.norm2(feed_forward_output + norm1_output))
        return norm2_output


class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            sequence_len: int,
            num_layers: int,
            embed_dim: int,
            num_heads: int,
            forward_expansion: int,
            dropout: float,
            pad_idx: int,
    ):
        super(GPT, self).__init__()

        # params
        self.sequence_len = sequence_len
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        self.num_heads = num_heads

        # structure
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(sequence_len, embed_dim)

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

        self.fully_connected_output = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_attn_mask(self, idx):
        N, seq_len = idx.size()
        mask = torch.triu(torch.ones((N, seq_len, seq_len)), diagonal=1).bool()
        mask = mask.masked_fill(idx[:, :, None] == self.pad_idx, value=True)
        mask = mask.repeat(self.num_heads, 1, 1)
        return mask.to(idx.device)

    def forward(self, x):
        N, seq_length = x.size()
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        word_embedding_output = self.word_embedding(x)
        position_embedding_output = self.position_embedding(positions)
        out = self.dropout(word_embedding_output + position_embedding_output)

        attn_mask = self.make_attn_mask(x)

        for layer in self.layers:
            out = layer(out, attn_mask=attn_mask)

        out = self.fully_connected_output(out)
        return out
