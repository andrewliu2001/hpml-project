import torch
import numpy as np
import torch.nn as nn

class GPTBlock(nn.Module):
    """"  Pre-norm transformer block   """
    def __init__(self, transition_dim, seq_len, embedding_dim, num_heads, attention_dropout, residual_dropout):
        print(f'Num heads: {num_heads}')
        print(f'embedding dim: {embedding_dim}')
        print(f'seq_len: {seq_len}')
        print(f'transition_dim: {transition_dim}')
        super().__init__()
        self.seq_len = seq_len
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, attention_dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer("attn_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool))
        # mask out previous value estimates (as they have information about future)
        self.attn_mask[:, transition_dim - 1::transition_dim] = True

    def forward(self, x, state=None, attn_pad_mask=None):
        # state is a previous input to this layer
        x = self.norm1(x)

        if state is None:
            # if context_len < seq_len
            attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]]
            q, k, v = x, x, x
        else:
            assert x.size(1) == 1, f'when using memory input should be 1-time-step tensor, got {x.size(1)} timesteps.'
            assert state.shape[1] + 1 <= self.seq_len, f"{state.shape[1] + 1}"

            attn_mask = None
            q, k, v = x, torch.cat([state, x], dim=1), torch.cat([state, x], dim=1)

        new_state = k
        x = x + self.drop(self.attention(q, k, v, attn_mask=attn_mask, key_padding_mask=attn_pad_mask, need_weights=False)[0])
        x = x + self.mlp(self.norm2(x))

        return x, new_state

