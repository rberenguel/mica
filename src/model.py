import torch
import torch.nn as nn
from torch.nn import functional as F
from config import MicaConfig


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, block_size):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        t = torch.arange(block_size)
        freqs = torch.outer(t, inv_freq)
        # (block_size, head_size) — duplicate so rotate_half lines up
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer('cos_cache', cos)
        self.register_buffer('sin_cache', sin)

    def forward(self, q, k, start_pos: int = 0):
        # q, k: (B, nh, T, hs)
        T = q.shape[2]
        cos = self.cos_cache[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


class MultiResolutionAttention(nn.Module):
    # 192 dims split into 3 blocks of 64:
    #   large: 1 head x 64 dims  (semantic)
    #   med:   2 heads x 32 dims (context)
    #   small: 8 heads x 8 dims  (syntax)
    def __init__(self, config):
        super().__init__()
        assert config.n_embd == 192

        self.W_q_large = nn.Linear(config.n_embd, 64, bias=config.bias)
        self.W_k_large = nn.Linear(config.n_embd, 64, bias=config.bias)
        self.W_v_large = nn.Linear(config.n_embd, 64, bias=config.bias)

        self.W_q_med = nn.Linear(config.n_embd, 64, bias=config.bias)
        self.W_k_med = nn.Linear(config.n_embd, 64, bias=config.bias)
        self.W_v_med = nn.Linear(config.n_embd, 64, bias=config.bias)

        self.W_q_small = nn.Linear(config.n_embd, 64, bias=config.bias)
        self.W_k_small = nn.Linear(config.n_embd, 64, bias=config.bias)
        self.W_v_small = nn.Linear(config.n_embd, 64, bias=config.bias)

        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.rope_large = RotaryEmbedding(64, config.block_size)
        self.rope_med   = RotaryEmbedding(32, config.block_size)
        self.rope_small = RotaryEmbedding(8,  config.block_size)

        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, start_pos: int = 0):
        B, T, _ = x.size()
        dp = self.dropout if self.training else 0.0

        q_l = self.W_q_large(x).view(B, T, 1, 64).transpose(1, 2)
        k_l = self.W_k_large(x).view(B, T, 1, 64).transpose(1, 2)
        v_l = self.W_v_large(x).view(B, T, 1, 64).transpose(1, 2)
        q_l, k_l = self.rope_large(q_l, k_l, start_pos)
        out_l = F.scaled_dot_product_attention(q_l, k_l, v_l, dropout_p=dp, is_causal=True)
        out_l = out_l.transpose(1, 2).contiguous().view(B, T, 64)

        q_m = self.W_q_med(x).view(B, T, 2, 32).transpose(1, 2)
        k_m = self.W_k_med(x).view(B, T, 2, 32).transpose(1, 2)
        v_m = self.W_v_med(x).view(B, T, 2, 32).transpose(1, 2)
        q_m, k_m = self.rope_med(q_m, k_m, start_pos)
        out_m = F.scaled_dot_product_attention(q_m, k_m, v_m, dropout_p=dp, is_causal=True)
        out_m = out_m.transpose(1, 2).contiguous().view(B, T, 64)

        q_s = self.W_q_small(x).view(B, T, 8, 8).transpose(1, 2)
        k_s = self.W_k_small(x).view(B, T, 8, 8).transpose(1, 2)
        v_s = self.W_v_small(x).view(B, T, 8, 8).transpose(1, 2)
        q_s, k_s = self.rope_small(q_s, k_s, start_pos)
        out_s = F.scaled_dot_product_attention(q_s, k_s, v_s, dropout_p=dp, is_causal=True)
        out_s = out_s.transpose(1, 2).contiguous().view(B, T, 64)

        return self.resid_dropout(self.W_o(torch.cat([out_l, out_m, out_s], dim=-1)))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.ffn_ratio * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.ffn_ratio * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiResolutionAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, start_pos: int = 0):
        x = x + self.attn(self.ln_1(x), start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class MicaTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos: int = 0):
        b, t = idx.size()

        x = self.transformer.drop(self.transformer.wte(idx))

        for block in self.transformer.h:
            x = block(x, start_pos)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
