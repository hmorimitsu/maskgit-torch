# This code is taken from: https://github.com/dome272/MaskGIT-pytorch
#
# Copyright (c) 2022 Dominic Rampas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
import torch.nn as nn


def weights_init(m) -> None:
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)


class Attention(nn.Module):
    """
    Simple Self-Attention algorithm. Potential for optimization using a non-quadratic attention mechanism in complexity.
    -> Linformer, Reformer etc.
    """
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8
    ) -> None:
        super(Attention, self).__init__()
        d = dim // heads
        self.q, self.k, self.v = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(dim, d)
        self.norm = d ** 0.5
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.softmax(q @ torch.transpose(k, 1, 2) / self.norm, dim=1)
        qk = self.dropout(qk)
        attn = torch.matmul(qk, v)
        return attn


class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention, splitting it up to multiple Self-Attention layers and concatenating
    the results and subsequently running it through one linear layer of same dimension.
    """
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.self_attention_heads = nn.ModuleList([Attention(dim, heads) for _ in range(heads)])
        self.projector = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        for i, sa_head in enumerate(self.self_attention_heads):
            if i == 0:
                out = sa_head(x)
            else:
                out = torch.cat((out, sa_head(x)), axis=-1)
        out = self.projector(out)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 512
    ) -> None:
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """
    def __init__(
        self,
        dim: int = 768,
        hidden_dim: int = 3072,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1
    ) -> None:
        super(Encoder, self).__init__()
        self.MultiHeadAttention = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=attention_dropout)
        self.AttentionLN = nn.LayerNorm(dim, eps=1e-12)
        self.MlpLN = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        ])
        self.dropout = nn.Dropout(p=hidden_dropout)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        attn, _ = self.MultiHeadAttention(x, x, x, need_weights=False)
        attn = self.dropout(attn)
        x = x.add(attn)
        x = self.AttentionLN(x)
        mlp = self.MLP(x)
        x = x.add(mlp)
        x = self.MlpLN(x)
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(
        self,
        num_image_tokens: int,
        num_codebook_vectors: int,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        n_layers: int
    ) -> None:
        super(BidirectionalTransformer, self).__init__()
        self.num_image_tokens = num_image_tokens
        self.tok_emb = nn.Embedding(num_codebook_vectors, dim)
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.num_image_tokens, dim)), 0., 0.02)
        self.blocks = nn.Sequential(*[Encoder(dim, hidden_dim, num_heads, attention_dropout, hidden_dropout) for _ in range(n_layers)])
        self.Token_Prediction = nn.Sequential(*[
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        ])
        self.bias = nn.Parameter(torch.zeros(num_codebook_vectors))
        self.emb_ln = nn.LayerNorm(dim, eps=1e-12)
        self.drop = nn.Dropout(p=0.1)
        self.apply(weights_init)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = self.tok_emb(x)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:t, :]
        comb_embed = token_embeddings + position_embeddings
        embed_ln = self.emb_ln(comb_embed)
        embed = self.drop(embed_ln)
        embed = self.blocks(embed)
        embed = self.Token_Prediction(embed)
        logits = torch.matmul(embed, self.tok_emb.weight.T)
        logits = logits + self.bias[None]
        return logits
