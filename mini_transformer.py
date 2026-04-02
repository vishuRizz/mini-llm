from src.mini_llm.model import build_tiny_decoder_only_transformer

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * normalized_x + self.beta


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, sequence_length, _ = x.size()

        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        output = self.output_proj(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = LayerNorm(normalized_shape=d_model)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        sublayer_output = self.dropout(sublayer_output)
        added_output = x + sublayer_output
        return self.norm(added_output)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.masked_self_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.residual_conn1 = ResidualConnection(d_model, dropout_rate)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        self.residual_conn2 = ResidualConnection(d_model, dropout_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_input = x
        attn_output = self.masked_self_attention(x, mask)
        x = self.residual_conn1(attn_input, attn_output)
        ffn_input = x
        ffn_output = self.feed_forward(x)
        return self.residual_conn2(ffn_input, ffn_output)


class FinalLinearOutput(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int = 512,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout_rate)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        )

        self.final_linear = FinalLinearOutput(d_model=d_model, vocab_size=vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = token_ids.size()

        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=token_ids.device)
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for block in self.decoder_blocks:
            x = block(x, mask=causal_mask)

        logits = self.final_linear(x)
        return logits


def build_tiny_decoder_only_transformer(
    vocab_size: int,
    max_len: int,
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    d_ff: int = 512,
    dropout_rate: float = 0.1,
) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout_rate=dropout_rate,
    )

