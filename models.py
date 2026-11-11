"""Lightweight temporal models for fixed-length KSL keypoint sequences."""

from __future__ import annotations

import torch
from torch import nn


class BiGRUAttention(nn.Module):
    """Bidirectional GRU classifier with additive temporal attention."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_dim: int = 450,
        embedding_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_frames: int = 200,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or input_dim <= 0 or max_frames <= 0:
            raise ValueError("num_classes, input_dim, and max_frames must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = input_dim
        self.max_frames = max_frames
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, frames, features), got {tuple(inputs.shape)}."
            )
        if inputs.shape[1] != self.max_frames or inputs.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch, {self.max_frames}, {self.input_dim}), "
                f"got {tuple(inputs.shape)}."
            )

        projected = self.input_norm(self.input_projection(inputs))
        outputs, _ = self.gru(projected)
        scores = self.attention_score(torch.tanh(self.attention_projection(outputs)))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(outputs * weights, dim=1)
        return self.classifier(context)


class SmallTransformer(nn.Module):
    """Transformer encoder classifier using a learned classification token."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_dim: int = 450,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_frames: int = 200,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or input_dim <= 0 or max_frames <= 0:
            raise ValueError("num_classes, input_dim, and max_frames must be positive.")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = input_dim
        self.max_frames = max_frames
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.positional_embedding = nn.Parameter(
            torch.empty(1, max_frames + 1, embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, frames, features), got {tuple(inputs.shape)}."
            )
        if inputs.shape[1] != self.max_frames or inputs.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch, {self.max_frames}, {self.input_dim}), "
                f"got {tuple(inputs.shape)}."
            )

        projected = self.input_projection(inputs)
        cls_tokens = self.cls_token.expand(inputs.shape[0], -1, -1)
        sequence = torch.cat((cls_tokens, projected), dim=1)
        sequence = sequence + self.positional_embedding
        encoded = self.encoder(sequence)
        return self.classifier(encoded[:, 0])
