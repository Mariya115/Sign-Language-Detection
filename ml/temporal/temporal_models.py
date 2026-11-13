"""
Temporal Sequence Architectures for Dynamic Indian Sign Language Recognition
Includes Bidirectional GRU, Multi-layer LSTM, and Transformer Encoder.
"""

import math
from collections import deque
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class ISLBiGRU(nn.Module):
    """
    Bidirectional Gated Recurrent Unit (Bi-GRU) for sequence gesture modeling.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 26, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, Feature_Dim)
        out, _ = self.gru(x)
        # Global temporal average pooling across sequence
        pooled = torch.mean(out, dim=1)
        return self.fc(pooled)


class ISLLSTM(nn.Module):
    """
    Multi-layer Long Short-Term Memory (LSTM) for dynamic sign language classification.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 26, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # Last time-step representation
        return self.fc(last_step)


class ISLTransformer(nn.Module):
    """
    Temporal Transformer Encoder using self-attention across frames.
    """

    def __init__(self, input_dim: int = 512, d_model: int = 128, nhead: int = 4, num_layers: int = 2, num_classes: int = 26, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        proj = self.input_proj(x)
        encoded = self.pos_encoder(proj)
        out = self.transformer(encoded)
        pooled = torch.mean(out, dim=1)  # Mean pool across time tokens
        return self.classifier(pooled)


class TemporalSequenceBuffer:
    """
    Sliding window buffer maintaining recent visual embeddings from consecutive frames.
    """

    def __init__(self, sequence_length: int = 16, feature_dim: int = 512):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=sequence_length)

    def add_frame_features(self, feature_tensor: torch.Tensor):
        """Add a 1D feature embedding (feature_dim,) to the buffer."""
        feat = feature_tensor.squeeze().detach().cpu()
        self.buffer.append(feat)

    def is_ready(self) -> bool:
        """Check if buffer has accumulated sufficient frames for dynamic recognition."""
        return len(self.buffer) == self.sequence_length

    def get_sequence_tensor(self) -> Optional[torch.Tensor]:
        """
        Return stacked sequence tensor of shape (1, sequence_length, feature_dim).
        Returns None if buffer is not full.
        """
        if not self.is_ready():
            return None
        stacked = torch.stack(list(self.buffer), dim=0).unsqueeze(0)
        return stacked

    def clear(self):
        """Reset sequence buffer."""
        self.buffer.clear()

