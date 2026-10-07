"""Handwriting line recognition model: CNN -> sequence encoder -> CTC head.

The three stages are separate modules so each can be swapped independently:

    image (B,1,H,W)
      -> CNNBackbone          (computer vision: local stroke/shape features)
      -> height collapse + projection to a sequence of feature vectors (B,T,d_model)
      -> BiLSTMEncoder | TransformerSeqEncoder   (sequence modelling: context across the line)
      -> Linear head          (B,T,num_classes) logits for CTC

The model's ``forward`` takes only tensors and returns only a tensor, which keeps
ONNX export simple. Padding-aware behaviour (``widths``) is for batched training;
at inference time batch size is 1, so no padding and no lengths are needed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

WIDTH_STRIDE = 4  # the CNN downsamples the width axis by 4 (two 2x2 poolings)
HEIGHT_STRIDE = 16  # and the height axis by 16


@dataclass
class HTRConfig:
    num_classes: int  # including the CTC blank at index 0
    img_height: int = 64
    cnn_channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    d_model: int = 256
    encoder: Literal["bilstm", "transformer"] = "bilstm"
    num_layers: int = 2
    dropout: float = 0.2
    # transformer-only settings
    num_heads: int = 4
    ff_dim: int = 1024
    max_len: int = 4096  # longest sequence (time steps) the positional encoding supports

    def __post_init__(self) -> None:
        if self.img_height % HEIGHT_STRIDE != 0:
            raise ValueError(f"img_height must be a multiple of {HEIGHT_STRIDE}, got {self.img_height}.")
        if self.encoder not in ("bilstm", "transformer"):
            raise ValueError(f"Unknown encoder {self.encoder!r}; use 'bilstm' or 'transformer'.")
        if self.encoder == "bilstm" and self.d_model % 2 != 0:
            raise ValueError("d_model must be even for the BiLSTM encoder.")
        if self.encoder == "transformer" and self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads for the Transformer encoder.")

    def to_dict(self) -> dict:
        return asdict(self)


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CNNBackbone(nn.Module):
    """VGG-style feature extractor. Shrinks width by 4 and height by 16."""

    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.net = nn.Sequential(
            _conv_block(1, c0),
            nn.MaxPool2d(2, 2),  # H/2,  W/2
            _conv_block(c0, c1),
            nn.MaxPool2d(2, 2),  # H/4,  W/4
            _conv_block(c1, c2),
            _conv_block(c2, c2),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8,  W/4  (keep horizontal resolution)
            _conv_block(c2, c3),
            _conv_block(c3, c3),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM. ``d_model // 2`` units per direction so the output is ``d_model`` wide."""

    def __init__(self, d_model: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            d_model,
            d_model // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        if lengths is None:
            out, _ = self.lstm(x)
            return out
        # Pack so the backward direction does not read padding.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        t = x.size(1)
        if t > self.pe.size(0):
            raise ValueError(
                f"Sequence has {t} time steps but max_len is {self.pe.size(0)}. Increase HTRConfig.max_len."
            )
        return x + self.pe[:t].unsqueeze(0)


class TransformerSeqEncoder(nn.Module):
    """Self-attention encoder (the same building block used in NLP models)."""

    def __init__(self, d_model: int, num_layers: int, num_heads: int, ff_dim: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        key_padding_mask = None
        if lengths is not None:
            steps = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            key_padding_mask = steps >= lengths.to(x.device).unsqueeze(1)  # True = padding
        out = self.encoder(self.pos(x), src_key_padding_mask=key_padding_mask)
        return self.norm(out)


class HTRModel(nn.Module):
    def __init__(self, config: HTRConfig) -> None:
        super().__init__()
        self.config = config
        self.cnn = CNNBackbone(config.cnn_channels)
        feat_dim = config.cnn_channels[-1] * (config.img_height // HEIGHT_STRIDE)
        self.proj = nn.Sequential(nn.Linear(feat_dim, config.d_model), nn.Dropout(config.dropout))
        if config.encoder == "bilstm":
            self.encoder: nn.Module = BiLSTMEncoder(config.d_model, config.num_layers, config.dropout)
        else:
            self.encoder = TransformerSeqEncoder(
                config.d_model, config.num_layers, config.num_heads, config.ff_dim, config.dropout, config.max_len
            )
        self.head = nn.Linear(config.d_model, config.num_classes)

    @staticmethod
    def output_lengths(widths: Tensor) -> Tensor:
        """Number of CTC time steps produced for images with the given pixel widths."""
        return torch.div(widths, WIDTH_STRIDE, rounding_mode="floor")

    def forward(self, images: Tensor, widths: Optional[Tensor] = None) -> Tensor:
        """images: (B,1,H,W) float. widths: (B,) true pixel widths before padding (training only).

        Returns logits of shape (B,T,num_classes). Apply log_softmax and permute to (T,B,C) for CTCLoss.
        """
        if images.dim() != 4 or images.size(1) != 1 or images.size(2) != self.config.img_height:
            raise ValueError(
                f"Expected images of shape (B,1,{self.config.img_height},W), got {tuple(images.shape)}."
            )
        feats = self.cnn(images)  # (B,C,H',W')
        b, c, h, w = feats.shape
        seq = feats.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (B,T,C*H')
        seq = self.proj(seq)
        lengths = None
        if widths is not None:
            lengths = self.output_lengths(widths).clamp(min=1, max=w)
        seq = self.encoder(seq, lengths)
        return self.head(seq)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)