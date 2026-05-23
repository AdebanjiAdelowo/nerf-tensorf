import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding from the original NeRF paper.

    Maps each coordinate x to [x, sin(2^0 π x), cos(2^0 π x), ..., sin(2^(L-1) π x), cos(2^(L-1) π x)].
    """

    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs)
        self.out_dim = (2 * num_freqs + (1 if include_input else 0)) * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3)
        parts = [x] if self.include_input else []
        for freq in self.freqs:
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)
