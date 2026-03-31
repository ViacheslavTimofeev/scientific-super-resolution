from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class BicubicUpsampler(nn.Module):
    def __init__(
        self,
        scale_factor: int,
        *,
        align_corners: bool = False,
    ) -> None:
        super().__init__()

        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}.")

        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected input tensor in NCHW format. "
                f"Got tensor with shape {tuple(x.shape)}."
            )

        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=self.align_corners,
        )
