from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from torch import nn

from src.models.bicubic import BicubicUpsampler
from src.models.cnn import RLFN
from src.models.swinir_wrapper import SwinIR


ModelKind = Literal["bicubic", "cnn", "swinir"]
ModelBuilder = Callable[..., nn.Module]


def _build_bicubic(**kwargs: Any) -> nn.Module:
    return BicubicUpsampler(**kwargs)


def _build_cnn(**kwargs: Any) -> nn.Module:
    return RLFN(**kwargs)


def _build_swinir(**kwargs: Any) -> nn.Module:
    return SwinIR(**kwargs)


MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "bicubic": _build_bicubic,
    "cnn": _build_cnn,
    "swinir": _build_swinir,
}


def build_model(
    model_kind: ModelKind,
    /,
    **model_kwargs: Any,
) -> nn.Module:
    builder = MODEL_BUILDERS.get(model_kind)
    if builder is None:
        available = ", ".join(sorted(MODEL_BUILDERS))
        raise KeyError(
            f"Unknown model_kind '{model_kind}'. Available models: {available}."
        )

    return builder(**model_kwargs)
