"""Pick CUDA, else MPS, else CPU, and attach it to HF model_args."""

from __future__ import annotations


def available_device() -> str:
    """CUDA if present, else MPS, else CPU. Run lists should not hardcode a machine."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def apply_device(model_args, device: str):
    """Add device to HF model_args unless the configuration already set one."""
    if isinstance(model_args, dict):
        if "device" in model_args:
            return model_args
        patched = dict(model_args)
        patched["device"] = device
        return patched
    text = model_args or ""
    if "device=" in str(text):
        return text
    if not text:
        return f"device={device}"
    return f"{text},device={device}"
