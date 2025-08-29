# utils_feature.py
import torch
from typing import Any

def extract_feature_tensor(out: Any) -> torch.Tensor:

    if isinstance(out, torch.Tensor):

        if out.dim() == 4:
            return out.mean(dim=[2, 3])  # (B, C)

        if out.dim() == 2:
            return out

        if out.dim() == 3:
            return out[:, 0, :]
        raise ValueError(f"Unsupported tensor shape: {tuple(out.shape)}")

    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output  # (B, D)
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:

        return out.last_hidden_state[:, 0, :]  # (B, D)
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):

        t = out[0]
        return t[:, 0, :] if t.dim() == 3 else t
    
    if isinstance(out, dict):
        for k in ("pooler_output", "last_hidden_state"):
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k] if k == "pooler_output" else out[k][:, 0, :]

        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v if v.dim() == 2 else v[:, 0, :]

    raise TypeError(f"Cannot extract features from type: {type(out)}")
