from pathlib import Path

import torch

from .moe_duns import DUMoE


def model_generator(method="dumoe", pretrained_model_path=None, device="cpu"):
    if method != "dumoe":
        raise ValueError("Only DUMoE is supported.")
    model = DUMoE(depth=5)
    if pretrained_model_path is not None:
        path = Path(pretrained_model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}. Extract the pretrained weights first."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(
            {key.removeprefix("module."): value for key, value in state.items()}, strict=True
        )
    return model.to(device)
