from pathlib import Path

import torch

from moe_duns import DUMoE


def dumoe(ratio, pretrained=False, checkpoint_path=None):
    model = DUMoE(ratio=ratio, dim=32, mult=1.5, depth=5)
    if pretrained:
        path = (
            Path(checkpoint_path)
            if checkpoint_path
            else Path(__file__).resolve().parent / "model" / f"checkpoint-dumoe-{ratio}-best.pth"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}. Extract the pretrained weights first."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
    return model
