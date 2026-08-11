from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch


OLD_OBS_DIM = 37
LEGACY_PLANNER_OBS_DIM = 42
NEW_OBS_DIM = 43
RECURRENT_INPUT_KEYS = ("memory_a.rnn.weight_ih_l0", "memory_c.rnn.weight_ih_l0")
LEGACY_FIXED_HEIGHT_COMMAND = 1.30 / 2.0


def migrate_stable_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Expand 37-D/42-D recurrent inputs to the 43-D height-horizon contract."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = deepcopy(checkpoint["model_state_dict"])
    migrated = False
    for key in RECURRENT_INPUT_KEYS:
        weight = state_dict[key]
        if weight.shape[1] == NEW_OBS_DIM:
            continue
        if weight.shape[1] not in (OLD_OBS_DIM, LEGACY_PLANNER_OBS_DIM):
            raise ValueError(
                f"{key} has observation width {weight.shape[1]}, expected 37, 42 or 43"
            )
        expanded = torch.zeros(weight.shape[0], NEW_OBS_DIM, dtype=weight.dtype)
        expanded[:, : weight.shape[1]] = weight
        if weight.shape[1] == LEGACY_PLANNER_OBS_DIM:
            # V10 saw only the constant 1.30/2 height value, so its height
            # column cannot encode command conditioning. Fold that constant
            # contribution into the LSTM input bias, then initialize both new
            # height channels at zero. Their weights can now learn H_t/H_t+1
            # without an arbitrary fixed-height correlation.
            bias_key = key.replace("weight_ih_l0", "bias_ih_l0")
            state_dict[bias_key] = (
                state_dict[bias_key] + weight[:, 41] * LEGACY_FIXED_HEIGHT_COMMAND
            )
            expanded[:, 41] = 0.0
            expanded[:, 42] = 0.0
        state_dict[key] = expanded
        migrated = True
    output = {
        "model_state_dict": state_dict,
        "iter": 0 if migrated else checkpoint.get("iter", 0),
        "infos": {
            "source_checkpoint": str(source),
            "observation_migration": (
                "43-D: stable37 + Pt_xy + Pt1_xy + H_t + H_t1; "
                "legacy fixed 1.30 m H_t contribution folded into LSTM bias; "
                "new H_t/H_t1 weights initialized to zero"
            ),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    return destination
