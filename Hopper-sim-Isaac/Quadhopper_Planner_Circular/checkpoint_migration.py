from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch


OLD_OBS_DIM = 37
NEW_OBS_DIM = 42
RECURRENT_INPUT_KEYS = ("memory_a.rnn.weight_ih_l0", "memory_c.rnn.weight_ih_l0")


def migrate_stable_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Expand only the recurrent observation inputs; preserve all learned behavior."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = deepcopy(checkpoint["model_state_dict"])
    migrated = False
    for key in RECURRENT_INPUT_KEYS:
        weight = state_dict[key]
        if weight.shape[1] == NEW_OBS_DIM:
            continue
        if weight.shape[1] != OLD_OBS_DIM:
            raise ValueError(f"{key} has observation width {weight.shape[1]}, expected 37 or 42")
        expanded = torch.zeros(weight.shape[0], NEW_OBS_DIM, dtype=weight.dtype)
        expanded[:, :OLD_OBS_DIM] = weight
        state_dict[key] = expanded
        migrated = True
    output = {
        "model_state_dict": state_dict,
        "iter": 0 if migrated else checkpoint.get("iter", 0),
        "infos": {
            "source_checkpoint": str(source),
            "observation_migration": "37 stable dims + 5 zero-initialized planner dims",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    return destination
