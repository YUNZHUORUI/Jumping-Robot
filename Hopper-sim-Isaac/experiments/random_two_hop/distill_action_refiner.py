"""Distill local action-search winners into the Semi-MDP hop policy.

The default mode freezes the shared actor and updates only the long-hop adapter,
so short-hop behavior stays exactly unchanged.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--target_metric", choices=("score", "position"), default="score")
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--anchor_weight", type=float, default=0.25)
parser.add_argument("--adapter_only", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=17)
args = parser.parse_args()


class HopActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 69, action_dim: int = 4, init_log_std: float = math.log(0.20)):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.second_hop_adapter = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def mean(self, obs):
        long_hop = obs[:, -2:-1]
        return self.actor(obs) + long_hop * self.second_hop_adapter(obs)


def load_model_state_compat(model: nn.Module, state: dict[str, torch.Tensor]) -> bool:
    current = model.state_dict()
    migrated = False
    for key in ("actor.0.weight", "critic.0.weight"):
        if key in state and state[key].shape != current[key].shape:
            if state[key].shape[0] != current[key].shape[0] or state[key].shape[1] > current[key].shape[1]:
                raise ValueError(f"Cannot migrate {key}: {state[key].shape} -> {current[key].shape}")
            expanded = torch.zeros_like(current[key])
            expanded[:, : state[key].shape[1]] = state[key]
            state[key] = expanded
            migrated = True
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {key for key in missing if key.startswith("second_hop_adapter.")}
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    return migrated


def best_rows_by_context(data: dict[str, torch.Tensor]) -> torch.Tensor:
    selected = []
    context_ids = torch.unique(data["context_id"], sorted=True)
    for context in context_ids:
        ids = torch.where(data["context_id"] == context)[0]
        if args.target_metric == "position":
            best = ids[data["position"][ids].argmin()]
        else:
            best = ids[data["score"][ids].argmax()]
        selected.append(best)
    return torch.stack(selected)


def main():
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(Path(args.data).expanduser().resolve(), map_location=device, weights_only=False)
    checkpoint = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location=device, weights_only=False)

    model = HopActorCritic().to(device)
    load_model_state_compat(model, checkpoint["model_state_dict"])
    model.train()
    reference = HopActorCritic().to(device)
    load_model_state_compat(reference, checkpoint["model_state_dict"])
    reference.eval()

    if args.adapter_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("second_hop_adapter."))

    rows = best_rows_by_context(data)
    rows = rows[torch.randperm(len(rows), device=device)]
    split = int(0.8 * len(rows))
    train_rows, val_rows = rows[:split], rows[split:]
    obs = data["observation"]
    target_action = data["action"].clamp(-1.0, 1.0)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        shuffled = train_rows[torch.randperm(len(train_rows), device=device)]
        for ids in shuffled.split(args.batch_size):
            predicted = model.mean(obs[ids]).clamp(-1.0, 1.0)
            with torch.no_grad():
                original = reference.mean(obs[ids]).clamp(-1.0, 1.0)
            imitation = nn.functional.mse_loss(predicted, target_action[ids])
            anchor = nn.functional.mse_loss(predicted, original)
            loss = imitation + args.anchor_weight * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            with torch.no_grad():
                val_predicted = model.mean(obs[val_rows]).clamp(-1.0, 1.0)
                val_original = reference.mean(obs[val_rows]).clamp(-1.0, 1.0)
                val_imitation = nn.functional.mse_loss(val_predicted, target_action[val_rows])
                val_anchor = nn.functional.mse_loss(val_predicted, val_original)
                val_loss = val_imitation + args.anchor_weight * val_anchor
            if val_loss.item() < best_val:
                best_val = val_loss.item()
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"[DISTILL] epoch={epoch}/{args.epochs} "
                f"val_loss={val_loss.item():.6f} "
                f"val_imitation={val_imitation.item():.6f} "
                f"val_anchor={val_anchor.item():.6f}",
                flush=True,
            )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "update": int(checkpoint.get("update", 0)),
        "infos": {
            "method": "local action-search distillation",
            "source_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "source_data": str(Path(args.data).expanduser().resolve()),
            "target_metric": args.target_metric,
            "adapter_only": args.adapter_only,
            "anchor_weight": args.anchor_weight,
            "best_val_loss": best_val,
        },
    }, output)
    print(f"[DISTILL] saved={output} best_val_loss={best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
