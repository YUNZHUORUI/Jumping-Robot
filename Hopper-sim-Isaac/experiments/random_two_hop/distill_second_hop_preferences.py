"""Distill best-of-N second-hop actions into the long-hop adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--pairs", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=7)
args = parser.parse_args()


class HopActorCritic(nn.Module):
    def __init__(self, obs_dim=69, action_dim=4):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, action_dim))
        self.second_hop_adapter = nn.Sequential(nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, action_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), -3.0))

    def adapter(self, obs):
        return self.second_hop_adapter(obs)

    def mean(self, obs):
        return self.actor(obs) + obs[:, -2:-1] * self.adapter(obs)


def load_compat(model, state):
    current = model.state_dict()
    for key in ("actor.0.weight", "critic.0.weight"):
        if state[key].shape != current[key].shape:
            expanded = torch.zeros_like(current[key])
            expanded[:, :state[key].shape[1]] = state[key]
            state[key] = expanded
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any(not key.startswith("second_hop_adapter.") for key in missing) or unexpected:
        raise ValueError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")


def main():
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.load(Path(args.checkpoint).resolve(), map_location=device, weights_only=False)
    data = torch.load(Path(args.pairs).resolve(), map_location=device, weights_only=False)
    model = HopActorCritic().to(device)
    load_compat(model, source["model_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.second_hop_adapter.parameters():
        parameter.requires_grad_(True)

    obs = data["observation"]
    preferred = data["preferred_action"]
    rejected = data["rejected_action"]
    count = len(obs)
    permutation = torch.randperm(count, device=device)
    split = int(0.8 * count)
    train_ids, val_ids = permutation[:split], permutation[split:]
    optimizer = torch.optim.Adam(model.second_hop_adapter.parameters(), lr=args.lr, weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        shuffled = train_ids[torch.randperm(len(train_ids), device=device)]
        for ids in shuffled.split(args.batch_size):
            prediction = model.mean(obs[ids])
            preferred_distance = (prediction - preferred[ids]).square().mean(dim=1)
            rejected_distance = (prediction - rejected[ids]).square().mean(dim=1)
            # Directly imitate the best action while explicitly requiring it
            # to be closer than the rejected action from the same state.
            imitation = preferred_distance.mean()
            ranking = torch.relu(0.01 + preferred_distance - rejected_distance).mean()
            adapter_l2 = model.adapter(obs[ids]).square().mean()
            loss = imitation + 0.5 * ranking + 0.01 * adapter_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.second_hop_adapter.parameters(), 1.0)
            optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            with torch.no_grad():
                prediction = model.mean(obs[val_ids])
                pref_mse = (prediction - preferred[val_ids]).square().mean(dim=1)
                reject_mse = (prediction - rejected[val_ids]).square().mean(dim=1)
                accuracy = (pref_mse < reject_mse).float().mean()
                print(f"[DISTILL] epoch={epoch}/{args.epochs} val_mse={pref_mse.mean().item():.6f} "
                      f"preference_accuracy={accuracy.item():.4f} adapter_abs={model.adapter(obs[val_ids]).abs().mean().item():.4f}", flush=True)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "update": int(source.get("update", 0)),
        "infos": {
            "method": "best-of-8 second-hop preference distillation",
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "pairs": str(Path(args.pairs).resolve()),
        },
    }, output)
    print(f"[DISTILL] saved={output}", flush=True)


if __name__ == "__main__":
    main()
