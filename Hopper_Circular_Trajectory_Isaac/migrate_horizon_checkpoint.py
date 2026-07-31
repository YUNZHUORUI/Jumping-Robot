import argparse
import os

import torch


def expand_first_layer(state_dict, key, old_obs_dim, new_obs_dim):
    weight = state_dict[key]
    if weight.shape[1] == new_obs_dim:
        return
    if weight.shape[1] != old_obs_dim:
        raise ValueError(f"{key} has input dim {weight.shape[1]}, expected {old_obs_dim}")
    expanded = torch.zeros(weight.shape[0], new_obs_dim, dtype=weight.dtype, device=weight.device)
    expanded[:, :old_obs_dim] = weight
    state_dict[key] = expanded


def main():
    parser = argparse.ArgumentParser(description="Migrate 40-dim circular hopper checkpoints to horizon observations.")
    parser.add_argument("--input", required=True, help="Source 40-dim checkpoint")
    parser.add_argument("--output", required=True, help="Output 50-dim checkpoint")
    parser.add_argument("--old_obs_dim", type=int, default=40)
    parser.add_argument("--new_obs_dim", type=int, default=50)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    expand_first_layer(state_dict, "actor.0.weight", args.old_obs_dim, args.new_obs_dim)
    expand_first_layer(state_dict, "critic.0.weight", args.old_obs_dim, args.new_obs_dim)

    migrated = {
        "model_state_dict": state_dict,
        "iter": checkpoint.get("iter", 0),
        "infos": checkpoint.get("infos"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(migrated, args.output)
    print(f"saved migrated checkpoint: {args.output}")


if __name__ == "__main__":
    main()
