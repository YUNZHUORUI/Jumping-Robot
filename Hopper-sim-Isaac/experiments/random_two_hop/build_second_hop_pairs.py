"""Convert grouped candidate rollouts into explicit preferred/rejected pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--min_candidates", type=int, default=4)
args = parser.parse_args()


def main():
    data = torch.load(Path(args.input).resolve(), map_location="cpu", weights_only=False)
    pairs = {key: [] for key in (
        "context_id", "observation", "preferred_action", "rejected_action",
        "preferred_score", "rejected_score", "preferred_hit", "rejected_hit",
        "action_delta", "score_margin",
    )}
    baseline_hits = []
    oracle_hits = []
    for context_id in torch.unique(data["context_id"], sorted=True):
        ids = torch.where(data["context_id"] == context_id)[0]
        if len(ids) < args.min_candidates:
            continue
        # A hit dominates any shaped-score difference. Within equal hit
        # labels, the dense touchdown-state score chooses the better state.
        rank = data["hit"][ids] * 1000.0 + data["score"][ids]
        preferred = ids[torch.argmax(rank)]
        rejected = ids[torch.argmin(rank)]
        baseline = ids[torch.argmin(data["candidate_id"][ids])]
        pairs["context_id"].append(context_id.reshape(1))
        pairs["observation"].append(data["observation"][preferred:preferred + 1])
        pairs["preferred_action"].append(data["action"][preferred:preferred + 1])
        pairs["rejected_action"].append(data["action"][rejected:rejected + 1])
        pairs["preferred_score"].append(data["score"][preferred:preferred + 1])
        pairs["rejected_score"].append(data["score"][rejected:rejected + 1])
        pairs["preferred_hit"].append(data["hit"][preferred:preferred + 1])
        pairs["rejected_hit"].append(data["hit"][rejected:rejected + 1])
        pairs["action_delta"].append((data["action"][preferred] - data["action"][rejected]).unsqueeze(0))
        pairs["score_margin"].append((rank[torch.argmax(rank)] - rank[torch.argmin(rank)]).reshape(1))
        baseline_hits.append(data["hit"][baseline])
        oracle_hits.append(data["hit"][preferred])

    output = {key: torch.cat(value) for key, value in pairs.items()}
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    baseline_rate = torch.stack(baseline_hits).float().mean().item()
    oracle_rate = torch.stack(oracle_hits).float().mean().item()
    print(f"[PAIRS] contexts={len(output['context_id'])}")
    print(f"[PAIRS] baseline_candidate_hit_rate={baseline_rate:.6f}")
    print(f"[PAIRS] best_of_group_hit_rate={oracle_rate:.6f}")
    print(f"[PAIRS] mean_score_margin={output['score_margin'].mean().item():.6f}")
    print(f"[PAIRS] saved={path}")


if __name__ == "__main__":
    main()
