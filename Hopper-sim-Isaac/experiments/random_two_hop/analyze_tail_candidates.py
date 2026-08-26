"""Analyze oracle tail-risk reduction in grouped two-hop candidate datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--tail_threshold", type=float, default=0.10)
args = parser.parse_args()


def main():
    data = torch.load(Path(args.data).resolve(), map_location="cpu", weights_only=False)
    context_id = data["context_id"]
    candidate_id = data["candidate_id"]
    error = data["position"].clamp_min(0.0).sqrt() * 0.10
    score = data["score"]
    hit = data.get("pair_hit", data.get("hit")).float()
    contexts = torch.unique(context_id, sorted=True)
    base_errors = []
    oracle_errors = []
    oracle_scores = []
    base_hits = []
    oracle_hits = []
    tail_contexts = 0
    rescued_tail10 = 0
    rescued_tail15 = 0
    for context in contexts:
        ids = torch.where(context_id == context)[0]
        base_local = torch.where(candidate_id[ids] == 0)[0]
        if len(base_local) != 1:
            continue
        base = ids[base_local[0]]
        best_error = ids[torch.argmin(error[ids])]
        best_score = ids[torch.argmax(score[ids])]
        base_errors.append(error[base])
        oracle_errors.append(error[best_error])
        oracle_scores.append(error[best_score])
        base_hits.append(hit[base])
        oracle_hits.append(hit[best_error])
        if error[base] > args.tail_threshold:
            tail_contexts += 1
            rescued_tail10 += int(error[best_error] <= 0.10)
            rescued_tail15 += int(error[best_error] <= 0.15)
    base_errors = torch.stack(base_errors)
    oracle_errors = torch.stack(oracle_errors)
    oracle_scores = torch.stack(oracle_scores)
    base_hits = torch.stack(base_hits)
    oracle_hits = torch.stack(oracle_hits)
    print(f"[TAIL-ANALYZE] data={args.data}")
    print(f"[TAIL-ANALYZE] contexts={len(base_errors)} tail_contexts={tail_contexts}")
    print(f"[TAIL-ANALYZE] base_error={base_errors.mean().item():.6f}")
    print(f"[TAIL-ANALYZE] oracle_min_error={oracle_errors.mean().item():.6f}")
    print(f"[TAIL-ANALYZE] oracle_max_score_error={oracle_scores.mean().item():.6f}")
    print(f"[TAIL-ANALYZE] base_tail10={(base_errors > 0.10).float().mean().item():.6f}")
    print(f"[TAIL-ANALYZE] oracle_tail10={(oracle_errors > 0.10).float().mean().item():.6f}")
    print(f"[TAIL-ANALYZE] base_tail15={(base_errors > 0.15).float().mean().item():.6f}")
    print(f"[TAIL-ANALYZE] oracle_tail15={(oracle_errors > 0.15).float().mean().item():.6f}")
    print(f"[TAIL-ANALYZE] base_hit={base_hits.mean().item():.6f}")
    print(f"[TAIL-ANALYZE] oracle_min_error_hit={oracle_hits.mean().item():.6f}")
    if tail_contexts:
        print(f"[TAIL-ANALYZE] tail_rescued_to_10={rescued_tail10 / tail_contexts:.6f}")
        print(f"[TAIL-ANALYZE] tail_rescued_to_15={rescued_tail15 / tail_contexts:.6f}")


if __name__ == "__main__":
    main()
