"""Train a context-held-out second-hop state-action success scorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Quadhopper_Planner_Random.second_hop_scorer import SecondHopScorer

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--label_key", choices=("hit", "second_hit", "pair_hit"), default="hit")
parser.add_argument("--selection_quality_weight", type=float, default=0.10,
                    help="Weight on predicted touchdown quality when selecting within a context")
parser.add_argument("--classification_weight", type=float, default=1.0)
parser.add_argument("--quality_loss_weight", type=float, default=0.25)
parser.add_argument(
    "--model_selection_metric",
    choices=("top1_hit", "top1_score", "top1_error"),
    default="top1_hit",
    help="Held-out metric used to keep the best scorer weights.",
)
parser.add_argument(
    "--threshold_metric",
    choices=("hit", "score", "error"),
    default="hit",
    help="Held-out metric used to choose the conservative deployment threshold.",
)
parser.add_argument("--deployment_candidates", type=int, default=32)
parser.add_argument("--deployment_candidate_scale", type=float, default=0.20)
args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(Path(args.data).resolve(), map_location=device, weights_only=False)
    contexts = torch.unique(data["context_id"], sorted=True)
    contexts = contexts[torch.randperm(len(contexts), device=device)]
    split = int(0.8 * len(contexts))
    train_contexts, val_contexts = contexts[:split], contexts[split:]
    train_mask = torch.isin(data["context_id"], train_contexts)
    val_mask = torch.isin(data["context_id"], val_contexts)
    train_ids = torch.where(train_mask)[0]
    val_ids = torch.where(val_mask)[0]
    obs, action, hit = data["observation"], data["action"], data[args.label_key]
    score = data["score"]
    score_mean, score_std = score[train_ids].mean(), score[train_ids].std().clamp_min(1.0)
    quality = ((score - score_mean) / score_std).clamp(-5.0, 5.0)
    positive_weight = (1.0 - hit[train_ids].mean()) / hit[train_ids].mean().clamp_min(1e-4)
    landing_error = data["position"].clamp_min(0.0).sqrt() * 0.10

    model = SecondHopScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    best_metric = -float("inf")
    best_state = None

    def selected_context_metrics(contexts_subset):
        selected_hits = []
        selected_scores = []
        selected_errors = []
        for context in contexts_subset:
            ids = torch.where(data["context_id"] == context)[0]
            context_logit, context_quality = model(obs[ids], action[ids])
            selected = torch.argmax(context_logit + args.selection_quality_weight * context_quality)
            selected_hits.append(hit[ids[selected]])
            selected_scores.append(score[ids[selected]])
            selected_errors.append(landing_error[ids[selected]])
        return {
            "top1_hit": torch.stack(selected_hits).float().mean(),
            "top1_score": torch.stack(selected_scores).float().mean(),
            "top1_error": torch.stack(selected_errors).float().mean(),
        }

    for epoch in range(1, args.epochs + 1):
        shuffled = train_ids[torch.randperm(len(train_ids), device=device)]
        model.train()
        for ids in shuffled.split(args.batch_size):
            logit, predicted_quality = model(obs[ids], action[ids])
            classification = bce(logit, hit[ids])
            regression = nn.functional.smooth_l1_loss(predicted_quality, quality[ids])
            loss = args.classification_weight * classification + args.quality_loss_weight * regression
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                logit, predicted_quality = model(obs[val_ids], action[val_ids])
                probability = logit.sigmoid()
                accuracy = ((probability >= 0.5) == hit[val_ids].bool()).float().mean()
                selected = selected_context_metrics(val_contexts)
                top1_hit = selected["top1_hit"]
                top1_score = selected["top1_score"]
                top1_error = selected["top1_error"]
                current_metric = -top1_error if args.model_selection_metric == "top1_error" else selected[args.model_selection_metric]
                if current_metric.item() > best_metric:
                    best_metric = current_metric.item()
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                print(f"[SCORER] epoch={epoch}/{args.epochs} val_accuracy={accuracy.item():.4f} "
                      f"heldout_top1_hit={top1_hit.item():.4f} "
                      f"heldout_top1_error={top1_error.item():.4f} "
                      f"heldout_top1_score={top1_score.item():.4f}", flush=True)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    model.eval()
    # Conservative deployment: deviate from candidate zero only when the
    # learned logit improvement clears a held-out threshold.
    threshold_grid = torch.linspace(0.0, 5.0, 101, device=device)
    threshold_metrics = []
    with torch.no_grad():
        for threshold in threshold_grid:
            selected_hits = []
            selected_scores = []
            selected_errors = []
            for context in val_contexts:
                ids = torch.where(data["context_id"] == context)[0]
                logits, predicted_quality = model(obs[ids], action[ids])
                base_local = torch.where(data["candidate_id"][ids] == 0)[0]
                if len(base_local) != 1:
                    continue
                ranking = logits + args.selection_quality_weight * predicted_quality
                best_local = ranking.argmax()
                chosen = best_local if ranking[best_local] - ranking[base_local[0]] > threshold else base_local[0]
                selected_hits.append(hit[ids[chosen]])
                selected_scores.append(score[ids[chosen]])
                selected_errors.append(landing_error[ids[chosen]])
            hit_rate = torch.stack(selected_hits).float().mean()
            mean_score = torch.stack(selected_scores).float().mean()
            mean_error = torch.stack(selected_errors).float().mean()
            if args.threshold_metric == "hit":
                threshold_metrics.append(hit_rate)
            elif args.threshold_metric == "score":
                threshold_metrics.append(mean_score)
            else:
                threshold_metrics.append(-mean_error)
    threshold_metrics = torch.stack(threshold_metrics)
    threshold_index = threshold_metrics.argmax()
    selection_threshold = threshold_grid[threshold_index].item()
    gated_metric = threshold_metrics[threshold_index].item()
    # Deployment candidates are independent of any one collection batch.
    # Candidate zero always preserves the original planner action.
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 99173)
    candidate_offsets = torch.zeros(args.deployment_candidates, 4)
    if args.deployment_candidates > 1:
        candidate_offsets[1:] = args.deployment_candidate_scale * (
            2.0 * torch.rand(args.deployment_candidates - 1, 4, generator=generator) - 1.0
        )
    torch.save({"model_state_dict": best_state, "score_mean": score_mean.cpu(),
                "score_std": score_std.cpu(), "infos": {"heldout_contexts": len(val_contexts),
                "best_metric": best_metric, "label_key": args.label_key,
                "selection_quality_weight": args.selection_quality_weight,
                "classification_weight": args.classification_weight,
                "quality_loss_weight": args.quality_loss_weight,
                "model_selection_metric": args.model_selection_metric,
                "threshold_metric": args.threshold_metric,
                "gated_metric": gated_metric},
                "candidate_offsets": candidate_offsets,
                "selection_threshold": selection_threshold}, output)
    final_metrics = selected_context_metrics(val_contexts)
    print(f"[SCORER] best_metric={best_metric:.6f} "
          f"final_top1_hit={final_metrics['top1_hit'].item():.6f} "
          f"final_top1_error={final_metrics['top1_error'].item():.6f} "
          f"final_top1_score={final_metrics['top1_score'].item():.6f} "
          f"gated_metric={gated_metric:.6f} "
          f"threshold={selection_threshold:.3f} saved={output}", flush=True)


if __name__ == "__main__":
    main()
