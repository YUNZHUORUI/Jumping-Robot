#!/usr/bin/env python3
"""Export the accepted Quadhopper v22 recurrent policy for real-robot inference."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from rsl_rl.modules import ActorCriticRecurrent

try:
    import onnxruntime as ort
except ImportError:  # Report a focused setup error when export verification is requested.
    ort = None


OBSERVATIONS = [
    {"slice": [0, 3], "name": "linear_velocity_body", "frame": "body", "units": "m/s"},
    {"slice": [3, 6], "name": "angular_velocity_body", "frame": "body", "units": "rad/s"},
    {"slice": [6, 10], "name": "root_quaternion_world", "order": "wxyz"},
    {"slice": [10, 13], "name": "planner_reference_position_error", "frame": "body", "units": "m"},
    {"slice": [13, 14], "name": "root_height_world", "units": "m"},
    {"slice": [14, 15], "name": "spring_contact", "definition": "spring_position > 0.002 m"},
    {"slice": [15, 16], "name": "spring_position", "units": "m"},
    {"slice": [16, 17], "name": "spring_velocity", "units": "m/s"},
    {"slice": [17, 37], "name": "last_five_raw_policy_actions", "layout": "oldest_to_newest, F1..F4"},
    {"slice": [37, 39], "name": "current_landing_xy_error_body", "scale": "divide by 0.22 m"},
    {"slice": [39, 41], "name": "next_landing_xy_error_body", "scale": "divide by 0.22 m"},
    {"slice": [41, 42], "name": "current_absolute_apex_command", "scale": "divide by 2.0 m"},
    {"slice": [42, 43], "name": "next_absolute_apex_command", "scale": "divide by 2.0 m"},
]


class RecurrentActorExporter(torch.nn.Module):
    def __init__(self, policy: ActorCriticRecurrent):
        super().__init__()
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.rnn = copy.deepcopy(policy.memory_a.rnn)
        self.actor = copy.deepcopy(policy.actor)

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.normalizer(obs)
        latent, (h_out, c_out) = self.rnn(normalized.unsqueeze(0), (h_in, c_in))
        return self.actor(latent.squeeze(0)), h_out, c_out


def export_onnx(policy: ActorCriticRecurrent, output_path: Path) -> None:
    exporter = RecurrentActorExporter(policy).cpu().eval()
    obs = torch.zeros(1, 43)
    h_in = torch.zeros(1, 1, 256)
    c_in = torch.zeros(1, 1, 256)
    torch.onnx.export(
        exporter,
        (obs, h_in, c_in),
        str(output_path),
        export_params=True,
        opset_version=18,
        input_names=["obs", "h_in", "c_in"],
        output_names=["actions", "h_out", "c_out"],
        dynamic_axes={},
        dynamo=False,
    )


def materialize_lstm_sequence_lengths(model_path: Path) -> None:
    """Replace the optional empty LSTM sequence-length input for strict runtimes."""
    model = onnx.load(model_path)
    changed = False
    for index, node in enumerate(model.graph.node):
        if node.op_type == "LSTM" and len(node.input) > 4 and not node.input[4]:
            name = f"lstm_sequence_lengths_{index}"
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray([1], dtype=np.int32), name=name)
            )
            node.input[4] = name
            changed = True
    if changed:
        onnx.save(model, model_path)


def build_policy(state_dict: dict[str, torch.Tensor]) -> ActorCriticRecurrent:
    has_actor_normalizer = any(key.startswith("actor_obs_normalizer.") for key in state_dict)
    has_critic_normalizer = any(key.startswith("critic_obs_normalizer.") for key in state_dict)
    observations = {"policy": torch.zeros(1, 43)}
    policy = ActorCriticRecurrent(
        obs=observations,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=4,
        actor_obs_normalization=has_actor_normalizer,
        critic_obs_normalization=has_critic_normalizer,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=0.1,
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def verify_onnx(policy: ActorCriticRecurrent, onnx_path: Path) -> float:
    if ort is None:
        raise RuntimeError(
            "onnxruntime is required to verify the exported recurrent policy; "
            "install deployment/requirements-runtime.txt"
        )
    generator = torch.Generator().manual_seed(20260811)
    obs = torch.randn(1, 43, generator=generator) * 0.1
    h_in = torch.randn(1, 1, 256, generator=generator) * 0.01
    c_in = torch.randn(1, 1, 256, generator=generator) * 0.01
    with torch.inference_mode():
        normalized = policy.actor_obs_normalizer(obs)
        latent, (h_out, c_out) = policy.memory_a.rnn(normalized.unsqueeze(0), (h_in, c_in))
        actions = policy.actor(latent.squeeze(0))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["actions", "h_out", "c_out"],
        {
            "obs": obs.numpy(),
            "h_in": h_in.numpy(),
            "c_in": c_in.numpy(),
        },
    )
    expected = (actions.numpy(), h_out.numpy(), c_out.numpy())
    return max(float(np.max(np.abs(got - want))) for got, want in zip(actual, expected, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = data["model_state_dict"]
    input_width = int(state_dict["memory_a.rnn.weight_ih_l0"].shape[1])
    if input_width != 43:
        raise ValueError(f"Expected a 43-D planner policy, got {input_width}")

    policy = build_policy(state_dict)
    normalizer = policy.actor_obs_normalizer if policy.actor_obs_normalization else None
    onnx_path = output_dir / "quadhopper_v22_policy.onnx"
    export_onnx(policy, onnx_path)
    materialize_lstm_sequence_lengths(onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path))
    max_error = verify_onnx(policy, onnx_path)
    if max_error > 1.0e-5:
        raise RuntimeError(f"ONNX parity failed: max absolute error {max_error}")

    shutil.copy2(checkpoint, output_dir / "quadhopper_v22_model_100.pt")
    metadata = {
        "source_checkpoint": str(checkpoint),
        "checkpoint_iteration": int(data.get("iter", -1)),
        "policy": "ActorCriticRecurrent/LSTM",
        "control_frequency_hz": 100,
        "inputs": {"obs": [1, 43], "h_in": [1, 1, 256], "c_in": [1, 1, 256]},
        "outputs": {"actions": [1, 4], "h_out": [1, 1, 256], "c_out": [1, 1, 256]},
        "normalization": "identity" if normalizer is None else "embedded empirical normalizer",
        "motor_order": ["F1", "F2", "F3", "F4"],
        "action_postprocess": "clip actions to [-1,1], target_u=clip(0.5*action+0.5,0,1)",
        "nominal_action_delay_samples": 3,
        "nominal_motor_time_constant_s": 0.125,
        "height_sequence_m": [0.7, 1.0],
        "hop_distance_m": 0.22,
        "onnx_opset": 18,
        "onnx_parity_max_abs_error": max_error,
        "observations": OBSERVATIONS,
    }
    (output_dir / "policy_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Exported {onnx_path}")
    print(f"ONNX parity max abs error: {max_error:.3e}")


if __name__ == "__main__":
    main()
