---
name: quadhopper-baseline
description: Understand, reuse, copy, and extend Terry's Quadhopper jumping-robot baseline built with Isaac Lab and RSL-RL. Use when creating a new Quadhopper task or project, changing its trajectory optimization, jumping, waypoint, obstacle, ring-traversal/钻环, reward, observation, command, curriculum, training, evaluation, deployment, USD model, motor dynamics, or domain-randomization code, or when explaining/debugging the existing Hopper-sim-Isaac/Quadhopper_Isaac baseline. Preserve the proven robot model and dynamics unless the user explicitly requests a physical-model change.
---

# Reuse the Quadhopper baseline

Treat the six files supplied in `/home/terry/Downloads` (`rsl_rl_ppo_cfg.py`, `Jump+Base.stl`, `Jump+Leg.stl`, `Create-Hopper-Model.py`, `quadhopper_env.py`, and `my_hopper_cfg.py`) as the canonical stable-jump training source. Their frozen, runnable copy is `../../Quadhopper_Stable`. Git branch `experiment/higher-jump` is an older snapshot and differs in motor and power-model parameters. Do not fall back to the legacy main-branch `QuadhopperAsset.usd`.

## Start every task

1. Read [references/architecture.md](references/architecture.md) for ownership and data flow.
2. Read [references/baseline-contract.md](references/baseline-contract.md) before copying or editing physics, observations, actions, or PPO settings.
3. Read [references/workflows.md](references/workflows.md) when creating a new task, training, playing, or deploying.
4. Read [references/task-extension.md](references/task-extension.md) for trajectory, waypoint, obstacle, or ring-traversal work.
5. Inspect the current source files before editing. The references describe the captured baseline but source code remains authoritative.

## Preserve the baseline

- Prefer creating a sibling task package or new environment/config classes. Do not overwrite the working `myhopper` task merely to prototype another objective.
- Copy the model assets, asset configuration, motor mapping, action convention, domain randomization, and PPO baseline together; they form one coupled contract.
- Keep task-specific commands, observations, rewards, termination, curriculum, and visualization separate from robot physics.
- State exactly which baseline components remain unchanged and which task components change.
- Update observation dimensions, policy inputs, environment registration, runner experiment name, train/play task IDs, and export metadata together.
- Validate tensor shapes and paths statically before launching Isaac Sim. Then smoke-test with few environments before a full run.

## Handle ambiguity and drift

Do not silently guess frame conventions, quaternion order, motor order, asset prim paths, or checkpoint compatibility. Trace them from code/USD and record the decision.

The captured tree contains known path/export inconsistencies listed in `baseline-contract.md`. Recheck them rather than propagating them into a new project. Historical logs prove an earlier hopper variant trained, but do not prove every current checked-in file is mutually consistent.

## Expected handoff

For a new task, report:

- source baseline and destination paths;
- unchanged physical/dynamics contract;
- new command, observation, reward, reset, termination, and curriculum definitions;
- observation/action dimensions and frame conventions;
- train/play commands and checkpoint compatibility;
- validation performed and unresolved Isaac Sim runtime checks.
