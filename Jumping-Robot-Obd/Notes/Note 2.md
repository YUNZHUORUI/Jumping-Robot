# Research Idea: Passive Stable Landing and Spring-assisted Relaunch for Quadhopper

## Inspiration

The idea is inspired by the recent hopping robot paper, which uses a spring-based actuation mechanism to preload elastic energy during flight and rapidly release it for the next jump.
![[../Image/reactive_latch_hopping_robot_mechanism.png|Reactive latch hopping robot mechanism]]
The paper demonstrates an efficient hopping mechanism by combining:

- Motor-driven spring preload
- Mechanical latch
- Elastic energy release for high-power jumping

---

## Limitation of the Paper

Although the jumping mechanism is highly efficient, the robot still has several practical limitations:

- Unable to remain statically stable after landing.
- Cannot safely park on the ground.
- Cannot autonomously initiate jumping from a stable resting state.
- No passive anti-tip or impact protection.

These limitations are also acknowledged in the paper as future work.

---

# Proposed Idea

Instead of adding additional actuators, I propose adding **four passive compliant legs** around the Quadhopper body.


```
          Quadrotor
        /    |     \
       /     |      \
   Passive   |   Passive
    Leg      |      Leg
             |
      Main Spring Leg
```
```
正常跳跃
        Quadcopter
      ╱     │     ╲
  flexible  │  flexible
    leg   main leg
            │
            ●

落地压缩主腿
        Quadcopter
      ╱     |     \
     / compressed  ╲
     ●   main leg   ●
           ●
```


The compliant legs have two functions:

- Passive anti-collision protection
- Stable multi-point support during landing

---

## Landing Mode

Unlike normal hopping, when the robot intends to stop:

1. During the aerial phase, the **main spring leg is actively retracted** and locked.
2. The quadrotor provides partial thrust to perform a slow terminal descent (similar to Apollo lunar landing).
3. Since the main leg becomes shorter, the four passive compliant legs naturally touch the ground first.
4. The robot lands on four compliant legs and becomes statically stable.

<div align="center">

```
Normal Hopping

        |
        |
        ●

↓
s
Landing Mode

   ●         ●

       Robot

   ●         ●
```

</div>

No actuator is required for the four compliant legs.

---

## Relaunch

When the robot needs to move again:

1. Unlock the latch.
2. The stored elastic energy in the compressed spring is released.
3. The spring assists the initial takeoff.
4. Once airborne, the passive legs naturally leave the ground and no longer participate in locomotion.

---

# Advantages

Compared with the original design:

- Passive static stability after landing.
- Passive impact protection.
- Spring-assisted autonomous relaunch.
- No additional actuators for the support legs.
- Minimal increase in system complexity.

---

# Key Difference from the Reference Paper

The reference paper focuses on **efficient continuous hopping**.

This work extends the concept by introducing **multiple operating modes**:

- Hopping Mode
- Stable Landing Mode
- Spring-assisted Relaunch Mode

The proposed mechanism aims to address the practical limitations identified in the reference paper while preserving efficient hopping performance
