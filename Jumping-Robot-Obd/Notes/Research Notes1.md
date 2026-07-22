#  Research Ideas

## Self-righting Mechanism

### Inspiration

The Quadhopper structure resembles a roly-poly toy, which inspired the idea of using an internal flywheel (reaction wheel) for self-righting.

Reference video:

**How Do Gyroscopes Lift Themselves Up?**

https://www.youtube.com/watch?v=HL9RRLdcSZc

---

## Possible Recovery Methods

### 1. Rotor-assisted Recovery

Since Quadhopper already has four propellers, differential thrust may generate enough torque for self-righting.

Pros:
- No additional hardware
- No extra weight

Cons:
- Propellers have very small rotational inertia.
- The generated torque may be insufficient for self-erection.
- This needs to be verified experimentally.

---

### 2. Flywheel-assisted Recovery

Adding an internal flywheel may improve both self-righting capability and balance control.

Pros:
- Stronger recovery torque
- May improve stability during hopping

Cons:
- Additional weight
- Increased system complexity

The trade-off between recovery performance and added mass requires further investigation.

---

## Potential Applications

### Military

- Ground mobility helps avoid radar detection and air-defense systems.
- Suitable for reconnaissance or loitering-type missions.

### Cave Exploration

- Can switch between hopping on flat ground and controlled descent into caves or vertical shafts.
- Suitable for underground exploration where both aerial and ground mobility are beneficial.

---

## Future Work

- Experimentally compare rotor-assisted and flywheel-assisted recovery.
- Evaluate recovery success, added weight, energy consumption, and overall hopping performance.