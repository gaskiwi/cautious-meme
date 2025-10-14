# URDF Robot Models

This directory contains URDF (Unified Robot Description Format) models for the RL robotics environment.

## Models

### 1. Bar with Ball Joint (`bar_with_joint.urdf`)

A bar model with a ball-and-socket joint at its midpoint, allowing for 3 degrees of freedom rotation.

**Specifications:**
- **Total length**: 1.0m (fully extended)
- **Segment length**: 0.5m each (2 segments)
- **Bar radius**: 0.015m
- **Joint type**: Spherical (ball and socket)
- **Joint location**: Center of the model
- **Mass**: 0.5kg per segment, 0.1kg for center joint

**Components:**
- `base`: Central sphere representing the ball joint (radius: 0.025m)
- `bar_1`: First bar segment extending from center
- `bar_2`: Second bar segment extending opposite direction
- `ball_joint_1`, `ball_joint_2`: Spherical joints providing 3-DOF rotation

### 2. Rolling Sphere (`rolling_sphere.urdf`)

A simple sphere designed for rolling physics simulation.

**Specifications:**
- **Diameter**: 0.334m (~1/3 of bar length)
- **Radius**: 0.167m
- **Mass**: 1.0kg
- **Inertia**: Calculated for solid sphere (I = 2/5 * m * r²)

**Dimensional Relationship:**
- Sphere diameter = 1/3 × Bar total length
- 0.334m ≈ 1/3 × 1.0m ✓

## Usage in PyBullet

```python
import pybullet as p
import pybullet_data

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load models
bar_id = p.loadURDF("models/bar_with_joint.urdf", [0, 0, 1])
sphere_id = p.loadURDF("models/rolling_sphere.urdf", [2, 0, 0.5])

# Run simulation
for _ in range(1000):
    p.stepSimulation()
```

## Multi-Instance Support

These models are designed to be instantiated multiple times in the same environment. Each instance will have its own physics state and can be loaded at different positions:

```python
# Load multiple instances
bars = []
spheres = []

for i in range(5):
    bar = p.loadURDF("models/bar_with_joint.urdf", [i*2, 0, 1])
    sphere = p.loadURDF("models/rolling_sphere.urdf", [i*2, 2, 0.5])
    bars.append(bar)
    spheres.append(sphere)
```

## Notes

- All models use SI units (meters, kilograms)
- Inertia tensors are calculated based on geometry and mass distribution
- Color materials are defined for visualization (red for joints, blue for bars, green for spheres)
- Collision geometries match visual geometries for accurate physics simulation
