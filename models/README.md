# URDF Robot Models

This directory contains URDF (Unified Robot Description Format) models for the RL robotics environment.

## Robot Types

The environment uses two types of robots that can work together to build structures:

- **Type A**: Bar robots with joints (`bar_with_joint.urdf`)
- **Type B**: Sphere robots (`rolling_sphere.urdf`)

In each training episode, N Type B robots and 2N Type A robots are deployed randomly on the plane.

## Models

### Type A: Bar with Ball Joint (`bar_with_joint.urdf`)

A bar robot with a central base and two bar segments connected via spherical (ball-and-socket) joints, allowing for 3 degrees of freedom rotation per joint. The endpoints of the bars can connect to Type B robots.

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

**Key Features:**
- **Endpoints can connect to Type B robots** for collaborative structure building
- **Motor force**: Configured to lift 2× body weight (~21.6 N)
- **Total mass**: 1.1 kg
- **Control**: 2 joints × 3 DOF = 6 control dimensions per robot

### Type B: Rolling Sphere (`rolling_sphere.urdf`)

A simple sphere robot designed for rolling and serving as connection points for Type A robots.

**Specifications:**
- **Diameter**: 0.334m (~1/3 of bar length)
- **Radius**: 0.167m
- **Mass**: 1.0kg
- **Inertia**: Calculated for solid sphere (I = 2/5 * m * r²)

**Key Features:**
- **Can be grasped by Type A endpoints** for connection-based structures
- **Movable**: Can apply external forces for repositioning
- **Control**: 3 force dimensions (fx, fy, fz) per robot

**Dimensional Relationship:**
- Sphere diameter = 1/3 × Bar total length
- 0.334m ≈ 1/3 × 1.0m ✓

## Usage in Training Environment

The environment (`HeightMaximizeEnv`) automatically loads and manages these robots:

```python
from src.environments import HeightMaximizeEnv

# Create environment with N=2
# This deploys 2 Type B and 4 Type A robots (total: 6 robots)
env = HeightMaximizeEnv(num_type_b_robots=2)

# Reset for new episode (robots are randomly positioned)
observation, info = env.reset()

# Take actions
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
```

## Direct PyBullet Usage

```python
import pybullet as p
import pybullet_data

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load Type A robot (bar with joint)
type_a_id = p.loadURDF("models/bar_with_joint.urdf", [0, 0, 1])

# Load Type B robot (sphere)
type_b_id = p.loadURDF("models/rolling_sphere.urdf", [2, 0, 0.5])

# Run simulation
for _ in range(1000):
    p.stepSimulation()
```

## Robot Deployment Pattern

For training parameter N:
- **Type B robots**: N instances
- **Type A robots**: 2N instances  
- **Total robots**: 3N instances

Example: N=2 → 2 Type B + 4 Type A = 6 total robots

```python
# The environment handles multi-robot deployment automatically
# Each robot is positioned randomly within a spawn radius
# Minimum separation ensures no initial collisions

env = HeightMaximizeEnv(
    num_type_b_robots=2,    # N parameter
    spawn_radius=3.0        # Random spawn area
)
```

## Robot Connections

Type A robots can dynamically connect to Type B robots:

- **Connection trigger**: When Type A endpoint comes within 0.2m of Type B center
- **Connection type**: Point-to-point constraint (PyBullet)
- **Disconnection**: Automatic when robots drift > 0.5m apart
- **Purpose**: Enables collaborative structure building for height maximization

## Notes

- All models use SI units (meters, kilograms)
- Inertia tensors are calculated based on geometry and mass distribution
- Color materials are defined for visualization (red for joints, blue for bars, green for spheres)
- Collision geometries match visual geometries for accurate physics simulation
