# Environment Setup Summary: AGA-8

## Overview

This document describes the implementation of the first robotics training environment with URDF-based robot models as specified in Linear issue AGA-8.

## Robot Types

### Type A: Bar Robot (`bar_with_joint.urdf`)
- **Structure**: Central base with 2 bar segments connected via spherical joints
- **Mass**: 1.1 kg total (base: 0.1 kg, each bar: 0.5 kg)
- **Dimensions**: 0.5m long bars, 0.015m radius cylinders
- **Joints**: 2 spherical joints allowing 3 DOF each
- **Motor Force**: 21.6 N (enough to lift 2x body weight)
- **Special Feature**: Bar endpoints can connect to Type B robots

### Type B: Sphere Robot (`rolling_sphere.urdf`)
- **Structure**: Simple rolling sphere
- **Mass**: 1.0 kg
- **Radius**: 0.167m
- **Special Feature**: Can be grasped by Type A robot endpoints

## Deployment Strategy

For each episode with parameter `N` (number of Type B robots):
- **Type B Robots**: N robots
- **Type A Robots**: 2N robots
- **Total Robots**: 3N robots

### Random Deployment
- Robots are randomly positioned within a configurable spawn radius (default: 3.0m)
- Minimum separation of 0.4m between robots to prevent initial collisions
- Random orientation around Z-axis for Type A robots
- All robots spawn at height 0.5m above ground plane

## Physics Implementation

### Joint Motors
- Each Type A robot has 2 spherical joints (ball-and-socket)
- Each spherical joint is controlled via 3 DOF (represented as quaternion targets)
- Motor force configured to lift twice the robot's body weight: `Force = 2 × mass × gravity`
- Type A joint force: ~21.6 N

### Connection System
Type A robot bar endpoints can dynamically connect to Type B robot spheres:

1. **Detection**: Every step, checks distance between Type A endpoints and Type B centers
2. **Connection Threshold**: 0.2m (configurable via `connection_distance`)
3. **Connection Type**: Point-to-point constraint (PyBullet `JOINT_POINT2POINT`)
4. **Disconnection**: Automatic removal if robots drift > 2.5x connection distance
5. **Implementation**: Tracks all active connections with constraint IDs

This allows Type A robots to:
- Attach to Type B robots for collaborative structures
- Build towers or formations by connecting multiple robots
- Dynamically disconnect and reconnect as needed

## Action Space

The action space is continuous and scales with the number of robots:

**Total Dimensions**: `num_type_a × 6 + num_type_b × 3`

- **Type A robots** (6 values per robot):
  - 2 spherical joints × 3 DOF each
  - Values in range [-1, 1] mapped to joint angle targets (-π to π)
  
- **Type B robots** (3 values per robot):
  - External forces in x, y, z directions
  - Values in range [-1, 1] scaled to force magnitude (20.0 N)

**Example**: N=2 → Action space: (30,)
- 4 Type A × 6 = 24 dimensions
- 2 Type B × 3 = 6 dimensions

## Observation Space

The observation space provides full state information for all robots:

**Total Dimensions**: `num_type_a × 19 + num_type_b × 13`

### Type A Robot Observation (19 values):
- Position (x, y, z): 3 values
- Orientation quaternion (qx, qy, qz, qw): 4 values
- Linear velocity (vx, vy, vz): 3 values
- Angular velocity (wx, wy, wz): 3 values
- Joint states (2 joints × 3 values): 6 values

### Type B Robot Observation (13 values):
- Position (x, y, z): 3 values
- Orientation quaternion (qx, qy, qz, qw): 4 values
- Linear velocity (vx, vy, vz): 3 values
- Angular velocity (wx, wy, wz): 3 values

**Example**: N=2 → Observation space: (102,)
- 4 Type A × 19 = 76 dimensions
- 2 Type B × 13 = 26 dimensions

## Reward Function

As specified in the Linear issue, the reward is based on the **highest z-level** of all robots:

### During Episode:
- Small step reward: `0.1 × current_max_height`
- Guides learning during the episode

### At Episode End:
- Large bonus: `10.0 × max_height_achieved`
- Emphasizes final configuration over intermediate states

### Total Reward:
```
reward = step_reward + (end_bonus if terminated/truncated else 0)
```

This encourages agents to build the highest possible structures using the available robots.

## Episode Termination

Episodes terminate when:
1. Any robot falls significantly below ground (z < -1.0m)
2. Any robot moves too far horizontally (distance > 20.0m from origin)
3. Maximum episode steps reached (default: 1000 steps)

## Environment Configuration

### Default Parameters
```python
env = HeightMaximizeEnv(
    render_mode=None,           # 'human' for visualization, None for headless
    num_type_b_robots=2,        # N parameter (Type A will be 2N)
    spawn_radius=3.0,           # Random spawn area radius
    max_episode_steps=1000      # Max steps per episode
)
```

### Customization Options
- `num_type_b_robots`: Controls total robot count (3N total robots)
- `spawn_radius`: Controls spatial distribution of initial deployment
- `max_episode_steps`: Training episode length
- `render_mode`: Enable/disable visualization

## Testing

Comprehensive test suite in `examples/test_height_env.py`:

✓ **Environment initialization** - Proper setup with URDF models  
✓ **Robot loading** - Type A and Type B robots loaded correctly  
✓ **Random deployment** - Proper spatial distribution  
✓ **Action/observation spaces** - Correct dimensions that scale with N  
✓ **Episode execution** - Full episode runs without errors  
✓ **Reward calculation** - Height-based rewards computed correctly  
✓ **Connection tracking** - Type A to Type B connections monitored  
✓ **Multiple episodes** - Consistent behavior across resets  
✓ **Different robot counts** - Scales properly with different N values  

### Run Tests
```bash
# From workspace root
PYTHONPATH=/workspace python3 examples/test_height_env.py
```

## Implementation Files

- `src/environments/height_maximize_env.py` - Main environment implementation
- `models/bar_with_joint.urdf` - Type A robot URDF
- `models/rolling_sphere.urdf` - Type B robot URDF
- `examples/test_height_env.py` - Comprehensive test suite

## Training Ready

The environment is fully configured and tested for RL training:

1. **Stable-Baselines3 Compatible** - Implements Gymnasium interface
2. **Scalable** - Works with different robot counts
3. **Physics-Based** - Realistic PyBullet simulation
4. **Reward Signal** - Clear objective (maximize height)
5. **Connection System** - Enables collaborative robot behaviors

### Next Steps
1. Configure training hyperparameters in `configs/training_config.yaml`
2. Run training with: `python train.py`
3. Monitor training progress with TensorBoard
4. Evaluate trained policies with: `python evaluate.py`

## Key Features Implemented

✅ **URDF-based robot models** (Type A bars, Type B spheres)  
✅ **Random deployment** (N Type B, 2N Type A robots per episode)  
✅ **Joint motors** with force to lift 2x body weight  
✅ **Dynamic connections** between Type A endpoints and Type B spheres  
✅ **Height-based reward** (highest z-level at episode end)  
✅ **Scalable architecture** (configurable robot counts)  
✅ **Comprehensive testing** (validated all major features)  

The environment is ready for reinforcement learning training!
