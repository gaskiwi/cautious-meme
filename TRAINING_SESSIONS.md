# Training Sessions Overview

This document provides a quick reference for all available training environments.

## Environment Summary Table

| Session | Environment | Goal | Difficulty | Key Skills |
|---------|-------------|------|------------|------------|
| 1 | Height Maximization | Build vertical structures | Medium | Coordination, connections |
| 2 | Crush Resistance | Resist hydraulic press | Hard | Structural stability |
| 3 | **Obstacle Navigation** ✨ | Navigate to targets through obstacles | Medium | Pathfinding, avoidance |
| 4 | **Object Manipulation** ✨ | Push objects to target zones | Hard | Force application, teamwork |
| 5 | **Terrain Traversal** ✨ | Cross varied terrain | Hard | Adaptive locomotion |
| 6 | **Narrow Passages** ✨ | Navigate tight corridors | Medium | Precision control |

✨ = New environments

---

## Quick Start Guide

### Session 1: Height Maximization
```python
from src.environments import HeightMaximizeEnv

env = HeightMaximizeEnv(
    num_type_b_robots=2,  # N Type B robots (creates 2N Type A)
    spawn_radius=3.0,
    render_mode=None
)
```
**Goal**: Get at least one robot as high as possible  
**Test**: `python examples/test_height_env.py`

---

### Session 2: Crush Resistance
```python
from src.environments import CrushResistanceEnv

env = CrushResistanceEnv(
    num_type_b_robots=2,
    reference_height=5.0,        # Press starts at 5m
    press_descent_speed=0.05,    # 0.05 m/s
    press_force_increment=50.0   # Force increase per step
)
```
**Goal**: Resist descending press as long as possible  
**Test**: `python examples/test_crush_env.py`

---

### Session 3: Obstacle Navigation ✨
```python
from src.environments import ObstacleNavigationEnv

env = ObstacleNavigationEnv(
    num_type_b_robots=2,
    num_obstacles=8,        # Number of obstacles
    arena_size=20.0,        # 20m × 20m arena
    target_radius=1.0       # Target zone size
)
```
**Goal**: Navigate through obstacles to reach targets  
**Test**: `python examples/test_navigation_env.py`

**Difficulty Settings**:
- Easy: `num_obstacles=4, arena_size=25.0`
- Medium: `num_obstacles=8, arena_size=20.0`
- Hard: `num_obstacles=12, arena_size=15.0`

---

### Session 4: Object Manipulation ✨
```python
from src.environments import ObjectManipulationEnv

env = ObjectManipulationEnv(
    num_type_b_robots=2,
    num_objects=3,          # Objects to manipulate
    arena_size=15.0,        # Arena size
    target_radius=0.8       # Target zone size
)
```
**Goal**: Push/pull all objects into their target zones  
**Test**: `python examples/test_manipulation_env.py`

**Difficulty Settings**:
- Easy: `num_objects=1, arena_size=15.0`
- Medium: `num_objects=2, arena_size=12.0`
- Hard: `num_objects=4, arena_size=10.0`

---

### Session 5: Terrain Traversal ✨
```python
from src.environments import TerrainTraversalEnv

env = TerrainTraversalEnv(
    num_type_b_robots=2,
    target_distance=20.0,         # Distance to travel (m)
    terrain_difficulty=0.5,       # 0.0 (easy) to 1.0 (hard)
    num_terrain_sections=5        # Number of challenges
)
```
**Goal**: Traverse challenging terrain to reach target distance  
**Test**: `python examples/test_terrain_env.py`

**Difficulty Settings**:
- Easy: `terrain_difficulty=0.2, target_distance=15.0`
- Medium: `terrain_difficulty=0.5, target_distance=20.0`
- Hard: `terrain_difficulty=0.8, target_distance=25.0`

**Terrain Types**:
- Slopes (10-45° based on difficulty)
- Stairs (3-8 steps, 0.15-0.3m height)
- Rough terrain (random bumps)
- Gaps (0.3-1.0m)
- Flat platforms

---

### Session 6: Narrow Passages ✨
```python
from src.environments import NarrowPassageEnv

env = NarrowPassageEnv(
    num_type_b_robots=2,
    num_passages=4,             # Number of passage sections
    passage_width=2.0,          # Base width (m)
    passage_difficulty=0.5      # 0.0 (wide) to 1.0 (narrow)
)
```
**Goal**: Navigate through all checkpoints in narrow passages  
**Test**: `python examples/test_passage_env.py`

**Difficulty Settings**:
- Easy: `passage_difficulty=0.2` → 1.88m wide
- Medium: `passage_difficulty=0.5` → 1.40m wide
- Hard: `passage_difficulty=0.8` → 1.04m wide

**Passage Types**:
- Straight corridors
- 90° turns
- S-curves
- Zigzag patterns

---

## Training Configuration

### Basic Training Loop
```python
from src.environments import ObstacleNavigationEnv
from stable_baselines3 import PPO

# Create environment
env = ObstacleNavigationEnv(num_type_b_robots=2)

# Create agent
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=1_000_000)

# Save
model.save("obstacle_navigation_model")
```

### Curriculum Learning
```python
# Progressive training across environments
envs = [
    ("height", HeightMaximizeEnv(num_type_b_robots=2)),
    ("navigation", ObstacleNavigationEnv(num_type_b_robots=2, num_obstacles=5)),
    ("terrain", TerrainTraversalEnv(num_type_b_robots=2, terrain_difficulty=0.3)),
    ("manipulation", ObjectManipulationEnv(num_type_b_robots=2, num_objects=2)),
]

for name, env in envs:
    print(f"Training on {name}...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500_000)
    model.save(f"model_{name}")
    env.close()
```

### Difficulty Progression
```python
# Start easy, increase difficulty over time
difficulties = [0.2, 0.4, 0.6, 0.8]

for diff in difficulties:
    env = TerrainTraversalEnv(
        num_type_b_robots=2,
        terrain_difficulty=diff
    )
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=250_000)
    model.save(f"terrain_model_diff_{diff}")
    env.close()
```

---

## Robot Types

All environments use the same robot types:

### Type A: Bar Robot with Joints
- **URDF**: `models/bar_with_joint.urdf`
- **Count**: 2N robots (twice the number of Type B)
- **Features**:
  - Central base (0.1 kg)
  - Two bars (0.5 kg each)
  - Spherical joints (3 DOF each)
  - Endpoints can connect to Type B robots
- **Total mass**: 1.1 kg
- **Joint force**: ~21.6 N (can lift 2× body weight)

### Type B: Sphere Robot
- **URDF**: `models/rolling_sphere.urdf`
- **Count**: N robots
- **Features**:
  - Spherical body
  - Can roll and move via external forces
  - Connection points for Type A endpoints
- **Mass**: 1.0 kg
- **Force magnitude**: 20.0 N

### Connections
- Type A bar endpoints automatically connect to nearby Type B spheres
- Connection distance: 0.2m
- Connections break if robots move too far apart (>0.5m)
- Enables collaborative structure building and movement

---

## Common Parameters

All environments support these common parameters:

```python
env = SomeEnvironment(
    num_type_b_robots=2,        # Number of Type B robots (Type A = 2×)
    spawn_radius=2.0,           # Spawn area radius (m)
    render_mode=None,           # None, 'human', or 'rgb_array'
    timestep=1./240.,           # Physics timestep
    frame_skip=4,               # Simulation steps per env step
    max_episode_steps=1000      # Maximum steps per episode
)
```

---

## Observation Spaces

### Type A Robot (19 values per robot)
- Position (3): x, y, z
- Orientation (4): quaternion
- Linear velocity (3): vx, vy, vz
- Angular velocity (3): ωx, ωy, ωz
- Joint states (6): 2 joints × 3 DOF

### Type B Robot (13 values per robot)
- Position (3): x, y, z
- Orientation (4): quaternion
- Linear velocity (3): vx, vy, vz
- Angular velocity (3): ωx, ωy, ωz

### Environment-Specific
Each environment adds relevant information:
- **Navigation**: Target position, vectors to target
- **Manipulation**: Object positions, target positions
- **Terrain**: Forward distance, terrain encoding
- **Passages**: Next checkpoint, checkpoints passed

---

## Action Spaces

### Type A Robots
- **6 values per robot**: 2 joints × 3 DOF
- Range: [-1, 1] mapped to [-π, π] joint angles
- Control: Position control with force limits

### Type B Robots
- **3 values per robot**: force in x, y, z
- Range: [-1, 1] mapped to ±20N force
- Control: Direct force application

### Total Action Dimension
- For N Type B robots: `6 × 2N + 3 × N = 15N`
- Example with N=2: 30 actions (24 for Type A, 6 for Type B)

---

## Reward Structures

### Height Maximization
- Small per-step reward based on current max height
- Large bonus at episode end based on max height achieved

### Crush Resistance
- Small reward during positioning phase
- Large reward for survival time after press activates
- Bonus for maintaining height while under pressure

### Obstacle Navigation ✨
- Progress toward target: +2.0 per meter
- Reaching target: +100.0
- Collision penalty: -1.0 per collision
- Distance penalty: -0.01 × distance

### Object Manipulation ✨
- Progress moving object: +5.0 per meter
- Object in target: +50.0
- All objects in targets: +200.0
- Encouragement to engage: bonus for being near objects

### Terrain Traversal ✨
- Forward progress: +10.0 per meter
- Reaching target: +500.0
- Stability bonus: +0.1 per step above ground
- Stuck penalty: -0.5 after extended no-progress

### Narrow Passages ✨
- Passing checkpoint: +100.0
- Completing all: +500.0
- Collision penalty: -2.0 per wall collision
- Progress encouragement: -0.05 × distance to next

---

## Performance Tuning

### For Faster Training
```python
env = Environment(
    render_mode=None,           # Disable rendering
    frame_skip=8,               # More simulation per step
    max_episode_steps=500,      # Shorter episodes
    num_type_b_robots=1         # Fewer robots
)
```

### For Better Learning
```python
env = Environment(
    render_mode=None,
    frame_skip=4,               # Standard simulation
    max_episode_steps=2000,     # Full episodes
    num_type_b_robots=2         # Standard robot count
)
```

### For Visualization
```python
env = Environment(
    render_mode='human',        # Visual rendering
    frame_skip=4,
    max_episode_steps=1000
)
```

---

## AWS Deployment

All environments work with the existing AWS infrastructure:

```bash
# Deploy infrastructure
cd aws
./deploy.sh --vpc-id vpc-xxx --subnet-ids subnet-xxx --key-name my-key

# Push Docker image
./push-image.sh

# Training runs automatically on Spot Fleet
```

Cost savings: ~70% using Spot Fleet vs on-demand instances.

---

## Training Session 3: Object Displacement

**Status:** ✅ Implemented

**Goal:** Displace a randomly shaped object as far as possible in a random cardinal direction.

### Overview
This is the third training session in the series. The primary objective is to teach agents to push/move objects of varying shapes in specific directions. Each episode presents a unique challenge with randomized object shapes, randomized placement of both objects and robots, and a randomly chosen target direction (North, East, South, or West).

### Key Details
- **Environment:** `DisplacementEnv` (`src/environments/displacement_env.py`)
- **Objective:** Maximize displacement of an object in a randomly chosen cardinal direction
- **Multi-Robot:** Same robot configuration as previous sessions (Type A bars + Type B spheres)
- **Physics:** Consistent physics with Session 1 & 2 for transfer learning
- **Scoring:** Based on object displacement in target direction
- **Robot Types:**
  - Type A: Bar robots with joints (bar_with_joint.urdf) - 2N robots
  - Type B: Sphere robots (rolling_sphere.urdf) - N robots

### Randomization Features

#### Object Shapes (8 varieties)
The environment randomly selects from diverse object shapes each episode:
1. **Sphere** - Simple rolling object (2.0 kg, radius 0.3m)
2. **Cube** - Standard box (2.5 kg, 0.5m sides)
3. **Cylinder** - Tall cylinder (2.2 kg, 0.2m radius, 0.6m height)
4. **Capsule** - Pill-shaped object (1.8 kg, 0.15m radius, 0.5m height)
5. **Coffee Mug** - Complex shape with handle (1.5 kg, approx 0.4m tall)
6. **Torus** - Ring/donut shape (2.0 kg, 0.3m major radius)
7. **Rectangular Prism** - Elongated box (2.3 kg, varying dimensions)
8. **Tall Cylinder** - Unstable tall object (2.0 kg, 0.15m radius, 0.8m height)

Each shape has unique:
- Mass distribution
- Friction coefficients
- Center of mass
- Stability characteristics
- Push resistance

#### Cardinal Directions (4 options)
One direction is randomly chosen per episode:
- **North:** +Y axis (vector: [0, 1])
- **South:** -Y axis (vector: [0, -1])
- **East:** +X axis (vector: [1, 0])
- **West:** -X axis (vector: [-1, 0])

#### Placement Randomization
- **Object:** Spawns 2-15 meters from origin at random angle
- **Robots:** Spawn in circle around object (1-3m radius from object)
- **Orientation:** Random rotation for all entities
- **Minimum Separation:** 0.4m between robots, 0.8m from object

### Reward Structure
The reward function for Training Session 3 consists of:

1. **Displacement Reward:** Primary reward based on current displacement
   - Formula: `displacement_in_target_direction × 1.0`
   - Can be positive (moving forward) or negative (moving backward)

2. **Velocity Bonus:** Encourages maintaining momentum
   - Formula: `object_velocity_in_target_direction × 0.1`
   - Rewards objects moving with speed in target direction

3. **Final Displacement Bonus:** Large bonus at episode end
   - Formula: `max_displacement_achieved × 10.0`
   - Rewards total distance achieved in target direction

### Action Space
Same as previous sessions:
- Type A robots: 6 values per robot (2 spherical joints × 3 DOF)
- Type B robots: 3 values per robot (force x, y, z)

Total action space dimension: `(num_type_a × 6) + (num_type_b × 3)`

With default N=2: `(4 × 6) + (2 × 3) = 30` continuous values ∈ [-1, 1]

### Observation Space
Enhanced from previous sessions to include object and direction info:
- Type A robots: 19 values per robot (pos, orn, vel, ang_vel, joint_states)
- Type B robots: 13 values per robot (pos, orn, vel, ang_vel)
- **Object state:** 13 values (pos, orn, vel, ang_vel)
- **Target direction:** 2 values (unit vector in XY plane)

Total observation space dimension: `(num_type_a × 19) + (num_type_b × 13) + 13 + 2`

With default N=2: `(4 × 19) + (2 × 13) + 13 + 2 = 117` continuous values

### Displacement Calculation
The displacement is calculated as:
1. Get object's current position: `(x, y, z)`
2. Get object's initial position: `(x₀, y₀, z₀)`
3. Calculate displacement vector: `Δ = (x - x₀, y - y₀)`
4. Project onto target direction: `displacement = Δ · direction_unit_vector`

This allows:
- Positive displacement when moving in target direction
- Negative displacement when moving opposite to target
- Zero displacement when moving perpendicular

### Termination Conditions
An episode terminates if:
- Object falls below ground (z < -0.5m)
- Object moves beyond boundary (distance > 50m from origin)
- Any robot falls below ground (z < -1.0m)
- Any robot moves beyond boundary (distance > 50m from origin)
- Maximum episode steps reached (default: 1000)

### Configuration Parameters

Key configurable parameters in environment initialization:
```python
DisplacementEnv(
    num_type_b_robots=2,           # N (Type A will be 2N)
    spawn_radius=3.0,              # Robot spawn radius around object
    max_object_distance=15.0,      # Max distance from origin for object
    max_episode_steps=1000         # Episode length
)
```

Configuration file: `configs/session3_config.yaml`

### Usage

#### Testing the Environment
```bash
# Simple test without GUI (quick verification)
python3 examples/test_displacement_simple.py

# Full test with visualization (if GUI available)
python3 examples/test_displacement_env.py
```

The simple test runs 5 automated tests:
1. Basic environment functionality
2. All object shapes
3. All cardinal directions
4. Displacement calculation accuracy
5. Randomized placement verification

#### Training
```bash
# Train with Session 3 configuration
python3 train.py --config configs/session3_config.yaml
```

#### Evaluation
```bash
python3 evaluate.py models/session3_best_model --render
```

### Expected Learning Outcomes
By the end of Training Session 3, the agent should learn to:
- **Perception:** Recognize different object shapes and their properties
- **Strategy:** Adapt pushing strategy based on object shape
  - Roll spheres vs. push cubes
  - Handle unstable vs. stable objects
  - Adjust force based on object mass
- **Direction Following:** Push objects in specified cardinal direction
- **Collaboration:** Coordinate multiple robots to push larger/heavier objects
- **Force Application:** Apply appropriate force magnitude and direction
- **Positioning:** Position robots optimally relative to object and target direction
- **Persistence:** Continue pushing over long distances

### Metrics to Track
- Maximum displacement achieved (primary metric)
- Final object position
- Displacement per object shape (shape-specific performance)
- Displacement per direction (directional bias detection)
- Average velocity in target direction
- Number of robot-robot connections used
- Episode efficiency (displacement per step)
- Object stability (did it tip over or stay upright)

### Relationship to Previous Sessions
This session builds on previous sessions by:
- Using identical physics and robot types (transfer learning ready)
- Requiring force application skills (from both sessions)
- Adding object interaction as new challenge
- Testing adaptability to randomization
- Requiring strategic positioning (from Session 2)
- Using collaborative pushing (connection system from Sessions 1 & 2)

Models trained in Sessions 1 & 2 can be fine-tuned for Session 3, as skills like:
- Force application (Session 1)
- Strategic positioning (Session 2)
- Robot-robot collaboration (Sessions 1 & 2)

are all relevant to object displacement.

### Technical Implementation Notes

#### Object Creation
- Objects created using PyBullet geometric primitives
- Custom compound shapes for complex objects (coffee mug, torus)
- Realistic friction properties for natural interaction
- Visual colors distinguish different shapes

#### Physics Properties
- Lateral friction: 0.8 (realistic sliding)
- Spinning friction: 0.1
- Rolling friction: 0.05
- Object masses: 1.5-2.5 kg (pushable but challenging)

#### Randomization Implementation
- Numpy random seed support for reproducibility
- Shape selection: uniform random from 8 shapes
- Direction selection: uniform random from 4 directions
- Position generation: polar coordinates with collision checking
- Minimum separation enforced to prevent spawn overlaps

#### Connection System
- Same as Sessions 1 & 2: Type A bar endpoints can connect to Type B spheres
- Enables collaborative pushing of heavier objects
- Connection distance: 0.2m proximity threshold
- Dynamic connection/disconnection during episode

---

## How to Add New Training Sessions

- **NEW_ENVIRONMENTS.md**: Detailed technical documentation
- **ENVIRONMENT_SUMMARY.md**: Solution summary for Linear issue
- **TRAINING_SESSIONS.md**: This quick reference guide
- **ARCHITECTURE.md**: Overall system architecture
- **README.md**: Project setup and usage

---

## Troubleshooting

### Environment won't start
- Check PyBullet is installed: `pip install pybullet`
- Verify URDF files exist in `models/` directory

### Training is slow
- Disable rendering: `render_mode=None`
- Increase frame skip: `frame_skip=8`
- Use fewer robots: `num_type_b_robots=1`
- Consider AWS deployment for GPU acceleration

### Robots fall through terrain
- Increase spawn height in environment code
- Check collision shapes are properly defined

### Rewards are too sparse
- Adjust reward shaping parameters in environment
- Consider curriculum learning (start easier)

### Can't see what's happening
- Enable rendering: `render_mode='human'`
- Add print statements in step() method
- Check info dict for metrics

---

## Next Steps

1. **Test environments**: Run test scripts to verify setup
2. **Train baseline**: Train on each environment individually
3. **Curriculum learning**: Combine environments for progressive training
4. **Hyperparameter tuning**: Optimize learning rates, batch sizes
5. **Scale up**: Deploy to AWS for faster training
6. **Evaluate**: Test transfer learning between environments
7. **Deploy**: Use trained models for real tasks

---

## Support

For issues or questions:
- Check test scripts in `examples/`
- Review documentation in `docs/`
- Examine existing environment implementations
- Refer to PyBullet and Gymnasium documentation

Happy training! 🤖🚀
