# New Training Environments Documentation

This document describes the four new training environments designed to train robots for complex navigation, manipulation, and locomotion tasks.

## Overview

The new environments build upon the foundation of the existing Height Maximization (Session 1) and Crush Resistance (Session 2) environments. They use the same robot types (Type A bar robots and Type B sphere robots) but introduce new challenges that collectively teach robots to:

1. **Navigate complex spaces** with obstacles
2. **Manipulate objects** in the environment  
3. **Traverse varied terrain** with different surface types
4. **Navigate narrow passages** requiring precision control

## Training Session 3: Obstacle Navigation Environment

**File:** `src/environments/obstacle_navigation_env.py`  
**Class:** `ObstacleNavigationEnv`  
**Test Script:** `examples/test_navigation_env.py`

### Purpose
Train robots to navigate through an arena filled with random obstacles to reach target positions.

### Features
- **Random obstacle placement**: Walls, boxes, and cylindrical pillars
- **Dynamic target repositioning**: Target moves to new location after being reached
- **Configurable difficulty**: Number and type of obstacles
- **Arena-based**: Square arena with boundaries

### Key Parameters
- `num_obstacles` (default: 8): Number of obstacles in the arena
- `obstacle_types`: Types of obstacles ['wall', 'box', 'cylinder']
- `arena_size` (default: 20.0m): Size of the square arena
- `target_radius` (default: 1.0m): Radius of target zone

### Reward Structure
- **Progress toward target**: +2.0 per meter of progress
- **Reaching target**: +100.0 (target then repositions)
- **Distance from target**: -0.01 per meter (encourages engagement)
- **Collision with obstacles**: -1.0 per collision
- **Step reward**: +0.1

### Termination Conditions
- Robot falls below ground (-1.0m)
- Robot exits arena boundaries
- Maximum episode steps reached

### Training Goals
This environment teaches:
- Spatial awareness and pathfinding
- Collision avoidance strategies
- Coordinated movement between robots
- Long-term planning to reach distant targets

---

## Training Session 4: Object Manipulation Environment

**File:** `src/environments/object_manipulation_env.py`  
**Class:** `ObjectManipulationEnv`  
**Test Script:** `examples/test_manipulation_env.py`

### Purpose
Train robots to push and pull objects to designated target locations.

### Features
- **Multiple manipulable objects**: Boxes and cylinders with varying mass
- **Target zones**: Visual indicators showing where objects should be placed
- **Physics-based manipulation**: Objects have realistic mass, friction, and inertia
- **Success condition**: All objects must be placed in their target zones

### Key Parameters
- `num_objects` (default: 3): Number of objects to manipulate
- `arena_size` (default: 15.0m): Size of the arena
- `target_radius` (default: 0.8m): Size of target zones

### Object Properties
- **Mass**: 2.0-8.0 kg (heavier than robots, requiring coordinated effort)
- **Friction**: Varies per object (0.5-1.5 lateral friction)
- **Types**: Boxes (0.5-1.0m cubed) and cylinders (0.3-0.6m radius)

### Reward Structure
- **Moving object toward target**: +5.0 per meter of progress
- **Object in target zone**: +50.0 per object
- **All objects in targets**: +200.0 (episode success)
- **Distance from objects**: -0.02 per meter (encourages engagement)
- **Being near objects**: +0.1 when within 2m
- **Distance to targets**: -0.02 per meter per object
- **Step reward**: +0.1

### Termination Conditions
- All objects successfully placed in target zones (success!)
- Robot falls below ground
- Robot exits arena boundaries
- Maximum episode steps reached

### Training Goals
This environment teaches:
- Force application and coordination
- Object physics understanding
- Team coordination for pushing heavy objects
- Strategic planning (which object to move first)
- Precise positioning for target placement

---

## Training Session 5: Terrain Traversal Environment

**File:** `src/environments/terrain_traversal_env.py`  
**Class:** `TerrainTraversalEnv`  
**Test Script:** `examples/test_terrain_env.py`

### Purpose
Train robots to traverse challenging terrain with slopes, stairs, gaps, and uneven surfaces.

### Features
- **Procedurally generated terrain**: Different terrain sections each episode
- **Multiple terrain types**:
  - Slopes (up and down, 10-45° based on difficulty)
  - Stairs (ascending and descending, 3-8 steps)
  - Rough terrain (random height variations)
  - Gaps (0.3-1.0m based on difficulty)
  - Flat platforms (rest areas)
- **Progressive difficulty**: Adjustable terrain difficulty parameter
- **Linear progression**: Travel distance measured in X direction

### Key Parameters
- `target_distance` (default: 20.0m): Distance to travel forward
- `terrain_difficulty` (default: 0.5): Difficulty level 0.0-1.0
  - 0.0: Gentle slopes, small steps, narrow gaps
  - 1.0: Steep slopes (45°), tall steps (0.3m), wide gaps (1.0m)
- `num_terrain_sections` (default: 5): Number of terrain challenges

### Terrain Section Types
1. **Slope Up/Down**: Angled surfaces (angle increases with difficulty)
2. **Stairs Up/Down**: Step sequences (height increases with difficulty)
3. **Rough Terrain**: Uneven blocks (variation increases with difficulty)
4. **Gaps**: Empty spaces to cross (width increases with difficulty)
5. **Platforms**: Flat rest areas between challenges

### Reward Structure
- **Forward progress**: +10.0 per meter moved in X direction
- **Maintaining height**: +0.1 per step if above 0.3m
- **Reaching target distance**: +500.0
- **Continuous distance reward**: +0.1 × distance covered
- **Being stuck**: -0.5 per step after 50 steps without progress
- **Step reward**: +0.1

### Termination Conditions
- Reaching target distance (success!)
- Robot falls below -2.0m
- Robot moves too far sideways (>10m from path)
- Stuck for 200 steps
- Maximum episode steps reached

### Training Goals
This environment teaches:
- Adaptive locomotion strategies
- Balance and stability maintenance
- Climbing and descending skills
- Gap crossing techniques
- Recovery from unstable situations

---

## Training Session 6: Narrow Passage Environment

**File:** `src/environments/narrow_passage_env.py`  
**Class:** `NarrowPassageEnv`  
**Test Script:** `examples/test_passage_env.py`

### Purpose
Train robots to navigate through narrow passages and tight spaces requiring precise control.

### Features
- **Maze-like structure**: Series of connected passages
- **Multiple passage types**:
  - Straight corridors
  - 90° turns (left and right)
  - S-curves
  - Zigzag patterns
- **Checkpoint system**: Visual markers indicating progress
- **Collision tracking**: Monitors wall collisions
- **Configurable narrowness**: Passage width adjusts with difficulty

### Key Parameters
- `num_passages` (default: 4): Number of passage sections
- `passage_width` (default: 2.0m): Base width of passages
- `passage_difficulty` (default: 0.5): Difficulty level 0.0-1.0
  - 0.0: Full width (100% of base width)
  - 1.0: Very narrow (40% of base width)
  - Formula: `effective_width = base_width × (1.0 - 0.6 × difficulty)`

### Passage Types
1. **Straight**: Simple corridor (4-6m long)
2. **Turn Left/Right**: 90° L-shaped passage (3m per side)
3. **S-Curve**: Sinusoidal path (5m long, 2m amplitude)
4. **Zigzag**: Alternating angled sections (4m long, 1.5m offset)

### Checkpoint System
- **Checkpoint markers**: Semi-transparent cylinders marking progress points
- **One checkpoint per passage**: Must pass through each
- **Final checkpoint**: Green marker at the end (goal)
- **Visual feedback**: Blue for regular checkpoints, green for final

### Reward Structure
- **Passing checkpoint**: +100.0 per checkpoint
- **Progress toward next checkpoint**: -0.05 × distance (encourages movement)
- **Wall collision**: -2.0 per collision (significant penalty)
- **Completing all checkpoints**: +500.0 (success)
- **Step reward**: +0.1

### Termination Conditions
- All checkpoints passed (success!)
- Robot falls below ground
- Robot moves very far out of bounds (>100m)
- Maximum episode steps reached

### Training Goals
This environment teaches:
- Precise motor control
- Spatial awareness in confined spaces
- Collision avoidance with narrow margins
- Sequential goal achievement
- Possibly reconfiguration (robots may need to reshape)

---

## Integration with Existing Framework

### Environment Registration

All new environments are registered in `src/environments/__init__.py`:

```python
from .obstacle_navigation_env import ObstacleNavigationEnv
from .object_manipulation_env import ObjectManipulationEnv
from .terrain_traversal_env import TerrainTraversalEnv
from .narrow_passage_env import NarrowPassageEnv
```

### Usage Example

```python
from src.environments import ObstacleNavigationEnv

# Create environment
env = ObstacleNavigationEnv(
    num_type_b_robots=2,      # N Type B robots (2N Type A)
    num_obstacles=8,           # 8 random obstacles
    arena_size=20.0,           # 20m × 20m arena
    render_mode='human'        # Visual rendering
)

# Train or evaluate
obs, info = env.reset()
for step in range(1000):
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Testing

Each environment has a dedicated test script:

```bash
# Test individual environments
python examples/test_navigation_env.py
python examples/test_manipulation_env.py
python examples/test_terrain_env.py
python examples/test_passage_env.py
```

---

## Training Curriculum Recommendation

For best results, train robots in progressive sessions:

### Phase 1: Basic Skills (Sessions 1-2)
1. **Height Maximization**: Learn coordination and connections
2. **Crush Resistance**: Learn structural stability

### Phase 2: Navigation (Session 3)
3. **Obstacle Navigation**: Learn pathfinding and collision avoidance

### Phase 3: Advanced Navigation (Sessions 5-6)
4. **Terrain Traversal**: Learn adaptive locomotion
5. **Narrow Passages**: Learn precision control

### Phase 4: Manipulation (Session 4)
6. **Object Manipulation**: Learn force application and coordination

### Phase 5: Integration
7. **Multi-task training**: Combine environments in curriculum
8. **Transfer learning**: Test on new environment combinations

---

## Design Principles

All environments follow these principles:

### 1. **Consistent Robot Types**
- Same Type A (bar with joints) and Type B (sphere) robots
- Same physics parameters across all environments
- Connection mechanics preserved

### 2. **Progressive Difficulty**
- Each environment has configurable difficulty
- Supports curriculum learning approaches
- Can start easy and gradually increase challenge

### 3. **Clear Success Criteria**
- Explicit success conditions (all checkpoints, objects in targets, etc.)
- Episode ends on success or failure
- Info dict includes progress metrics

### 4. **Rich Observations**
- Robot states (position, orientation, velocity)
- Environment-specific information (target location, object positions)
- Relative measurements (vectors to targets, distances)

### 5. **Shaped Rewards**
- Dense rewards for progress (not just sparse terminal rewards)
- Penalties for undesirable behaviors (collisions, falls)
- Large bonuses for achieving goals

### 6. **Gymnasium Compatibility**
- Standard Gymnasium API
- Compatible with Stable-Baselines3
- Works with existing training infrastructure

---

## Advanced Configuration

### Combining Environments

Environments can be combined for multi-task learning:

```python
# Curriculum learning example
envs = [
    HeightMaximizeEnv(num_type_b_robots=2),
    ObstacleNavigationEnv(num_type_b_robots=2, num_obstacles=5),
    TerrainTraversalEnv(num_type_b_robots=2, terrain_difficulty=0.3),
    ObjectManipulationEnv(num_type_b_robots=2, num_objects=2),
]

# Train on each for N episodes, then move to next
```

### Difficulty Progression

Start with easier configurations and gradually increase:

```python
# Week 1: Easy
env = ObstacleNavigationEnv(num_obstacles=4, arena_size=25.0)

# Week 2: Medium
env = ObstacleNavigationEnv(num_obstacles=8, arena_size=20.0)

# Week 3: Hard
env = ObstacleNavigationEnv(num_obstacles=12, arena_size=15.0)
```

---

## Performance Considerations

### Computational Cost
- **Obstacle Navigation**: Low (simple collision detection)
- **Object Manipulation**: Medium (physics of movable objects)
- **Terrain Traversal**: Medium-High (many terrain pieces)
- **Narrow Passages**: Medium (many wall collision checks)

### Rendering
- All environments support `render_mode='human'` for visualization
- Use `render_mode=None` for faster training
- RGB array mode available for observation-based learning

### Episode Length
Recommended `max_episode_steps`:
- **Obstacle Navigation**: 2000 steps (allows multiple target reaches)
- **Object Manipulation**: 2000 steps (time to push heavy objects)
- **Terrain Traversal**: 2000 steps (20m at slow pace)
- **Narrow Passages**: 2000 steps (careful navigation takes time)

---

## Future Enhancements

Possible extensions for these environments:

### 1. **Dynamic Obstacles**
- Moving obstacles in navigation environment
- Objects that roll or slide in manipulation environment

### 2. **Multi-Robot Coordination**
- Explicit team rewards
- Communication channels between robots
- Role specialization

### 3. **Partial Observability**
- Limited sensor range
- Occlusions and line-of-sight
- Exploration rewards

### 4. **Procedural Generation**
- More varied terrain generation
- Random maze generation for passages
- Procedural object shapes

### 5. **Real-World Transfer**
- Domain randomization for sim-to-real
- Noise injection for robustness
- Camera-based observations

---

## Troubleshooting

### Common Issues

**Issue**: Robots fall through terrain  
**Solution**: Check that robots spawn at safe height (1.0-1.5m above terrain)

**Issue**: Collisions not detected  
**Solution**: Verify PyBullet collision shapes are properly created with non-zero mass for dynamic objects

**Issue**: Rewards are too sparse  
**Solution**: Adjust reward shaping parameters to provide more frequent feedback

**Issue**: Episodes terminate too quickly  
**Solution**: Increase `max_episode_steps` or adjust termination conditions

**Issue**: Training is too slow  
**Solution**: Reduce `frame_skip`, simplify environment (fewer obstacles/terrain pieces), or disable rendering

---

## Conclusion

These four new environments significantly expand the training capabilities of the RL robotics framework. Together with the existing environments, they provide a comprehensive curriculum for training robots to:

- **Move anywhere**: Navigation, terrain traversal, narrow passages
- **Navigate any obstacle**: Walls, gaps, slopes, stairs
- **Do any task**: Object manipulation, height building, crush resistance

The environments are production-ready, well-tested, and fully integrated with the existing training infrastructure. They can be used individually or combined for multi-task and curriculum learning approaches.

**Next Steps**:
1. Run test scripts to verify environments work in your setup
2. Train agents on individual environments
3. Implement curriculum learning across multiple environments
4. Evaluate transfer learning capabilities
5. Scale training to AWS Spot Fleet for faster results

Happy training! 🤖🚀
