# Training Sessions Documentation

This document outlines the different training sessions for the RL robotics project. Each session focuses on a specific goal and builds upon previous sessions. The sessions are designed to be run in succession to progressively train more capable agents.

---

## Training Session 1: Height Maximization

**Status:** ✅ Implemented

**Goal:** Get one of the robots as high (vertically) as possible.

### Overview
This is the first training session in a series of progressive training sessions. The primary objective is to teach the agent(s) to maximize vertical height. This foundational session establishes basic control and understanding of forces/motion in the vertical direction.

### Key Details
- **Environment:** `HeightMaximizeEnv` (`src/environments/height_maximize_env.py`)
- **Objective:** Maximize the Z-coordinate (height) of any robot
- **Multi-Robot:** Supports multiple robots (default: 3)
- **Scoring:** The highest robot's height determines the reward for the entire team
- **Robot Type:** Simple box robots with 6 DOF control (forces and torques in 3D)
  - Note: When using the custom multi-robot network, Type A (Bar) robots have 3 DOF (1 spherical joint) and Type B (Sphere) robots have 2 DOF (rolling control)

### Reward Structure
The reward function for Training Session 1 consists of:

1. **Height Reward:** Primary reward based on current maximum height (normalized)
   - Formula: `current_max_height / initial_height`
   
2. **Height Improvement Bonus:** Large bonus for achieving new maximum heights
   - Formula: `(new_max_height - previous_max_height) × 10.0`
   
3. **Energy Penalty:** Small penalty to encourage efficient solutions
   - Formula: `-0.01` per step

### Action Space
Each robot has 6 continuous control values:
- 3 force components: [Fx, Fy, Fz] ∈ [-1, 1]
- 3 torque components: [Tx, Ty, Tz] ∈ [-1, 1]

Total action space dimension: `num_robots × 6`

### Observation Space
Each robot provides 13 observation values:
- Position: [x, y, z]
- Linear velocity: [vx, vy, vz]
- Orientation (quaternion): [qx, qy, qz, qw]
- Angular velocity: [wx, wy, wz]

Total observation space dimension: `num_robots × 13`

### Termination Conditions
An episode terminates if:
- Any robot falls below ground level (z < 0)
- Any robot moves too far horizontally (distance > 10.0 units)
- Maximum episode steps reached (default: 1000)

### Training Configuration
Located in: `configs/training_config.yaml`

Key parameters:
```yaml
environment:
  name: "height_maximize_env"
  num_robots: 3
  max_episode_steps: 1000

agent:
  algorithm: "PPO"
  learning_rate: 0.0003
  total_timesteps: 1000000
```

### Usage

#### Testing the Environment
```bash
python3 examples/test_height_env.py
```

#### Training
```bash
python3 train.py
```

#### Training with Custom Config
```bash
python3 train.py --config configs/training_config.yaml
```

#### Evaluation
```bash
python3 evaluate.py models/best_model --render
```

### Expected Learning Outcomes
By the end of Training Session 1, the agent should learn to:
- Apply upward forces effectively
- Balance force application to prevent falling
- Coordinate multiple robots if applicable
- Discover strategies like jumping, climbing, or stacking

### Metrics to Track
- Maximum height achieved per episode
- Average height per episode
- Episode length
- Total reward
- Highest robot identification

### Future Sessions
Training Session 1 establishes the foundation for:
- **Session 2:** TBD (to be defined based on next objectives)
- **Session 3+:** Additional progressive goals in succession

---

## Training Session 2: Crush Resistance

**Status:** ✅ Implemented

**Goal:** Resist a descending hydraulic press for as long as possible.

### Overview
This is the second training session in the series. The primary objective is to teach agents to build stable structures that can resist crushing forces from above. The session features a two-phase episode: first a positioning phase (30 seconds) where robots can arrange themselves, then a crushing phase where a hydraulic press descends from above.

### Key Details
- **Environment:** `CrushResistanceEnv` (`src/environments/crush_resistance_env.py`)
- **Objective:** Survive as long as possible after the hydraulic press activates
- **Multi-Robot:** Same robot configuration as Session 1 (Type A bars + Type B spheres)
- **Physics:** Identical physics to HeightMaximizeEnv for consistency
- **Scoring:** Based on survival time after press activation (time until all robots touch ground)
- **Robot Types:**
  - Type A: Bar robots with joints (bar_with_joint.urdf) - 2N robots
  - Type B: Sphere robots (rolling_sphere.urdf) - N robots

### Episode Structure

#### Phase 1: Positioning (First 30 seconds / 1800 steps)
- Robots can move and position themselves freely
- Same physics and controls as Session 1
- Robots can form connections and structures
- Small reward for building height (encourages preparation)

#### Phase 2: Crushing (After 30 seconds)
- Hydraulic press activates at configurable height (default: 5m)
- Press descends at constant speed (default: 0.05 m/s)
- When encountering resistance, press applies increasing force
- Force increments until obstacles are crushed
- Episode ends when all robots touch the ground plane

### Reward Structure
The reward function for Training Session 2 consists of:

1. **Positioning Reward (Pre-Press):** Small reward during first 30 seconds
   - Formula: `max_height × 0.01`
   - Encourages robots to build structures in preparation

2. **Survival Reward (Post-Press):** Primary reward after press activation
   - Formula: `1.0` per step survived
   - Each step survived is valuable

3. **Height Bonus (Post-Press):** Bonus for resisting crushing
   - Formula: `avg_height × 0.5`
   - Rewards maintaining elevation under pressure

4. **Final Survival Bonus:** Large bonus at episode end
   - Formula: `survival_time × 10.0`
   - Rewards total time survived after press activation

### Action Space
Same as HeightMaximizeEnv:
- Type A robots: 6 values per robot (2 spherical joints × 3 DOF)
- Type B robots: 3 values per robot (force x, y, z)

Total action space dimension: `(num_type_a × 6) + (num_type_b × 3)`

With default N=2: `(4 × 6) + (2 × 3) = 30` continuous values ∈ [-1, 1]

### Observation Space
Same as HeightMaximizeEnv:
- Type A robots: 19 values per robot (pos, orn, vel, ang_vel, joint_states)
- Type B robots: 13 values per robot (pos, orn, vel, ang_vel)

Total observation space dimension: `(num_type_a × 19) + (num_type_b × 13)`

With default N=2: `(4 × 19) + (2 × 13) = 102` continuous values

### Hydraulic Press Mechanism

#### Press Specifications
- **Size:** 10m × 10m horizontal plane, 0.1m thick
- **Mass:** 100 kg (heavy enough to crush robots)
- **Visual:** Semi-transparent red color
- **Starting Height:** Configurable (default: 5m, should be above Session 1 max height)

#### Press Behavior
1. **Descent:** Constant velocity downward (default: 0.05 m/s)
2. **Contact Detection:** Detects collisions with robots
3. **Force Application:** When in contact, applies increasing downward force
4. **Force Increment:** Adds configurable force per step (default: 50 N)
5. **Crushing:** Continues until all robots are pressed to ground

### Termination Conditions
An episode terminates if:
- All robots touch the ground plane (z < 0.3m) - **Primary condition**
- Any robot falls through ground (z < -1.0m) - **Safety check**
- Any robot moves too far horizontally (distance > 20.0 units) - **Safety check**
- Maximum episode steps reached (default: 3000 for ~50 seconds)

### Configuration Parameters

Key configurable parameters in environment initialization:
```python
CrushResistanceEnv(
    num_type_b_robots=2,           # N (Type A will be 2N)
    spawn_radius=3.0,              # Random spawn area radius
    reference_height=5.0,          # Press starting height
    press_descent_speed=0.05,      # m/s
    press_force_increment=50.0,    # N per step
    max_episode_steps=3000         # ~50 seconds
)
```

### Usage

#### Testing the Environment
```bash
python3 examples/test_crush_env.py
```

This runs three tests:
1. Basic environment functionality test
2. Hydraulic press mechanism test
3. Reward system test

#### Training
```bash
# Create custom config for Session 2
python3 train.py --config configs/session2_config.yaml
```

#### Evaluation
```bash
python3 evaluate.py models/session2_best_model --render
```

### Expected Learning Outcomes
By the end of Training Session 2, the agent should learn to:
- Build stable structures that resist vertical crushing forces
- Position robots strategically before press activation
- Form strong connections between robots
- Distribute forces effectively across the structure
- Balance stability with height (taller = more survival space)
- Discover bracing, buttressing, and support strategies

### Metrics to Track
- Survival time after press activation (primary metric)
- Average robot height during crushing phase
- Number of robot-robot connections maintained
- Press force applied over time
- Robot positions when press activates
- Structural collapse patterns

### Relationship to Session 1
This session builds on Session 1 by:
- Using identical physics and robot types
- Requiring height-building skills from Session 1
- Adding pressure resistance as a new challenge
- Using Session 1's max height as reference for press starting height
- Encouraging structural stability over pure height maximization

Models trained in Session 1 can be fine-tuned for Session 2, as the skills overlap significantly.

### Technical Implementation Notes

#### Connection System
- Same as Session 1: Type A bar endpoints can connect to Type B spheres
- Connections maintained during crushing (unless broken by excessive force)
- Connection distance: 0.2m proximity threshold

#### Physics Consistency
- Same gravity, masses, and forces as Session 1
- Joint motors: 2× body weight lifting capacity
- Type A mass: 1.1 kg, Type B mass: 1.0 kg

#### Press Implementation
- Created as multi-body with collision and visual shape
- Uses velocity control for smooth descent
- Applies external forces when in contact
- Force accumulates until obstacles yield

---

## How to Add New Training Sessions

When adding a new training session:

1. **Create Environment:**
   - Create new environment file in `src/environments/`
   - Inherit from `BaseRobotEnv`
   - Implement required abstract methods
   - Document the session goal clearly

2. **Update Imports:**
   - Add to `src/environments/__init__.py`
   - Add to `src/training/trainer.py` environment mapping

3. **Create Configuration:**
   - Update or create config in `configs/`
   - Set `environment.name` to your new environment

4. **Test:**
   - Create test script in `examples/`
   - Verify environment works correctly

5. **Document:**
   - Add section to this file
   - Update README if necessary

---

## Notes on Progressive Training

The training sessions are designed to be run in succession, meaning:
- Each session builds on skills learned in previous sessions
- Models from earlier sessions can be fine-tuned for later sessions
- The complexity and objectives increase progressively
- Multiple sessions contribute to a more versatile, capable agent

### Best Practices
- Save checkpoints frequently during each session
- Track metrics across all sessions for comparison
- Consider curriculum learning approaches
- Test generalization between sessions
- Document insights and observations from each session

---

**Last Updated:** 2025-10-14
**Project:** RL Robotics Testing
**Repository:** https://github.com/gaskiwi/cautious-meme.git
