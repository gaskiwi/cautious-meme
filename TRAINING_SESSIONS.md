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

## Training Session 2: [To Be Defined]

**Status:** ⏳ Pending

**Goal:** TBD

*This session will be defined based on project requirements and the success of Session 1.*

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
