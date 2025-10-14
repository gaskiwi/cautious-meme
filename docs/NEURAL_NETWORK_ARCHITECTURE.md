# Neural Network Architecture for Multi-Robot Control

## Overview

This document describes the custom neural network architecture designed for controlling multiple robots in a reinforcement learning environment. The architecture supports:

- **Multiple robot types**: Type A (Bar Robot) and Type B (Sphere Robot)
- **Variable quantities**: Different numbers of each robot type
- **Dynamic environments**: Handle 1 to N robots simultaneously
- **Type awareness**: Network knows which robot is which type

## Architecture Components

### 1. Robot Encoders

Separate encoder networks for each robot type to process their unique state representations:

#### Type A Encoder (Bar Robot)
- **Input**: 51 dimensions
  - Base sphere: position (3) + orientation (4) + velocity (3) + angular velocity (3) = 13
  - Bar 1: position (3) + orientation (4) + velocity (3) + angular velocity (3) = 13
  - Bar 2: position (3) + orientation (4) + velocity (3) + angular velocity (3) = 13
  - Ball joint 1: joint angles (3) + joint velocities (3) = 6
  - Ball joint 2: joint angles (3) + joint velocities (3) = 6
  
- **Architecture**:
  ```
  Input (51) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → Embedding (128)
  ```

#### Type B Encoder (Sphere Robot)
- **Input**: 13 dimensions
  - Sphere: position (3) + orientation (4) + velocity (3) + angular velocity (3) = 13
  
- **Architecture**:
  ```
  Input (13) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → Embedding (128)
  ```

### 2. Robot Type Embeddings

A learned embedding layer that encodes robot type information:
- 3 types: `0` = padding, `1` = Type A (Bar), `2` = Type B (Sphere)
- Embedding dimension: 128
- Added to encoder output to create type-aware representations

### 3. Multi-Head Attention Mechanism

Allows the network to model interactions between robots:

- **Number of heads**: 4 (configurable)
- **Purpose**: Aggregate information across multiple robots
- **Input**: Robot embeddings `[batch_size, num_robots, 128]`
- **Output**: Attended embeddings `[batch_size, num_robots, 128]`

**Architecture**:
```
Q, K, V ← Linear projections of input
Attention_weights = softmax(Q·K^T / √d_k)
Output = Attention_weights · V
```

Supports masking to handle variable numbers of robots (padded positions are masked).

### 4. Global Context Encoder

Aggregates information from all robots to create a shared context:

```
Global_context = mean_pool(attended_embeddings, mask)
Global_features = Linear(256) → ReLU → Linear(256) → ReLU
```

### 5. Actor Network (Policy)

Generates actions for each robot based on individual embeddings + global context:

**Shared Layers**:
```
Combined = concat(robot_embedding, global_features)
Features = Linear(256) → ReLU → Linear(256) → ReLU
```

**Type-Specific Action Heads**:
- Type A head: `Linear(256) → 6` (2 spherical joints × 3 DOF each)
- Type B head: `Linear(256) → 3` (3 DOF torques for rolling)

### 6. Critic Network (Value Function)

Estimates the value of the current state:

```
Global_features → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1)
```

## Data Flow

### Forward Pass

```
1. Raw observations → Robot Encoders → Robot embeddings (128-dim)
                                     ↓
2. Robot embeddings + Type embeddings → Combined embeddings
                                     ↓
3. Combined embeddings → Multi-Head Attention → Attended embeddings
                                     ↓
4. Attended embeddings → Global Context (mean pooling) → Global features
                                     ↓
           ┌─────────────────────────┴────────────────────────┐
           ↓                                                   ↓
   Actor (per-robot)                                  Critic (global)
   Combined features → Action heads                   Global features → Value
   (Type A: 6-dim, Type B: 3-dim)                    (1-dim)
```

### Observation Format

Observations are structured as:
```python
[
  robot1_state (max_state_dim=51),
  robot1_type (1),
  robot2_state (max_state_dim=51),
  robot2_type (1),
  ...,
  num_robots (1)
]
```

### Action Format

Actions are output per robot:
```python
[
  robot1_actions (6 or 3, depending on type),
  robot2_actions (6 or 3, depending on type),
  ...
]
```

## Key Design Decisions

### 1. Separate Encoders for Robot Types

**Rationale**: Type A and Type B robots have vastly different state representations (51 vs 13 dimensions). Separate encoders allow the network to learn type-specific features optimally.

### 2. Type Embeddings

**Rationale**: Explicit type information helps the network distinguish between robot types in the attention mechanism and downstream processing.

### 3. Multi-Head Attention

**Rationale**: Robots may need to coordinate or compete for resources. Attention allows the network to model these interactions dynamically.

### 4. Masking for Variable Robot Counts

**Rationale**: Supports different numbers of robots without retraining. Padding positions are masked so they don't affect computations.

### 5. Global Context Sharing

**Rationale**: Creates a shared understanding of the overall environment state that informs individual robot actions.

### 6. Type-Specific Action Heads

**Rationale**: Type A robots have 6 controllable DOF while Type B robots have 3. Separate heads ensure each robot type gets appropriate action dimensions.

## Network Parameters

For a typical configuration with max_robots=10:

| Component | Parameters |
|-----------|-----------|
| Type A Encoder | ~131K |
| Type B Encoder | ~66K |
| Type Embeddings | 384 |
| Attention | ~66K |
| Global Encoder | ~132K |
| Actor Shared | ~132K |
| Type A Action Head | ~1.5K |
| Type B Action Head | ~768 |
| Critic | ~132K |
| **Total** | **~660K** |

## Integration with Stable Baselines3

The architecture is wrapped in a custom `RobotActorCriticPolicy` that integrates with SB3 algorithms:

```python
from src.agents import RobotActorCriticPolicy, get_robot_policy_kwargs
from stable_baselines3 import PPO

# Get policy configuration
policy_kwargs = get_robot_policy_kwargs(
    max_robots=10,
    num_type_a=3,
    num_type_b=2
)

# Create agent
agent = PPO(
    policy=RobotActorCriticPolicy,
    env=env,
    policy_kwargs=policy_kwargs,
    # ... other PPO parameters
)
```

## Usage Example

### Creating the Network

```python
from src.agents import create_robot_network

# Create network for 3 Type A and 2 Type B robots
network = create_robot_network(
    num_type_a=3,
    num_type_b=2,
    max_robots=10,
    embedding_dim=128,
    num_attention_heads=4
)
```

### Forward Pass

```python
import torch

# Prepare inputs
batch_size = 4
max_robots = 10
observations = torch.randn(batch_size, max_robots, 51)  # max state dim
robot_types = torch.tensor([
    [1, 1, 2, 2, 2, 0, 0, 0, 0, 0],  # 2 Type A, 3 Type B, rest padding
    [1, 1, 1, 2, 0, 0, 0, 0, 0, 0],  # 3 Type A, 1 Type B
    # ... more samples
])
num_robots_per_batch = torch.tensor([5, 4, ...])

# Forward pass
actions, values = network(observations, robot_types, num_robots_per_batch)
```

## Training Considerations

### 1. Curriculum Learning

Start with fewer robots and gradually increase:
1. Train with 1-2 robots
2. Fine-tune with 3-5 robots
3. Final training with full robot count

### 2. Type Balancing

Ensure training includes:
- Type A only scenarios
- Type B only scenarios  
- Mixed type scenarios

### 3. Masking Strategy

Always mask padded robots to prevent:
- Gradient flow from dummy data
- Attention weights on non-existent robots
- Value estimation bias

### 4. Action Scaling

Scale actions appropriately for each robot type:
- Type A: Joint torques (higher magnitude)
- Type B: Rolling torques (lower magnitude)

## Testing

Run the test suite to validate the architecture:

```bash
python tests/test_robot_network.py
```

Tests cover:
- Network creation with different configurations
- Forward pass with variable robot counts
- Robot type handling (A only, B only, mixed)
- Attention mechanism
- Observation/action formatting
- Parameter counts

## Future Enhancements

1. **Hierarchical Attention**: Add robot-to-robot and part-to-part attention layers
2. **Recurrent Components**: Add LSTM/GRU for temporal dependencies
3. **Graph Neural Networks**: Model robots as nodes in a graph
4. **Self-Play**: Train multiple agents to compete/cooperate
5. **Transfer Learning**: Pre-train encoders on single-robot tasks

## References

- **Multi-Head Attention**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Actor-Critic Methods**: "Asynchronous Methods for Deep RL" (Mnih et al., 2016)
- **Multi-Agent RL**: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (Lowe et al., 2017)
