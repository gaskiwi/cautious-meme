"""
Custom Neural Network Architecture for Multi-Robot Control

This module implements a neural network that can:
1. Handle inputs from multiple robots (Type A: Bar Robot, Type B: Sphere Robot)
2. Process different numbers of each robot type
3. Output control signals for each joint in the system
4. Distinguish between robot types using type embeddings

Architecture:
- Per-robot encoder networks (separate for Type A and Type B)
- Robot type embeddings to distinguish between types
- Attention mechanism to aggregate multi-robot information
- Separate actor and critic networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np
from gymnasium import spaces


class RobotEncoder(nn.Module):
    """
    Encodes the state of a single robot into a fixed-size embedding.
    
    For Type A (Bar Robot):
        - Base sphere: position (3), orientation (4), velocity (3), angular_vel (3) = 13
        - Bar 1: position (3), orientation (4), velocity (3), angular_vel (3) = 13
        - Bar 2: position (3), orientation (4), velocity (3), angular_vel (3) = 13
        - Ball_joint_1: joint angles (3), joint velocities (3) = 6
        - Ball_joint_2: joint angles (3), joint velocities (3) = 6
        Total: 13 + 13 + 13 + 6 + 6 = 51 dimensions
    
    For Type B (Sphere Robot):
        - Sphere: position (3), orientation (4), velocity (3), angular_vel (3) = 13
        Total: 13 dimensions
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 128):
        """
        Initialize robot encoder.
        
        Args:
            input_dim: Dimensionality of robot state input
            embedding_dim: Size of output embedding
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode robot state.
        
        Args:
            x: Robot state tensor [batch_size, input_dim]
            
        Returns:
            Embedded robot state [batch_size, embedding_dim]
        """
        return self.encoder(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for aggregating information across robots.
    Allows the network to attend to different robots based on their states.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        """
        Initialize multi-head attention.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_heads: Number of attention heads
        """
        super().__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor [batch_size, num_robots, embedding_dim]
            mask: Optional mask for padded robots [batch_size, num_robots]
            
        Returns:
            Attended output [batch_size, num_robots, embedding_dim]
        """
        batch_size, num_robots, _ = x.shape
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, num_robots, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, num_robots, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, num_robots, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_robots]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_robots, self.embedding_dim)
        output = self.out(attended)
        
        return output


class RobotAwareNetwork(nn.Module):
    """
    Main neural network architecture for multi-robot control.
    
    This network processes observations from multiple robots of different types
    and produces control outputs for each robot's joints.
    
    Features:
    - Separate encoders for Type A (Bar) and Type B (Sphere) robots
    - Robot type embeddings
    - Multi-head attention for robot interaction modeling
    - Separate actor (policy) and critic (value) heads
    """
    
    def __init__(
        self,
        type_a_state_dim: int = 51,  # Bar robot state dimension
        type_b_state_dim: int = 13,  # Sphere robot state dimension
        type_a_action_dim: int = 6,  # 2 spherical joints × 3 DOF each
        type_b_action_dim: int = 3,  # Torques for rolling (3 DOF)
        max_robots: int = 10,
        embedding_dim: int = 128,
        num_attention_heads: int = 4,
        hidden_dims: List[int] = [256, 256]
    ):
        """
        Initialize the robot-aware network.
        
        Args:
            type_a_state_dim: State dimension for Type A (Bar) robots
            type_b_state_dim: State dimension for Type B (Sphere) robots
            type_a_action_dim: Action dimension for Type A robots
            type_b_action_dim: Action dimension for Type B robots
            max_robots: Maximum number of robots in the environment
            embedding_dim: Dimension of robot embeddings
            num_attention_heads: Number of attention heads
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()
        
        self.type_a_state_dim = type_a_state_dim
        self.type_b_state_dim = type_b_state_dim
        self.type_a_action_dim = type_a_action_dim
        self.type_b_action_dim = type_b_action_dim
        self.max_robots = max_robots
        self.embedding_dim = embedding_dim
        
        # Robot encoders (separate for each type)
        self.type_a_encoder = RobotEncoder(type_a_state_dim, embedding_dim)
        self.type_b_encoder = RobotEncoder(type_b_state_dim, embedding_dim)
        
        # Robot type embeddings
        self.type_embedding = nn.Embedding(3, embedding_dim)  # 3 types: 0=padding, 1=Type A, 2=Type B
        
        # Multi-head attention for robot interactions
        self.attention = MultiHeadAttention(embedding_dim, num_attention_heads)
        
        # Global context encoder (processes aggregated robot information)
        self.global_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        # Actor network (policy) - outputs actions for each robot
        self.actor_shared = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dims[-1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        # Separate action heads for different robot types
        self.type_a_action_head = nn.Linear(hidden_dims[-1], type_a_action_dim)
        self.type_b_action_head = nn.Linear(hidden_dims[-1], type_b_action_dim)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
    def encode_robots(
        self,
        observations: torch.Tensor,
        robot_types: torch.Tensor,
        num_robots_per_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode robot observations into embeddings.
        
        Args:
            observations: Robot observations [batch_size, max_robots, max_state_dim]
            robot_types: Robot type indicators [batch_size, max_robots]
                        0=padding, 1=Type A, 2=Type B
            num_robots_per_batch: Number of actual robots per batch [batch_size]
            
        Returns:
            Tuple of (robot_embeddings, mask)
            - robot_embeddings: [batch_size, max_robots, embedding_dim]
            - mask: [batch_size, max_robots] (1 for real robots, 0 for padding)
        """
        batch_size, max_robots, _ = observations.shape
        device = observations.device
        
        # Initialize embeddings tensor
        robot_embeddings = torch.zeros(batch_size, max_robots, self.embedding_dim, device=device)
        
        # Create mask for valid robots
        mask = torch.zeros(batch_size, max_robots, device=device)
        for i in range(batch_size):
            mask[i, :num_robots_per_batch[i]] = 1
        
        # Encode Type A robots
        type_a_mask = (robot_types == 1) & (mask.bool())
        if type_a_mask.any():
            type_a_obs = observations[type_a_mask, :self.type_a_state_dim]
            type_a_emb = self.type_a_encoder(type_a_obs)
            robot_embeddings[type_a_mask] = type_a_emb
        
        # Encode Type B robots
        type_b_mask = (robot_types == 2) & (mask.bool())
        if type_b_mask.any():
            type_b_obs = observations[type_b_mask, :self.type_b_state_dim]
            type_b_emb = self.type_b_encoder(type_b_obs)
            robot_embeddings[type_b_mask] = type_b_emb
        
        # Add type embeddings
        type_emb = self.type_embedding(robot_types.long())
        robot_embeddings = robot_embeddings + type_emb
        
        return robot_embeddings, mask
    
    def forward_actor(
        self,
        observations: torch.Tensor,
        robot_types: torch.Tensor,
        num_robots_per_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the actor network to get actions.
        
        Args:
            observations: Robot observations [batch_size, max_robots, max_state_dim]
            robot_types: Robot type indicators [batch_size, max_robots]
            num_robots_per_batch: Number of robots per batch [batch_size]
            
        Returns:
            Actions for all robots [batch_size, max_robots, max_action_dim]
        """
        batch_size, max_robots, _ = observations.shape
        device = observations.device
        
        # Encode robots
        robot_embeddings, mask = self.encode_robots(observations, robot_types, num_robots_per_batch)
        
        # Apply attention
        attended_embeddings = self.attention(robot_embeddings, mask)
        
        # Global context (mean pooling over robots)
        global_context = (attended_embeddings * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        global_features = self.global_encoder(global_context)
        
        # Combine robot embeddings with global context
        global_features_expanded = global_features.unsqueeze(1).expand(-1, max_robots, -1)
        combined_features = torch.cat([attended_embeddings, global_features_expanded], dim=-1)
        
        # Actor shared layers
        actor_features = self.actor_shared(combined_features)
        
        # Generate actions for each robot type
        max_action_dim = max(self.type_a_action_dim, self.type_b_action_dim)
        actions = torch.zeros(batch_size, max_robots, max_action_dim, device=device)
        
        # Type A actions
        type_a_mask = (robot_types == 1) & (mask.bool())
        if type_a_mask.any():
            type_a_actions = self.type_a_action_head(actor_features[type_a_mask])
            actions[type_a_mask, :self.type_a_action_dim] = type_a_actions
        
        # Type B actions
        type_b_mask = (robot_types == 2) & (mask.bool())
        if type_b_mask.any():
            type_b_actions = self.type_b_action_head(actor_features[type_b_mask])
            actions[type_b_mask, :self.type_b_action_dim] = type_b_actions
        
        return actions
    
    def forward_critic(
        self,
        observations: torch.Tensor,
        robot_types: torch.Tensor,
        num_robots_per_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the critic network to get value estimate.
        
        Args:
            observations: Robot observations [batch_size, max_robots, max_state_dim]
            robot_types: Robot type indicators [batch_size, max_robots]
            num_robots_per_batch: Number of robots per batch [batch_size]
            
        Returns:
            Value estimates [batch_size, 1]
        """
        # Encode robots
        robot_embeddings, mask = self.encode_robots(observations, robot_types, num_robots_per_batch)
        
        # Apply attention
        attended_embeddings = self.attention(robot_embeddings, mask)
        
        # Global context (mean pooling over robots)
        global_context = (attended_embeddings * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        global_features = self.global_encoder(global_context)
        
        # Value estimate
        value = self.critic(global_features)
        
        return value
    
    def forward(
        self,
        observations: torch.Tensor,
        robot_types: torch.Tensor,
        num_robots_per_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass (both actor and critic).
        
        Args:
            observations: Robot observations [batch_size, max_robots, max_state_dim]
            robot_types: Robot type indicators [batch_size, max_robots]
            num_robots_per_batch: Number of robots per batch [batch_size]
            
        Returns:
            Tuple of (actions, values)
        """
        actions = self.forward_actor(observations, robot_types, num_robots_per_batch)
        values = self.forward_critic(observations, robot_types, num_robots_per_batch)
        
        return actions, values


def create_robot_network(
    num_type_a: int = 0,
    num_type_b: int = 0,
    **kwargs
) -> RobotAwareNetwork:
    """
    Factory function to create a robot-aware network.
    
    Args:
        num_type_a: Number of Type A (Bar) robots
        num_type_b: Number of Type B (Sphere) robots
        **kwargs: Additional arguments for RobotAwareNetwork
        
    Returns:
        Initialized RobotAwareNetwork
    """
    max_robots = kwargs.pop('max_robots', num_type_a + num_type_b)
    return RobotAwareNetwork(max_robots=max_robots, **kwargs)


if __name__ == "__main__":
    # Test the network architecture
    print("Testing Robot-Aware Network Architecture...")
    
    # Create network
    network = create_robot_network(num_type_a=2, num_type_b=1, max_robots=5)
    
    # Create dummy inputs
    batch_size = 4
    max_robots = 5
    max_state_dim = 51  # Max between Type A and Type B
    
    observations = torch.randn(batch_size, max_robots, max_state_dim)
    robot_types = torch.tensor([
        [1, 1, 2, 0, 0],  # 2 Type A, 1 Type B, 2 padding
        [1, 2, 2, 0, 0],  # 1 Type A, 2 Type B, 2 padding
        [2, 2, 2, 2, 0],  # 4 Type B, 1 padding
        [1, 1, 1, 2, 2],  # 3 Type A, 2 Type B
    ])
    num_robots_per_batch = torch.tensor([3, 3, 4, 5])
    
    # Forward pass
    actions, values = network(observations, robot_types, num_robots_per_batch)
    
    print(f"Input shape: {observations.shape}")
    print(f"Robot types shape: {robot_types.shape}")
    print(f"Output actions shape: {actions.shape}")
    print(f"Output values shape: {values.shape}")
    
    # Test individual forwards
    actions_only = network.forward_actor(observations, robot_types, num_robots_per_batch)
    values_only = network.forward_critic(observations, robot_types, num_robots_per_batch)
    
    print(f"\nActions only shape: {actions_only.shape}")
    print(f"Values only shape: {values_only.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ Network architecture test passed!")
