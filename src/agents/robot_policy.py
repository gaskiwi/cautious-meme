"""
Stable Baselines3 compatible policy wrapper for the Robot-Aware Network.

This module provides a custom policy that integrates the RobotAwareNetwork
with Stable Baselines3's PPO, SAC, and other algorithms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .robot_network import RobotAwareNetwork


class RobotFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that parses robot observations and types.
    
    This extractor processes the observation to separate:
    1. Robot states (positions, orientations, velocities)
    2. Robot type indicators
    3. Number of robots
    
    The observation is expected to be a dictionary with keys:
    - 'robot_states': [num_robots * max_state_dim] flattened array
    - 'robot_types': [num_robots] array of type indicators
    - 'num_robots': scalar indicating number of active robots
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        max_robots: int = 10,
        type_a_state_dim: int = 51,
        type_b_state_dim: int = 13,
        embedding_dim: int = 128,
        num_attention_heads: int = 4
    ):
        """
        Initialize the features extractor.
        
        Args:
            observation_space: Observation space of the environment
            features_dim: Dimension of extracted features
            max_robots: Maximum number of robots
            type_a_state_dim: State dimension for Type A robots
            type_b_state_dim: State dimension for Type B robots
            embedding_dim: Dimension of robot embeddings
            num_attention_heads: Number of attention heads
        """
        super().__init__(observation_space, features_dim)
        
        self.max_robots = max_robots
        self.type_a_state_dim = type_a_state_dim
        self.type_b_state_dim = type_b_state_dim
        self.max_state_dim = max(type_a_state_dim, type_b_state_dim)
        
        # Robot-aware network for encoding
        self.robot_network = RobotAwareNetwork(
            type_a_state_dim=type_a_state_dim,
            type_b_state_dim=type_b_state_dim,
            max_robots=max_robots,
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            hidden_dims=[features_dim, features_dim]
        )
        
    def parse_observation(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse the flattened observation into structured format.
        
        The observation format is:
        [robot1_state(max_state_dim), robot1_type(1),
         robot2_state(max_state_dim), robot2_type(1),
         ...,
         num_robots(1)]
        
        Args:
            observations: Flattened observations [batch_size, obs_dim]
            
        Returns:
            Tuple of (robot_states, robot_types, num_robots_per_batch)
        """
        batch_size = observations.shape[0]
        device = observations.device
        
        # Extract number of robots (last element)
        num_robots_per_batch = observations[:, -1].long()
        
        # Reshape to extract robot data
        # Each robot has max_state_dim + 1 (state + type)
        robot_data_dim = self.max_state_dim + 1
        robot_data = observations[:, :-1].view(batch_size, self.max_robots, robot_data_dim)
        
        # Separate states and types
        robot_states = robot_data[:, :, :-1]  # [batch_size, max_robots, max_state_dim]
        robot_types = robot_data[:, :, -1]    # [batch_size, max_robots]
        
        return robot_states, robot_types, num_robots_per_batch
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Observations from environment
            
        Returns:
            Extracted features [batch_size, features_dim]
        """
        # Parse observations
        robot_states, robot_types, num_robots = self.parse_observation(observations)
        
        # Encode robots
        robot_embeddings, mask = self.robot_network.encode_robots(
            robot_states, robot_types, num_robots
        )
        
        # Apply attention
        attended_embeddings = self.robot_network.attention(robot_embeddings, mask)
        
        # Global context (mean pooling)
        features = (attended_embeddings * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        features = self.robot_network.global_encoder(features)
        
        return features


class RobotActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy for multi-robot control.
    
    This policy uses the RobotAwareNetwork architecture to handle
    multiple robots of different types (Type A: Bar, Type B: Sphere).
    
    The policy is compatible with Stable Baselines3 algorithms like PPO.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        max_robots: int = 10,
        type_a_state_dim: int = 51,
        type_b_state_dim: int = 13,
        type_a_action_dim: int = 6,
        type_b_action_dim: int = 3,
        embedding_dim: int = 128,
        num_attention_heads: int = 4,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs
    ):
        """
        Initialize the robot actor-critic policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            max_robots: Maximum number of robots
            type_a_state_dim: State dimension for Type A robots
            type_b_state_dim: State dimension for Type B robots
            type_a_action_dim: Action dimension for Type A robots
            type_b_action_dim: Action dimension for Type B robots
            embedding_dim: Dimension of robot embeddings
            num_attention_heads: Number of attention heads
            net_arch: Network architecture specification
            activation_fn: Activation function
        """
        self.max_robots = max_robots
        self.type_a_state_dim = type_a_state_dim
        self.type_b_state_dim = type_b_state_dim
        self.type_a_action_dim = type_a_action_dim
        self.type_b_action_dim = type_b_action_dim
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        
        # Set default network architecture if not provided
        if net_arch is None:
            net_arch = dict(pi=[256, 256], vf=[256, 256])
        
        # Initialize parent class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )
        
    def _build_mlp_extractor(self) -> None:
        """
        Build the feature extractor and policy/value networks.
        
        This overrides the default MLP extractor to use our custom
        RobotFeaturesExtractor.
        """
        # Determine features_dim from net_arch
        if isinstance(self.net_arch, dict):
            features_dim = self.net_arch.get('pi', [256])[0]
        else:
            features_dim = self.net_arch[0] if self.net_arch else 256
        
        # Create custom features extractor
        self.features_extractor = RobotFeaturesExtractor(
            observation_space=self.observation_space,
            features_dim=features_dim,
            max_robots=self.max_robots,
            type_a_state_dim=self.type_a_state_dim,
            type_b_state_dim=self.type_b_state_dim,
            embedding_dim=self.embedding_dim,
            num_attention_heads=self.num_attention_heads
        )
        
        # Use the parent class's method for building the rest
        # But we need to manually set the latent dimensions
        self.mlp_extractor = None  # We'll handle this in forward


class MultiRobotEnvironmentWrapper:
    """
    Utility class to help format observations for the RobotActorCriticPolicy.
    
    This wrapper can be used to transform environment observations into
    the expected format for the custom policy.
    """
    
    @staticmethod
    def format_observation(
        robot_states: List[np.ndarray],
        robot_types: List[int],
        max_robots: int,
        max_state_dim: int = 51
    ) -> np.ndarray:
        """
        Format robot observations for the policy.
        
        Args:
            robot_states: List of robot state arrays
            robot_types: List of robot type indicators (1=Type A, 2=Type B)
            max_robots: Maximum number of robots
            max_state_dim: Maximum state dimension
            
        Returns:
            Formatted observation array
        """
        num_robots = len(robot_states)
        assert num_robots <= max_robots, f"Too many robots: {num_robots} > {max_robots}"
        
        # Create padded observation
        obs = np.zeros((max_robots, max_state_dim + 1), dtype=np.float32)
        
        for i in range(num_robots):
            # Pad robot state to max_state_dim
            state = robot_states[i]
            obs[i, :len(state)] = state
            # Set robot type
            obs[i, -1] = robot_types[i]
        
        # Flatten and append num_robots
        obs_flat = obs.flatten()
        obs_final = np.concatenate([obs_flat, [num_robots]])
        
        return obs_final
    
    @staticmethod
    def parse_action(
        action: np.ndarray,
        robot_types: List[int],
        type_a_action_dim: int = 6,
        type_b_action_dim: int = 3
    ) -> List[np.ndarray]:
        """
        Parse the network output into per-robot actions.
        
        Args:
            action: Flattened action array from policy
            robot_types: List of robot type indicators
            type_a_action_dim: Action dimension for Type A
            type_b_action_dim: Action dimension for Type B
            
        Returns:
            List of per-robot action arrays
        """
        max_action_dim = max(type_a_action_dim, type_b_action_dim)
        num_robots = len(robot_types)
        
        # Reshape action
        actions_reshaped = action.reshape(-1, max_action_dim)
        
        # Extract actions for each robot
        robot_actions = []
        for i, robot_type in enumerate(robot_types):
            if robot_type == 1:  # Type A
                robot_actions.append(actions_reshaped[i, :type_a_action_dim])
            elif robot_type == 2:  # Type B
                robot_actions.append(actions_reshaped[i, :type_b_action_dim])
            else:
                robot_actions.append(np.zeros(max_action_dim))
        
        return robot_actions


# Register the custom policy
def get_robot_policy_kwargs(
    max_robots: int = 10,
    num_type_a: int = 0,
    num_type_b: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Get policy kwargs for creating a RobotActorCriticPolicy.
    
    Args:
        max_robots: Maximum number of robots
        num_type_a: Number of Type A robots
        num_type_b: Number of Type B robots
        **kwargs: Additional policy arguments
        
    Returns:
        Dictionary of policy kwargs
    """
    return {
        'max_robots': max_robots,
        'type_a_state_dim': kwargs.get('type_a_state_dim', 51),
        'type_b_state_dim': kwargs.get('type_b_state_dim', 13),
        'type_a_action_dim': kwargs.get('type_a_action_dim', 6),
        'type_b_action_dim': kwargs.get('type_b_action_dim', 3),
        'embedding_dim': kwargs.get('embedding_dim', 128),
        'num_attention_heads': kwargs.get('num_attention_heads', 4),
        'net_arch': kwargs.get('net_arch', dict(pi=[256, 256], vf=[256, 256])),
        'activation_fn': kwargs.get('activation_fn', nn.ReLU),
    }
