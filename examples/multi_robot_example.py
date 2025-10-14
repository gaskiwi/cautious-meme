"""
Example: Multi-Robot Training with Custom Neural Network

This example demonstrates how to use the custom RobotAwareNetwork
for training multiple robots (Type A: Bar, Type B: Sphere) in the same environment.

The example shows:
1. Creating a multi-robot environment
2. Configuring the custom network architecture
3. Training with PPO using the robot-aware policy
4. Evaluating the trained model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Dict, Any

# This example requires the full dependencies to be installed
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    
    from src.agents import (
        RobotActorCriticPolicy,
        get_robot_policy_kwargs,
        MultiRobotEnvironmentWrapper
    )
    
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install -r requirements.txt")
    DEPENDENCIES_AVAILABLE = False


def create_multi_robot_config(num_type_a: int = 2, num_type_b: int = 2) -> Dict[str, Any]:
    """
    Create configuration for multi-robot training.
    
    Args:
        num_type_a: Number of Type A (Bar) robots
        num_type_b: Number of Type B (Sphere) robots
        
    Returns:
        Configuration dictionary
    """
    max_robots = num_type_a + num_type_b + 5  # Add buffer for dynamic scenarios
    
    config = {
        'robot': {
            'num_type_a': num_type_a,
            'num_type_b': num_type_b,
            'max_robots': max_robots,
            'type_a_state_dim': 51,  # Bar robot state dimension
            'type_b_state_dim': 13,  # Sphere robot state dimension
            'type_a_action_dim': 6,  # 2 spherical joints × 3 DOF
            'type_b_action_dim': 3,  # 3 DOF for rolling
        },
        'network': {
            'embedding_dim': 128,
            'num_attention_heads': 4,
            'policy_layers': [256, 256],
            'value_layers': [256, 256],
            'activation': 'relu'
        },
        'agent': {
            'algorithm': 'PPO',
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,  # Encourage exploration
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        },
        'training': {
            'total_timesteps': 1_000_000,
            'eval_freq': 10_000,
            'eval_episodes': 10,
            'save_freq': 50_000,
        },
        'paths': {
            'models': './models/multi_robot',
            'logs': './logs/multi_robot',
            'tensorboard': './runs/multi_robot',
        }
    }
    
    return config


def example_basic_usage():
    """
    Basic example: Create and test the network (without environment).
    """
    print("="*70)
    print("EXAMPLE 1: Basic Network Creation and Testing")
    print("="*70)
    
    from src.agents import create_robot_network
    
    # Create network
    network = create_robot_network(
        num_type_a=2,
        num_type_b=3,
        max_robots=10,
        embedding_dim=128,
        num_attention_heads=4
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"\nNetwork created with {total_params:,} parameters")
    
    print("\nNetwork structure:")
    print(f"  - Type A Encoder: {network.type_a_state_dim} → 128")
    print(f"  - Type B Encoder: {network.type_b_state_dim} → 128")
    print(f"  - Attention: {network.embedding_dim}-dim, {network.attention.num_heads} heads")
    print(f"  - Actor: Outputs {network.type_a_action_dim}-dim (Type A) or {network.type_b_action_dim}-dim (Type B)")
    print(f"  - Critic: Outputs 1-dim value")
    
    print("\n✓ Basic network creation successful!")


def example_observation_formatting():
    """
    Example: Format observations for the network.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Observation Formatting")
    print("="*70)
    
    from src.agents import MultiRobotEnvironmentWrapper
    
    # Simulate robot states
    type_a_state = np.random.randn(51)  # Bar robot
    type_b_state = np.random.randn(13)  # Sphere robot
    
    robot_states = [type_a_state, type_a_state, type_b_state]
    robot_types = [1, 1, 2]  # Type A, Type A, Type B
    max_robots = 5
    
    # Format observation
    obs = MultiRobotEnvironmentWrapper.format_observation(
        robot_states, robot_types, max_robots
    )
    
    print(f"\nFormatted observation shape: {obs.shape}")
    print(f"Expected shape: {max_robots * 52 + 1} (5 robots × (51 state + 1 type) + 1 count)")
    print(f"Number of robots: {int(obs[-1])}")
    
    print("\n✓ Observation formatting successful!")


def example_with_ppo_training():
    """
    Example: Full training loop with PPO and custom policy.
    
    Note: This requires a compatible multi-robot environment.
    """
    if not DEPENDENCIES_AVAILABLE:
        print("\n" + "="*70)
        print("EXAMPLE 3: PPO Training (SKIPPED - dependencies not available)")
        print("="*70)
        print("Install dependencies with: pip install -r requirements.txt")
        return
    
    print("\n" + "="*70)
    print("EXAMPLE 3: PPO Training with Custom Policy")
    print("="*70)
    
    # This is a template - you need to create your actual multi-robot environment
    print("\nNote: This example requires a multi-robot environment implementation.")
    print("See src/environments/ for environment examples.")
    
    # Configuration
    config = create_multi_robot_config(num_type_a=2, num_type_b=2)
    
    # Get policy kwargs
    policy_kwargs = get_robot_policy_kwargs(
        max_robots=config['robot']['max_robots'],
        num_type_a=config['robot']['num_type_a'],
        num_type_b=config['robot']['num_type_b'],
        type_a_state_dim=config['robot']['type_a_state_dim'],
        type_b_state_dim=config['robot']['type_b_state_dim'],
        type_a_action_dim=config['robot']['type_a_action_dim'],
        type_b_action_dim=config['robot']['type_b_action_dim'],
        embedding_dim=config['network']['embedding_dim'],
        num_attention_heads=config['network']['num_attention_heads'],
        net_arch={
            'pi': config['network']['policy_layers'],
            'vf': config['network']['value_layers']
        }
    )
    
    print("\nPolicy configuration:")
    print(f"  - Max robots: {policy_kwargs['max_robots']}")
    print(f"  - Type A robots: {config['robot']['num_type_a']}")
    print(f"  - Type B robots: {config['robot']['num_type_b']}")
    print(f"  - Embedding dim: {policy_kwargs['embedding_dim']}")
    print(f"  - Attention heads: {policy_kwargs['num_attention_heads']}")
    
    # Create environment (placeholder)
    # env = YourMultiRobotEnv(num_type_a=2, num_type_b=2)
    
    # Create agent
    # agent = PPO(
    #     policy=RobotActorCriticPolicy,
    #     env=env,
    #     learning_rate=config['agent']['learning_rate'],
    #     n_steps=config['agent']['n_steps'],
    #     batch_size=config['agent']['batch_size'],
    #     n_epochs=config['agent']['n_epochs'],
    #     gamma=config['agent']['gamma'],
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    #     tensorboard_log=config['paths']['tensorboard']
    # )
    
    # Train
    # agent.learn(total_timesteps=config['training']['total_timesteps'])
    
    print("\n✓ Training configuration ready (environment implementation needed)")


def example_curriculum_learning():
    """
    Example: Curriculum learning approach for multi-robot training.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Curriculum Learning Strategy")
    print("="*70)
    
    print("\nCurriculum stages:")
    
    stages = [
        {
            'stage': 1,
            'description': 'Single robot (Type A only)',
            'num_type_a': 1,
            'num_type_b': 0,
            'timesteps': 200_000,
        },
        {
            'stage': 2,
            'description': 'Single robot (Type B only)',
            'num_type_a': 0,
            'num_type_b': 1,
            'timesteps': 200_000,
        },
        {
            'stage': 3,
            'description': 'Two robots (one of each type)',
            'num_type_a': 1,
            'num_type_b': 1,
            'timesteps': 300_000,
        },
        {
            'stage': 4,
            'description': 'Multiple robots (mixed)',
            'num_type_a': 2,
            'num_type_b': 2,
            'timesteps': 500_000,
        },
        {
            'stage': 5,
            'description': 'Full complexity',
            'num_type_a': 3,
            'num_type_b': 3,
            'timesteps': 1_000_000,
        },
    ]
    
    total_timesteps = 0
    for stage in stages:
        print(f"\nStage {stage['stage']}: {stage['description']}")
        print(f"  Type A: {stage['num_type_a']}, Type B: {stage['num_type_b']}")
        print(f"  Training timesteps: {stage['timesteps']:,}")
        total_timesteps += stage['timesteps']
    
    print(f"\nTotal curriculum training: {total_timesteps:,} timesteps")
    print("\n✓ Curriculum learning strategy outlined!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MULTI-ROBOT NEURAL NETWORK EXAMPLES")
    print("="*70)
    
    # Run examples
    example_basic_usage()
    example_observation_formatting()
    example_with_ppo_training()
    example_curriculum_learning()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Implement your multi-robot environment (see src/environments/)")
    print("2. Configure training parameters (see configs/)")
    print("3. Run training with: python train.py --config your_config.yaml")
    print("4. Monitor with TensorBoard: tensorboard --logdir runs/")
    print("\nFor more information, see:")
    print("  - docs/NEURAL_NETWORK_ARCHITECTURE.md")
    print("  - tests/test_robot_network.py")
    print("="*70)


if __name__ == "__main__":
    main()
