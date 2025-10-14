"""Factory for creating RL agents with different algorithms."""

from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, Any
import torch.nn as nn


def create_agent(env, config: Dict[str, Any]):
    """
    Create an RL agent based on configuration.
    
    Args:
        env: Gymnasium environment
        config: Configuration dictionary
        
    Returns:
        Trained agent instance
    """
    algorithm = config['agent']['algorithm']
    
    # Common parameters
    common_params = {
        'policy': config['agent']['policy'],
        'env': env,
        'learning_rate': config['agent']['learning_rate'],
        'gamma': config['agent']['gamma'],
        'verbose': 1,
        'tensorboard_log': config['paths']['tensorboard']
    }
    
    # Policy kwargs for network architecture
    policy_kwargs = {
        'net_arch': {
            'pi': config['network']['policy_layers'],
            'vf': config['network']['value_layers']
        },
        'activation_fn': get_activation_function(config['network']['activation'])
    }
    common_params['policy_kwargs'] = policy_kwargs
    
    # Algorithm-specific parameters
    if algorithm == 'PPO':
        agent = PPO(
            **common_params,
            n_steps=config['agent']['n_steps'],
            batch_size=config['agent']['batch_size'],
            n_epochs=config['agent']['n_epochs'],
            gae_lambda=config['agent']['gae_lambda'],
            clip_range=config['agent']['clip_range'],
            ent_coef=config['agent']['ent_coef'],
            vf_coef=config['agent']['vf_coef'],
            max_grad_norm=config['agent']['max_grad_norm']
        )
    
    elif algorithm == 'SAC':
        agent = SAC(
            **common_params,
            buffer_size=config['agent'].get('buffer_size', 1000000),
            batch_size=config['agent']['batch_size'],
            tau=config['agent'].get('tau', 0.005),
            ent_coef=config['agent'].get('ent_coef', 'auto')
        )
    
    elif algorithm == 'TD3':
        agent = TD3(
            **common_params,
            buffer_size=config['agent'].get('buffer_size', 1000000),
            batch_size=config['agent']['batch_size'],
            tau=config['agent'].get('tau', 0.005),
            policy_delay=config['agent'].get('policy_delay', 2)
        )
    
    elif algorithm == 'A2C':
        agent = A2C(
            **common_params,
            n_steps=config['agent']['n_steps'],
            gae_lambda=config['agent']['gae_lambda'],
            ent_coef=config['agent']['ent_coef'],
            vf_coef=config['agent']['vf_coef'],
            max_grad_norm=config['agent']['max_grad_norm']
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent


def get_activation_function(name: str):
    """
    Get activation function by name.
    
    Args:
        name: Activation function name
        
    Returns:
        Activation function class
    """
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activations[name.lower()]
