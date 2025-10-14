"""Evaluation script for trained RL agents."""

from pathlib import Path
from typing import Optional
import numpy as np

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.environments import SimpleRobotEnv
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_agent(
    model_path: str,
    config_path: str = "configs/training_config.yaml",
    n_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True
):
    """
    Evaluate a trained RL agent.
    
    Args:
        model_path: Path to saved model
        config_path: Path to configuration file
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logger
    logger = setup_logger('evaluator')
    
    logger.info(f"Evaluating model: {model_path}")
    
    # Create environment
    render_mode = "human" if render else None
    env = SimpleRobotEnv(
        render_mode=render_mode,
        max_episode_steps=config['environment']['max_episode_steps']
    )
    
    # Load agent
    from stable_baselines3 import PPO, SAC, TD3, A2C
    algorithm_classes = {
        'PPO': PPO,
        'SAC': SAC,
        'TD3': TD3,
        'A2C': A2C
    }
    
    algorithm = config['agent']['algorithm']
    AgentClass = algorithm_classes[algorithm]
    
    logger.info(f"Loading {algorithm} agent...")
    agent = AgentClass.load(model_path)
    
    # Evaluate
    logger.info(f"Evaluating for {n_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        agent,
        env,
        n_eval_episodes=n_episodes,
        deterministic=deterministic,
        render=render
    )
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    
    return mean_reward, std_reward


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    
    args = parser.parse_args()
    
    evaluate_agent(
        model_path=args.model_path,
        config_path=args.config,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic
    )
