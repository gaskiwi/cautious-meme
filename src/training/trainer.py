"""Training script for RL agents."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import gymnasium as gym

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.agents.agent_factory import create_agent
from src.agents.callbacks import ProgressCallback, SaveBestModelCallback
from src.environments import SimpleRobotEnv, HeightMaximizeEnv


def train_agent(
    config_path: str = "configs/training_config.yaml",
    checkpoint_path: Optional[str] = None,
    total_timesteps: Optional[int] = None
):
    """
    Train an RL agent.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to checkpoint to resume from
        total_timesteps: Override total training timesteps
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logger
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('trainer', log_file=str(log_dir / 'training.log'))
    
    logger.info("Starting RL training...")
    logger.info(f"Algorithm: {config['agent']['algorithm']}")
    
    # Create directories
    models_dir = Path(config['paths']['models'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    logger.info("Creating environment...")
    env_name = config['environment'].get('name', 'simple_robot_env')
    
    # Map environment names to classes
    env_classes = {
        'simple_robot_env': SimpleRobotEnv,
        'height_maximize_env': HeightMaximizeEnv,
    }
    
    EnvClass = env_classes.get(env_name, SimpleRobotEnv)
    
    # Get environment-specific parameters
    env_kwargs = {
        'render_mode': config['environment']['render_mode'],
        'max_episode_steps': config['environment']['max_episode_steps']
    }
    
    # Add num_robots parameter for HeightMaximizeEnv
    if env_name == 'height_maximize_env' and 'num_robots' in config['environment']:
        env_kwargs['num_robots'] = config['environment']['num_robots']
    
    env = EnvClass(**env_kwargs)
    
    # Create evaluation environment (without rendering)
    eval_kwargs = env_kwargs.copy()
    eval_kwargs['render_mode'] = None
    eval_env = EnvClass(**eval_kwargs)
    
    # Create or load agent
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading agent from checkpoint: {checkpoint_path}")
        # Load the appropriate algorithm class
        from stable_baselines3 import PPO, SAC, TD3, A2C
        algorithm_classes = {
            'PPO': PPO,
            'SAC': SAC,
            'TD3': TD3,
            'A2C': A2C
        }
        AgentClass = algorithm_classes[config['agent']['algorithm']]
        agent = AgentClass.load(checkpoint_path, env=env)
    else:
        logger.info("Creating new agent...")
        agent = create_agent(env, config)
    
    # Setup callbacks
    callbacks = [
        ProgressCallback(verbose=1),
        SaveBestModelCallback(
            eval_env=eval_env,
            save_path=str(models_dir),
            eval_freq=config['training']['eval_freq'],
            verbose=1
        )
    ]
    
    # Train
    timesteps = total_timesteps or config['training']['total_timesteps']
    logger.info(f"Training for {timesteps} timesteps...")
    
    try:
        agent.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            log_interval=config['training']['log_interval'],
            progress_bar=True
        )
        
        # Save final model
        final_model_path = models_dir / "final_model"
        agent.save(final_model_path)
        logger.info(f"Training completed! Model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint
        checkpoint_path = models_dir / "checkpoint_interrupted"
        agent.save(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    finally:
        env.close()
        eval_env.close()
    
    return agent


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total timesteps to train for"
    )
    
    args = parser.parse_args()
    
    train_agent(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        total_timesteps=args.timesteps
    )
