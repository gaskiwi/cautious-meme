#!/usr/bin/env python3
"""
Main training entry point for RL robotics project.
"""

from src.training.trainer import train_agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL robot agent")
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
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total training timesteps"
    )
    
    args = parser.parse_args()
    
    train_agent(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        total_timesteps=args.timesteps
    )
