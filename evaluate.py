#!/usr/bin/env python3
"""
Evaluation entry point for trained RL models.
"""

from src.training.evaluator import evaluate_agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model (e.g., models/best_model)"
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
        help="Render the environment during evaluation"
    )
    
    args = parser.parse_args()
    
    evaluate_agent(
        model_path=args.model_path,
        config_path=args.config,
        n_episodes=args.episodes,
        render=args.render
    )
