#!/usr/bin/env python3
"""
Quick test script to verify the environment and agent setup.

This script creates a simple environment, runs a few random episodes,
and verifies that all components are working correctly.
"""

import numpy as np
from src.environments import SimpleRobotEnv
from src.utils.logger import setup_logger


def test_environment():
    """Test basic environment functionality."""
    logger = setup_logger('test')
    logger.info("Testing environment setup...")
    
    # Create environment
    env = SimpleRobotEnv(render_mode=None)
    
    # Reset environment
    obs, info = env.reset()
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Initial observation: {obs}")
    
    # Run a few random steps
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            logger.info(f"Episode ended at step {step + 1}")
            logger.info(f"Total reward: {total_reward:.2f}")
            break
    
    env.close()
    logger.info("Environment test completed successfully!")
    
    return True


def test_config():
    """Test configuration loading."""
    logger = setup_logger('test')
    logger.info("Testing configuration loading...")
    
    from src.utils.config_loader import load_config
    
    config = load_config('configs/training_config.yaml')
    
    logger.info(f"Algorithm: {config['agent']['algorithm']}")
    logger.info(f"Learning rate: {config['agent']['learning_rate']}")
    logger.info(f"Total timesteps: {config['training']['total_timesteps']}")
    
    logger.info("Configuration test completed successfully!")
    
    return True


def test_agent_creation():
    """Test agent creation (without training)."""
    logger = setup_logger('test')
    logger.info("Testing agent creation...")
    
    from src.environments import SimpleRobotEnv
    from src.agents import create_agent
    from src.utils.config_loader import load_config
    
    # Load config
    config = load_config('configs/training_config.yaml')
    
    # Create environment
    env = SimpleRobotEnv()
    
    # Create agent
    logger.info(f"Creating {config['agent']['algorithm']} agent...")
    agent = create_agent(env, config)
    
    logger.info("Agent created successfully!")
    logger.info(f"Policy: {agent.policy}")
    
    env.close()
    
    return True


def run_all_tests():
    """Run all tests."""
    logger = setup_logger('main')
    
    tests = [
        ("Environment Test", test_environment),
        ("Configuration Test", test_config),
        ("Agent Creation Test", test_agent_creation),
    ]
    
    results = []
    
    logger.info("=" * 60)
    logger.info("Running RL Robotics Quick Tests")
    logger.info("=" * 60)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {test_name}")
        logger.info("=" * 60)
        
        try:
            success = test_func()
            results.append((test_name, "PASSED"))
            logger.info(f"✓ {test_name} PASSED")
        except Exception as e:
            results.append((test_name, f"FAILED: {str(e)}"))
            logger.error(f"✗ {test_name} FAILED: {str(e)}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for test_name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        logger.info(f"{status} {test_name}: {result}")
    
    all_passed = all(result == "PASSED" for _, result in results)
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Your environment is ready.")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python train.py")
        logger.info("  2. Monitor: tensorboard --logdir runs/")
        logger.info("  3. Evaluate: python evaluate.py models/best_model")
    else:
        logger.error("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
