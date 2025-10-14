"""
Test script for Training Session 1: Height Maximization Environment

This script tests the HeightMaximizeEnv to ensure it works correctly.
"""

import numpy as np
from src.environments import HeightMaximizeEnv


def test_height_maximize_env():
    """Test the height maximization environment."""
    print("=" * 60)
    print("Testing Training Session 1: Height Maximization Environment")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment with 3 robots...")
    env = HeightMaximizeEnv(num_robots=3, render_mode=None)
    
    print(f"   ✓ Environment created successfully")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Number of robots: {env.num_robots}")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Environment reset successfully")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Training session: {info['training_session']}")
    print(f"   - Session goal: {info['session_goal']}")
    print(f"   - Initial max height: {info['max_height']:.3f}")
    
    # Run a few random steps
    print("\n3. Running 10 random steps...")
    total_reward = 0
    max_height_seen = 0
    
    for step in range(10):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        max_height_seen = max(max_height_seen, info['max_height'])
        
        print(f"   Step {step + 1}:")
        print(f"     - Reward: {reward:.3f}")
        print(f"     - Max height: {info['max_height']:.3f}")
        print(f"     - Robot heights: {[f'{h:.3f}' for h in info['robot_heights']]}")
        print(f"     - Highest robot: Robot #{info['highest_robot']}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step + 1}")
            break
    
    print(f"\n4. Episode Summary:")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Maximum height achieved: {max_height_seen:.3f}")
    
    # Close environment
    env.close()
    print("\n✓ All tests passed successfully!")
    print("=" * 60)
    
    return True


def test_multiple_episodes():
    """Test running multiple episodes."""
    print("\n" + "=" * 60)
    print("Testing Multiple Episodes")
    print("=" * 60)
    
    env = HeightMaximizeEnv(num_robots=3, render_mode=None)
    
    episode_max_heights = []
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset()
        
        episode_reward = 0
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        episode_max_heights.append(info['max_height'])
        print(f"  - Steps: {step + 1}")
        print(f"  - Total reward: {episode_reward:.3f}")
        print(f"  - Max height: {info['max_height']:.3f}")
    
    env.close()
    
    print(f"\nAverage max height across episodes: {np.mean(episode_max_heights):.3f}")
    print("✓ Multiple episodes test passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        # Run basic test
        test_height_maximize_env()
        
        # Run multiple episodes test
        test_multiple_episodes()
        
        print("\n🎉 All tests completed successfully!")
        print("\nTraining Session 1 environment is ready to use.")
        print("Run 'python train.py' to start training with the height maximization goal.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
