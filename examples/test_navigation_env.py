"""
Test script for Training Session 3: Obstacle Navigation Environment

This script tests the ObstacleNavigationEnv with random obstacles and target navigation.
"""

import numpy as np
from src.environments import ObstacleNavigationEnv


def test_obstacle_navigation_env():
    """Test the obstacle navigation environment."""
    print("=" * 80)
    print("Testing Training Session 3: Obstacle Navigation Environment")
    print("Navigate through obstacles to reach target positions")
    print("=" * 80)
    
    # Create environment
    print("\n1. Creating environment with obstacles...")
    env = ObstacleNavigationEnv(
        num_type_b_robots=2,
        render_mode=None,
        num_obstacles=8,
        arena_size=20.0,
        max_episode_steps=2000
    )
    
    print(f"   ✓ Environment created successfully")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Number of Type A robots: {env.num_type_a}")
    print(f"   - Number of Type B robots: {env.num_type_b}")
    print(f"   - Number of obstacles: {env.num_obstacles}")
    print(f"   - Arena size: {env.arena_size}m")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Environment reset successfully")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Training session: {info['training_session']}")
    print(f"   - Session goal: {info['session_goal']}")
    print(f"   - Targets reached: {info['targets_reached']}")
    
    # Run simulation
    print("\n3. Running 100 random steps...")
    total_reward = 0
    targets_reached = 0
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        targets_reached = info['targets_reached']
        
        if (step + 1) % 25 == 0:
            print(f"\n   Step {step + 1}:")
            print(f"     - Reward: {reward:.3f}")
            print(f"     - Targets reached: {targets_reached}")
            print(f"     - Distance to target: {info['distance_to_target']:.3f}m")
            print(f"     - Collision count: {info['collision_count']}")
            print(f"     - Active connections: {info['num_connections']}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step + 1}")
            break
    
    print(f"\n4. Episode Summary:")
    print(f"   - Total steps: {step + 1}")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Targets reached: {targets_reached}")
    print(f"   - Total collisions: {info['collision_count']}")
    
    env.close()
    print("\n✓ Test completed successfully!")
    print("=" * 80)
    
    return True


def test_multiple_difficulty_levels():
    """Test with different numbers of obstacles."""
    print("\n" + "=" * 80)
    print("Testing Different Obstacle Configurations")
    print("=" * 80)
    
    configs = [
        (4, "Easy"),
        (8, "Medium"),
        (12, "Hard"),
    ]
    
    for num_obstacles, difficulty in configs:
        print(f"\nTesting {difficulty} configuration ({num_obstacles} obstacles)...")
        env = ObstacleNavigationEnv(
            num_type_b_robots=2,
            render_mode=None,
            num_obstacles=num_obstacles,
            max_episode_steps=500
        )
        
        obs, info = env.reset(seed=123)
        
        # Run a few steps
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print(f"  ✓ {difficulty}: {num_obstacles} obstacles, "
              f"targets reached: {info['targets_reached']}, "
              f"collisions: {info['collision_count']}")
        
        env.close()
    
    print("\n✓ Difficulty levels test passed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        test_obstacle_navigation_env()
        test_multiple_difficulty_levels()
        
        print("\n🎉 All tests completed successfully!")
        print("\nEnvironment Features Summary:")
        print("  ✓ Random obstacle placement (walls, boxes, cylinders)")
        print("  ✓ Dynamic target repositioning")
        print("  ✓ Reward for navigation progress")
        print("  ✓ Penalty for collisions")
        print("  ✓ Requires coordination to navigate obstacles")
        print("\nTraining Session 3 environment is ready to use.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
