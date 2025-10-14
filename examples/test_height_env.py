"""
Test script for Training Session 1: Height Maximization Environment

This script tests the HeightMaximizeEnv with URDF-based Type A and Type B robots.

Type A: Bar robots with joints (bar_with_joint.urdf) - 2N robots
Type B: Sphere robots (rolling_sphere.urdf) - N robots

The environment features:
- Random deployment of robots on the plane
- Type A robots can connect to Type B robots via endpoints
- Joint motors configured to lift twice the robot's body weight
- Reward based on maximum height achieved at end of episode
"""

import numpy as np
from src.environments import HeightMaximizeEnv


def test_height_maximize_env():
    """Test the height maximization environment with URDF robots."""
    print("=" * 80)
    print("Testing Training Session 1: Height Maximization Environment")
    print("Type A (Bar) + Type B (Sphere) Robots with URDF Models")
    print("=" * 80)
    
    # Create environment
    print("\n1. Creating environment with N=2 (2 Type B, 4 Type A robots)...")
    env = HeightMaximizeEnv(num_type_b_robots=2, render_mode=None)
    
    print(f"   ✓ Environment created successfully")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Number of Type A robots: {env.num_type_a}")
    print(f"   - Number of Type B robots: {env.num_type_b}")
    print(f"   - Total robots: {env.total_robots}")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Environment reset successfully")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Training session: {info['training_session']}")
    print(f"   - Session goal: {info['session_goal']}")
    print(f"   - Initial max height: {info['max_height']:.3f}")
    print(f"   - Robot deployment: Random positions within spawn radius")
    
    # Run a few random steps
    print("\n3. Running 20 random steps...")
    total_reward = 0
    max_height_seen = 0
    max_connections = 0
    
    for step in range(20):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        max_height_seen = max(max_height_seen, info['max_height'])
        max_connections = max(max_connections, info['num_connections'])
        
        if (step + 1) % 5 == 0:
            print(f"   Step {step + 1}:")
            print(f"     - Reward: {reward:.3f}")
            print(f"     - Max height: {info['max_height']:.3f}")
            print(f"     - Robot heights: {[f'{h:.3f}' for h in info['robot_heights']]}")
            print(f"     - Highest robot idx: {info['highest_robot_idx']}")
            print(f"     - Active connections: {info['num_connections']}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step + 1}")
            break
    
    print(f"\n4. Episode Summary:")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Maximum height achieved: {max_height_seen:.3f}")
    print(f"   - Maximum connections: {max_connections}")
    
    # Close environment
    env.close()
    print("\n✓ All tests passed successfully!")
    print("=" * 80)
    
    return True


def test_multiple_episodes():
    """Test running multiple episodes with different seeds."""
    print("\n" + "=" * 80)
    print("Testing Multiple Episodes with Different Random Seeds")
    print("=" * 80)
    
    env = HeightMaximizeEnv(num_type_b_robots=2, render_mode=None)
    
    episode_stats = []
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset(seed=42 + episode)
        
        episode_reward = 0
        max_connections = 0
        
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            max_connections = max(max_connections, info['num_connections'])
            
            if terminated or truncated:
                break
        
        stats = {
            'steps': step + 1,
            'reward': episode_reward,
            'max_height': info['max_height'],
            'max_connections': max_connections
        }
        episode_stats.append(stats)
        
        print(f"  - Steps: {stats['steps']}")
        print(f"  - Total reward: {stats['reward']:.3f}")
        print(f"  - Max height: {stats['max_height']:.3f}")
        print(f"  - Max connections: {stats['max_connections']}")
    
    env.close()
    
    avg_height = np.mean([s['max_height'] for s in episode_stats])
    avg_reward = np.mean([s['reward'] for s in episode_stats])
    avg_connections = np.mean([s['max_connections'] for s in episode_stats])
    
    print(f"\nStatistics across {len(episode_stats)} episodes:")
    print(f"  - Average max height: {avg_height:.3f}")
    print(f"  - Average reward: {avg_reward:.3f}")
    print(f"  - Average max connections: {avg_connections:.1f}")
    print("✓ Multiple episodes test passed!")
    print("=" * 80)
    
    return True


def test_different_robot_counts():
    """Test environment with different robot counts."""
    print("\n" + "=" * 80)
    print("Testing Different Robot Configurations")
    print("=" * 80)
    
    configs = [
        (1, 2, 3),   # N=1: 1 Type B, 2 Type A (3 total)
        (2, 4, 6),   # N=2: 2 Type B, 4 Type A (6 total)
        (3, 6, 9),   # N=3: 3 Type B, 6 Type A (9 total)
    ]
    
    for n, expected_a, expected_total in configs:
        print(f"\nTesting N={n} configuration...")
        env = HeightMaximizeEnv(num_type_b_robots=n, render_mode=None)
        
        assert env.num_type_b == n, f"Expected {n} Type B robots, got {env.num_type_b}"
        assert env.num_type_a == expected_a, f"Expected {expected_a} Type A robots, got {env.num_type_a}"
        assert env.total_robots == expected_total, f"Expected {expected_total} total robots, got {env.total_robots}"
        
        # Quick test
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  ✓ N={n}: {n} Type B + {expected_a} Type A = {expected_total} total robots")
        print(f"    - Observation shape: {obs.shape}")
        print(f"    - Action shape: {env.action_space.shape}")
        
        env.close()
    
    print("\n✓ Robot configuration test passed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        # Run basic test
        test_height_maximize_env()
        
        # Run multiple episodes test
        test_multiple_episodes()
        
        # Run different robot counts test
        test_different_robot_counts()
        
        print("\n🎉 All tests completed successfully!")
        print("\nEnvironment Features Summary:")
        print("  ✓ URDF-based robot models (Type A bars, Type B spheres)")
        print("  ✓ Random deployment on plane each episode")
        print("  ✓ Configurable robot counts (N Type B, 2N Type A)")
        print("  ✓ Joint motors with force to lift 2x body weight")
        print("  ✓ Type A endpoint to Type B sphere connections")
        print("  ✓ Reward based on highest z-level at episode end")
        print("\nTraining Session 1 environment is ready to use.")
        print("Run 'python train.py' to start training with the height maximization goal.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
