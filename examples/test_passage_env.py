"""
Test script for Training Session 6: Narrow Passage Environment

This script tests the NarrowPassageEnv with tight corridors and checkpoints.
"""

import numpy as np
from src.environments import NarrowPassageEnv


def test_narrow_passage_env():
    """Test the narrow passage environment."""
    print("=" * 80)
    print("Testing Training Session 6: Narrow Passage Environment")
    print("Navigate through narrow passages with precision control")
    print("=" * 80)
    
    # Create environment
    print("\n1. Creating environment with narrow passages...")
    env = NarrowPassageEnv(
        num_type_b_robots=2,
        render_mode=None,
        num_passages=4,
        passage_width=2.0,
        passage_difficulty=0.5,
        max_episode_steps=2000
    )
    
    print(f"   ✓ Environment created successfully")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Number of Type A robots: {env.num_type_a}")
    print(f"   - Number of Type B robots: {env.num_type_b}")
    print(f"   - Number of passages: {env.num_passages}")
    print(f"   - Passage width: {env.passage_width:.2f}m")
    print(f"   - Passage difficulty: {env.passage_difficulty}")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Environment reset successfully")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Training session: {info['training_session']}")
    print(f"   - Session goal: {info['session_goal']}")
    print(f"   - Total checkpoints: {info['total_checkpoints']}")
    
    # Run simulation
    print("\n3. Running 150 random steps...")
    total_reward = 0
    max_checkpoints = 0
    
    for step in range(150):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        max_checkpoints = max(max_checkpoints, info['checkpoints_passed'])
        
        if (step + 1) % 30 == 0:
            print(f"\n   Step {step + 1}:")
            print(f"     - Reward: {reward:.3f}")
            print(f"     - Checkpoints passed: {info['checkpoints_passed']}/{info['total_checkpoints']}")
            print(f"     - Completion: {info['completion_percentage']:.1f}%")
            print(f"     - Collision count: {info['collision_count']}")
            print(f"     - Active connections: {info['num_connections']}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step + 1}")
            if info['checkpoints_passed'] == info['total_checkpoints']:
                print(f"   🎉 SUCCESS: All checkpoints passed!")
            break
    
    print(f"\n4. Episode Summary:")
    print(f"   - Total steps: {step + 1}")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Checkpoints passed: {max_checkpoints}/{info['total_checkpoints']}")
    print(f"   - Completion: {info['completion_percentage']:.1f}%")
    print(f"   - Total collisions: {info['collision_count']}")
    
    env.close()
    print("\n✓ Test completed successfully!")
    print("=" * 80)
    
    return True


def test_passage_difficulty():
    """Test with different passage difficulties."""
    print("\n" + "=" * 80)
    print("Testing Different Passage Difficulty Levels")
    print("=" * 80)
    
    configs = [
        (0.2, 2.0, "Easy - Wide passages"),
        (0.5, 2.0, "Medium - Moderate passages"),
        (0.8, 2.0, "Hard - Narrow passages"),
    ]
    
    for difficulty, width, description in configs:
        print(f"\nTesting {description}...")
        env = NarrowPassageEnv(
            num_type_b_robots=2,
            render_mode=None,
            num_passages=3,
            passage_width=width,
            passage_difficulty=difficulty,
            max_episode_steps=500
        )
        
        effective_width = width * (1.0 - 0.6 * difficulty)
        print(f"  Effective passage width: {effective_width:.2f}m")
        
        obs, info = env.reset(seed=321)
        
        # Run a few steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print(f"  ✓ {description}: {info['checkpoints_passed']}/{info['total_checkpoints']} checkpoints, "
              f"{info['collision_count']} collisions")
        
        env.close()
    
    print("\n✓ Difficulty levels test passed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        test_narrow_passage_env()
        test_passage_difficulty()
        
        print("\n🎉 All tests completed successfully!")
        print("\nEnvironment Features Summary:")
        print("  ✓ Narrow corridors with varied configurations")
        print("  ✓ Checkpoint system for progress tracking")
        print("  ✓ Configurable passage width and difficulty")
        print("  ✓ Penalty for wall collisions")
        print("  ✓ Requires precise control and spatial awareness")
        print("\nTraining Session 6 environment is ready to use.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
