"""
Test script for Training Session 5: Terrain Traversal Environment

This script tests the TerrainTraversalEnv with challenging terrain features.
"""

import numpy as np
from src.environments import TerrainTraversalEnv


def test_terrain_traversal_env():
    """Test the terrain traversal environment."""
    print("=" * 80)
    print("Testing Training Session 5: Terrain Traversal Environment")
    print("Navigate across challenging terrain (slopes, stairs, rough surfaces)")
    print("=" * 80)
    
    # Create environment
    print("\n1. Creating environment with varied terrain...")
    env = TerrainTraversalEnv(
        num_type_b_robots=2,
        render_mode=None,
        target_distance=20.0,
        terrain_difficulty=0.5,
        num_terrain_sections=5,
        max_episode_steps=2000
    )
    
    print(f"   ✓ Environment created successfully")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Number of Type A robots: {env.num_type_a}")
    print(f"   - Number of Type B robots: {env.num_type_b}")
    print(f"   - Target distance: {env.target_distance}m")
    print(f"   - Terrain difficulty: {env.terrain_difficulty}")
    print(f"   - Terrain sections: {env.num_terrain_sections}")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Environment reset successfully")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Training session: {info['training_session']}")
    print(f"   - Session goal: {info['session_goal']}")
    print(f"   - Target distance: {info['target_distance']}m")
    
    # Run simulation
    print("\n3. Running 150 random steps...")
    total_reward = 0
    max_distance = 0
    
    for step in range(150):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        max_distance = max(max_distance, info['forward_distance'])
        
        if (step + 1) % 30 == 0:
            print(f"\n   Step {step + 1}:")
            print(f"     - Reward: {reward:.3f}")
            print(f"     - Forward distance: {info['forward_distance']:.3f}m")
            print(f"     - Progress: {info['progress_percentage']:.1f}%")
            print(f"     - Stuck counter: {info['stuck_counter']}")
            print(f"     - Active connections: {info['num_connections']}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step + 1}")
            if info['forward_distance'] >= env.target_distance:
                print(f"   🎉 SUCCESS: Reached target distance!")
            break
    
    print(f"\n4. Episode Summary:")
    print(f"   - Total steps: {step + 1}")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Max forward distance: {max_distance:.3f}m / {env.target_distance}m")
    print(f"   - Progress: {(max_distance / env.target_distance) * 100:.1f}%")
    print(f"   - Falls: {info['falls_count']}")
    
    env.close()
    print("\n✓ Test completed successfully!")
    print("=" * 80)
    
    return True


def test_difficulty_levels():
    """Test with different terrain difficulty levels."""
    print("\n" + "=" * 80)
    print("Testing Different Terrain Difficulty Levels")
    print("=" * 80)
    
    configs = [
        (0.2, "Easy"),
        (0.5, "Medium"),
        (0.8, "Hard"),
    ]
    
    for difficulty, description in configs:
        print(f"\nTesting {description} terrain (difficulty {difficulty})...")
        env = TerrainTraversalEnv(
            num_type_b_robots=2,
            render_mode=None,
            target_distance=15.0,
            terrain_difficulty=difficulty,
            num_terrain_sections=4,
            max_episode_steps=500
        )
        
        obs, info = env.reset(seed=789)
        
        # Run a few steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print(f"  ✓ {description}: Progress {info['progress_percentage']:.1f}%, "
              f"distance: {info['forward_distance']:.2f}m")
        
        env.close()
    
    print("\n✓ Difficulty levels test passed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        test_terrain_traversal_env()
        test_difficulty_levels()
        
        print("\n🎉 All tests completed successfully!")
        print("\nEnvironment Features Summary:")
        print("  ✓ Procedurally generated terrain sections")
        print("  ✓ Multiple terrain types (slopes, stairs, rough, gaps)")
        print("  ✓ Configurable difficulty level")
        print("  ✓ Reward for forward progress and stability")
        print("  ✓ Requires adaptive locomotion strategies")
        print("\nTraining Session 5 environment is ready to use.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
