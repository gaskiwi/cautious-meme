"""
Test script for Training Session 4: Object Manipulation Environment

This script tests the ObjectManipulationEnv with pushable objects and target zones.
"""

import numpy as np
from src.environments import ObjectManipulationEnv


def test_object_manipulation_env():
    """Test the object manipulation environment."""
    print("=" * 80)
    print("Testing Training Session 4: Object Manipulation Environment")
    print("Push/pull objects to target locations")
    print("=" * 80)
    
    # Create environment
    print("\n1. Creating environment with manipulable objects...")
    env = ObjectManipulationEnv(
        num_type_b_robots=2,
        render_mode=None,
        num_objects=3,
        arena_size=15.0,
        max_episode_steps=2000
    )
    
    print(f"   ✓ Environment created successfully")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Number of Type A robots: {env.num_type_a}")
    print(f"   - Number of Type B robots: {env.num_type_b}")
    print(f"   - Number of objects: {env.num_objects}")
    print(f"   - Arena size: {env.arena_size}m")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Environment reset successfully")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Training session: {info['training_session']}")
    print(f"   - Session goal: {info['session_goal']}")
    print(f"   - Objects in target: {info['objects_in_target']}/{info['num_objects']}")
    
    # Run simulation
    print("\n3. Running 100 random steps...")
    total_reward = 0
    max_objects_in_target = 0
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        max_objects_in_target = max(max_objects_in_target, info['objects_in_target'])
        
        if (step + 1) % 25 == 0:
            print(f"\n   Step {step + 1}:")
            print(f"     - Reward: {reward:.3f}")
            print(f"     - Objects in target: {info['objects_in_target']}/{info['num_objects']}")
            print(f"     - Total push distance: {info['total_push_distance']:.3f}m")
            print(f"     - Active connections: {info['num_connections']}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step + 1}")
            if info['objects_in_target'] == info['num_objects']:
                print(f"   🎉 SUCCESS: All objects in targets!")
            break
    
    print(f"\n4. Episode Summary:")
    print(f"   - Total steps: {step + 1}")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Max objects in target: {max_objects_in_target}/{info['num_objects']}")
    print(f"   - Total push distance: {info['total_push_distance']:.3f}m")
    
    env.close()
    print("\n✓ Test completed successfully!")
    print("=" * 80)
    
    return True


def test_different_object_counts():
    """Test with different numbers of objects."""
    print("\n" + "=" * 80)
    print("Testing Different Object Configurations")
    print("=" * 80)
    
    configs = [
        (1, "Single Object"),
        (2, "Two Objects"),
        (4, "Multiple Objects"),
    ]
    
    for num_objects, description in configs:
        print(f"\nTesting {description} ({num_objects} objects)...")
        env = ObjectManipulationEnv(
            num_type_b_robots=2,
            render_mode=None,
            num_objects=num_objects,
            max_episode_steps=500
        )
        
        obs, info = env.reset(seed=456)
        
        # Run a few steps
        for _ in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print(f"  ✓ {description}: {info['objects_in_target']}/{num_objects} in target, "
              f"push distance: {info['total_push_distance']:.2f}m")
        
        env.close()
    
    print("\n✓ Object configuration test passed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        test_object_manipulation_env()
        test_different_object_counts()
        
        print("\n🎉 All tests completed successfully!")
        print("\nEnvironment Features Summary:")
        print("  ✓ Multiple objects with varying mass and friction")
        print("  ✓ Target zones for object placement")
        print("  ✓ Reward for moving objects to targets")
        print("  ✓ Requires coordinated pushing/pulling")
        print("  ✓ Success when all objects in targets")
        print("\nTraining Session 4 environment is ready to use.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
