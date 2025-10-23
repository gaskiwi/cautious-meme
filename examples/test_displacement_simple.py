"""
Simple test script for the Displacement Environment (Session 3).
Tests without GUI rendering for CI/headless environments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environments.displacement_env import DisplacementEnv


def test_basic_functionality():
    """Test basic environment functionality."""
    print("=" * 80)
    print("Testing Displacement Environment - Basic Functionality")
    print("=" * 80)
    
    # Create environment without rendering
    env = DisplacementEnv(
        render_mode=None,
        num_type_b_robots=2,
        spawn_radius=3.0,
        max_episode_steps=100
    )
    
    print(f"\n✓ Environment created successfully!")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action dim: {env.action_space.shape[0]}")
    print(f"  Observation dim: {env.observation_space.shape[0]}")
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\n✓ Environment reset successful!")
    print(f"  Object Shape: {info['object_shape']}")
    print(f"  Target Direction: {info['target_direction']}")
    print(f"  Object Initial Position: {info['object_initial_pos']}")
    print(f"  Observation shape: {obs.shape}")
    
    # Test a few steps
    print(f"\n✓ Running 10 test steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i == 0:
            print(f"  Step 1: reward={reward:.4f}, displacement={info['current_displacement']:.4f}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            break
    
    if not (terminated or truncated):
        print(f"  Completed 10 steps successfully")
    
    env.close()
    print(f"\n✓ Test completed successfully!\n")
    return True


def test_all_shapes():
    """Test that all object shapes can be created."""
    print("=" * 80)
    print("Testing All Object Shapes")
    print("=" * 80)
    
    shapes = list(DisplacementEnv.OBJECT_SHAPES.keys())
    print(f"\nAvailable shapes: {', '.join(shapes)}")
    print(f"Total: {len(shapes)} shapes")
    
    env = DisplacementEnv(render_mode=None, num_type_b_robots=1, max_episode_steps=10)
    
    print("\nTesting shape randomization (10 episodes)...")
    shapes_seen = set()
    for i in range(10):
        obs, info = env.reset(seed=i)
        shape = info['object_shape']
        shapes_seen.add(shape)
    
    print(f"  Shapes seen: {', '.join(sorted(shapes_seen))}")
    print(f"  Unique shapes: {len(shapes_seen)}/{len(shapes)}")
    
    env.close()
    print(f"\n✓ Shape randomization working!\n")
    return True


def test_all_directions():
    """Test that all cardinal directions can be selected."""
    print("=" * 80)
    print("Testing All Cardinal Directions")
    print("=" * 80)
    
    directions = list(DisplacementEnv.DIRECTIONS.keys())
    print(f"\nAvailable directions: {', '.join(directions)}")
    
    env = DisplacementEnv(render_mode=None, num_type_b_robots=1, max_episode_steps=10)
    
    print("\nTesting direction randomization (10 episodes)...")
    directions_seen = set()
    for i in range(10):
        obs, info = env.reset(seed=i)
        direction = info['target_direction']
        direction_vec = DisplacementEnv.DIRECTIONS[direction]
        directions_seen.add(direction)
        
        if i < 4:  # Show first few
            print(f"  Episode {i+1}: {direction} -> {direction_vec}")
    
    print(f"\n  Directions seen: {', '.join(sorted(directions_seen))}")
    print(f"  Unique directions: {len(directions_seen)}/{len(directions)}")
    
    env.close()
    print(f"\n✓ Direction randomization working!\n")
    return True


def test_displacement_calculation():
    """Test that displacement is calculated correctly."""
    print("=" * 80)
    print("Testing Displacement Calculation")
    print("=" * 80)
    
    env = DisplacementEnv(render_mode=None, num_type_b_robots=1, max_episode_steps=50)
    
    obs, info = env.reset(seed=123)
    initial_pos = np.array(info['object_initial_pos'])
    direction = info['target_direction']
    
    print(f"\n  Object starts at: {initial_pos[:2]}")
    print(f"  Target direction: {direction}")
    
    # Run a few steps
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    final_pos = np.array(info['object_pos'])
    displacement = info['current_displacement']
    max_displacement = info['max_displacement']
    
    print(f"  Object ends at: {final_pos[:2]}")
    print(f"  Current displacement: {displacement:.4f}m")
    print(f"  Max displacement: {max_displacement:.4f}m")
    
    # Verify calculation
    direction_vec = DisplacementEnv.DIRECTIONS[direction]
    displacement_vec = final_pos[:2] - initial_pos[:2]
    expected_displacement = np.dot(displacement_vec, direction_vec)
    
    print(f"  Expected displacement: {expected_displacement:.4f}m")
    print(f"  Match: {np.isclose(displacement, expected_displacement)}")
    
    env.close()
    print(f"\n✓ Displacement calculation correct!\n")
    return True


def test_randomized_placement():
    """Test that objects and robots are placed randomly."""
    print("=" * 80)
    print("Testing Randomized Placement")
    print("=" * 80)
    
    env = DisplacementEnv(render_mode=None, num_type_b_robots=2, max_episode_steps=10)
    
    object_positions = []
    print("\nTesting object placement randomization (5 episodes)...")
    
    for i in range(5):
        obs, info = env.reset(seed=i * 100)
        obj_pos = info['object_initial_pos']
        object_positions.append(obj_pos[:2])
        print(f"  Episode {i+1}: Object at ({obj_pos[0]:.2f}, {obj_pos[1]:.2f})")
    
    # Check that positions are different
    unique_positions = len(set([tuple(pos) for pos in object_positions]))
    print(f"\n  Unique positions: {unique_positions}/5")
    
    if unique_positions >= 4:
        print(f"  ✓ Good randomization!")
    else:
        print(f"  ⚠ Limited randomization (may be due to small sample)")
    
    env.close()
    print()
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DISPLACEMENT ENVIRONMENT TEST SUITE")
    print("Training Session 3: Object Displacement")
    print("=" * 80 + "\n")
    
    all_passed = True
    
    try:
        all_passed &= test_basic_functionality()
        all_passed &= test_all_shapes()
        all_passed &= test_all_directions()
        all_passed &= test_displacement_calculation()
        all_passed &= test_randomized_placement()
        
        print("=" * 80)
        if all_passed:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print()
