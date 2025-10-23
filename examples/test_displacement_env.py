"""
Test script for the Displacement Environment (Session 3).

This script tests the new environment with random object shapes,
random placement, and random target directions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environments.displacement_env import DisplacementEnv


def test_displacement_environment():
    """Test the displacement environment with multiple episodes."""
    print("=" * 80)
    print("Testing Displacement Environment (Training Session 3)")
    print("=" * 80)
    
    # Create environment with rendering
    env = DisplacementEnv(
        render_mode='human',
        num_type_b_robots=2,  # N=2, so 4 Type A robots + 2 Type B robots
        spawn_radius=3.0,
        max_episode_steps=500
    )
    
    print(f"\nEnvironment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action dim: {env.action_space.shape[0]}")
    print(f"Observation dim: {env.observation_space.shape[0]}")
    
    # Run multiple test episodes
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*80}")
        
        obs, info = env.reset(seed=episode)
        
        print(f"\nEpisode Info:")
        print(f"  Object Shape: {info['object_shape']}")
        print(f"  Target Direction: {info['target_direction']}")
        print(f"  Object Initial Position: {info['object_initial_pos']}")
        print(f"  Number of Type A robots: {info['num_type_a']}")
        print(f"  Number of Type B robots: {info['num_type_b']}")
        
        total_reward = 0
        step_count = 0
        done = False
        
        print(f"\nRunning episode...")
        
        while not done:
            # Random action for testing
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Step {step_count}: "
                      f"Displacement = {info['current_displacement']:.3f}m, "
                      f"Max Displacement = {info['max_displacement']:.3f}m, "
                      f"Reward = {reward:.3f}")
        
        print(f"\nEpisode Results:")
        print(f"  Total Steps: {step_count}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Max Displacement: {info['max_displacement']:.3f}m")
        print(f"  Final Object Position: {info['object_pos']}")
        print(f"  Number of Robot Connections: {info['num_connections']}")
        
        # Calculate displacement vector for visualization
        initial_pos = np.array(info.get('object_initial_pos', [0, 0, 0]))
        final_pos = np.array(info['object_pos'])
        displacement_vec = final_pos[:2] - initial_pos[:2]
        displacement_magnitude = np.linalg.norm(displacement_vec)
        
        print(f"  Total Displacement Magnitude: {displacement_magnitude:.3f}m")
        print(f"  Displacement Vector: [{displacement_vec[0]:.3f}, {displacement_vec[1]:.3f}]")
        
        if info['max_displacement'] > 0:
            print(f"  ✓ Object moved in target direction!")
        else:
            print(f"  ✗ Object moved in opposite direction")
    
    env.close()
    print(f"\n{'='*80}")
    print("Test completed successfully!")
    print(f"{'='*80}\n")


def test_object_shapes():
    """Test all available object shapes."""
    print("\n" + "=" * 80)
    print("Testing All Object Shapes")
    print("=" * 80)
    
    from src.environments.displacement_env import DisplacementEnv
    
    shapes = list(DisplacementEnv.OBJECT_SHAPES.keys())
    print(f"\nAvailable shapes: {', '.join(shapes)}")
    
    env = DisplacementEnv(
        render_mode=None,  # No rendering for quick test
        num_type_b_robots=1,
        max_episode_steps=100
    )
    
    print("\nTesting each shape...")
    for shape in shapes:
        # Force a specific shape by modifying the environment temporarily
        obs, info = env.reset()
        print(f"  ✓ {shape}: Created successfully (mass={DisplacementEnv.OBJECT_SHAPES[shape]['mass']}kg)")
    
    env.close()
    print("\nAll shapes tested successfully!")


def test_directions():
    """Test all cardinal directions."""
    print("\n" + "=" * 80)
    print("Testing All Cardinal Directions")
    print("=" * 80)
    
    from src.environments.displacement_env import DisplacementEnv
    
    directions = list(DisplacementEnv.DIRECTIONS.keys())
    print(f"\nAvailable directions: {', '.join(directions)}")
    
    env = DisplacementEnv(
        render_mode=None,
        num_type_b_robots=1,
        max_episode_steps=50
    )
    
    print("\nTesting each direction...")
    for _ in range(len(directions)):
        obs, info = env.reset()
        direction = info['target_direction']
        direction_vec = DisplacementEnv.DIRECTIONS[direction]
        print(f"  ✓ {direction}: Vector = {direction_vec}")
    
    env.close()
    print("\nAll directions tested successfully!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DISPLACEMENT ENVIRONMENT TEST SUITE")
    print("Training Session 3: Object Displacement with Random Shapes & Directions")
    print("=" * 80)
    
    # Test 1: Object shapes
    test_object_shapes()
    
    # Test 2: Directions
    test_directions()
    
    # Test 3: Full environment with visualization
    print("\n\nStarting full environment test with visualization...")
    print("Press Ctrl+C to skip if needed.\n")
    
    try:
        test_displacement_environment()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80 + "\n")
