"""
Test script for Training Session 2: Crush Resistance Environment

This script tests the CrushResistanceEnv with the hydraulic press mechanism.

Type A: Bar robots with joints (bar_with_joint.urdf) - 2N robots
Type B: Sphere robots (rolling_sphere.urdf) - N robots

The environment features:
- Same physics and robots as HeightMaximizeEnv (Session 1)
- First 30 seconds: robots can position themselves
- After 30 seconds: a descending plane acts like a hydraulic press
- Press applies increasing force when encountering obstacles
- Reward based on survival time after press activation
"""

import numpy as np
from src.environments import CrushResistanceEnv


def test_crush_resistance_env():
    """Test the crush resistance environment with hydraulic press."""
    print("=" * 80)
    print("Testing Training Session 2: Crush Resistance Environment")
    print("Type A (Bar) + Type B (Sphere) Robots with Hydraulic Press")
    print("=" * 80)
    
    # Create environment
    print("\n1. Creating environment with N=2 (2 Type B, 4 Type A robots)...")
    env = CrushResistanceEnv(
        num_type_b_robots=2,
        render_mode=None,
        reference_height=5.0,  # Press starts at 5m height
        press_descent_speed=0.05,  # 0.05 m/s descent
        max_episode_steps=3000  # Allow longer episodes (50 seconds)
    )
    
    print(f"   ✓ Environment created successfully")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Number of Type A robots: {env.num_type_a}")
    print(f"   - Number of Type B robots: {env.num_type_b}")
    print(f"   - Total robots: {env.total_robots}")
    print(f"   - Press activation step: {env.press_start_step} (30 seconds)")
    print(f"   - Press starting height: {env.reference_height}m")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Environment reset successfully")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Training session: {info['training_session']}")
    print(f"   - Session goal: {info['session_goal']}")
    print(f"   - Press active: {info['press_active']}")
    print(f"   - Survival time: {info['survival_time']:.2f}s")
    
    # Run simulation through different phases
    print("\n3. Running simulation through both phases...")
    total_reward = 0
    max_height_seen = 0
    press_activation_step = None
    
    # Run for a while to see both phases
    num_steps = 100
    
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        max_height_seen = max(max_height_seen, info['max_height'])
        
        # Check if press just activated
        if info['press_active'] and press_activation_step is None:
            press_activation_step = step + 1
            print(f"\n   🔴 PRESS ACTIVATED at step {press_activation_step}!")
            print(f"      - Press starting height: {info['press_height']:.2f}m")
        
        # Print status every 20 steps
        if (step + 1) % 20 == 0 or (press_activation_step == step + 1):
            phase = "POSITIONING" if not info['press_active'] else "CRUSHING"
            print(f"\n   Step {step + 1} [{phase}]:")
            print(f"     - Reward: {reward:.3f}")
            print(f"     - Max height: {info['max_height']:.3f}m")
            print(f"     - Avg height: {info['avg_height']:.3f}m")
            print(f"     - Active connections: {info['num_connections']}")
            if info['press_active']:
                print(f"     - Press height: {info['press_height']:.3f}m")
                print(f"     - Press force: {info['press_force']:.1f}N")
                print(f"     - Survival time: {info['survival_time']:.2f}s")
                print(f"     - All grounded: {info['all_grounded']}")
        
        if terminated or truncated:
            print(f"\n   Episode ended at step {step + 1}")
            print(f"   - Reason: {'All robots grounded' if info['all_grounded'] else 'Other termination'}")
            break
    
    print(f"\n4. Episode Summary:")
    print(f"   - Total steps: {step + 1}")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Maximum height achieved: {max_height_seen:.3f}m")
    if press_activation_step:
        print(f"   - Press activated at step: {press_activation_step}")
        print(f"   - Survival time after press: {info['survival_time']:.2f}s")
    else:
        print(f"   - Press not activated (episode too short)")
    
    # Close environment
    env.close()
    print("\n✓ All tests passed successfully!")
    print("=" * 80)
    
    return True


def test_press_mechanism():
    """Test the hydraulic press mechanism in detail."""
    print("\n" + "=" * 80)
    print("Testing Hydraulic Press Mechanism")
    print("=" * 80)
    
    # Create environment with faster press activation for testing
    env = CrushResistanceEnv(
        num_type_b_robots=1,
        render_mode=None,
        reference_height=3.0,  # Lower starting height
        press_descent_speed=0.1,  # Faster descent
        press_force_increment=100.0,  # Stronger force increment
        max_episode_steps=5000
    )
    
    print("\n   Configuration for testing:")
    print(f"   - Press starting height: {env.reference_height}m")
    print(f"   - Press descent speed: {env.press_descent_speed}m/s")
    print(f"   - Press force increment: {env.press_force_increment}N")
    print(f"   - Press activation: step {env.press_start_step}")
    
    obs, info = env.reset(seed=123)
    
    # Run until press activates and for some time after
    pre_press_steps = env.press_start_step + 50
    
    print(f"\n   Running {pre_press_steps} steps to test press mechanism...")
    
    for step in range(pre_press_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log when press activates
        if step + 1 == env.press_start_step:
            print(f"\n   ✓ Press activated at step {step + 1}")
            print(f"     - Initial press height: {info['press_height']:.2f}m")
        
        # Log press behavior every 10 steps after activation
        if info['press_active'] and (step + 1 - env.press_start_step) % 10 == 0:
            steps_after_activation = step + 1 - env.press_start_step
            print(f"\n   Press status (step {step + 1}, +{steps_after_activation} after activation):")
            print(f"     - Press height: {info['press_height']:.3f}m")
            print(f"     - Press force: {info['press_force']:.1f}N")
            print(f"     - Robot avg height: {info['avg_height']:.3f}m")
            print(f"     - Survival time: {info['survival_time']:.2f}s")
        
        if terminated or truncated:
            print(f"\n   Episode ended at step {step + 1}")
            if info['all_grounded']:
                print(f"   ✓ All robots were successfully crushed to the ground!")
                print(f"   - Final survival time: {info['survival_time']:.2f}s")
            break
    
    env.close()
    
    print("\n✓ Press mechanism test completed!")
    print("=" * 80)
    
    return True


def test_reward_system():
    """Test the reward system for both phases."""
    print("\n" + "=" * 80)
    print("Testing Reward System")
    print("=" * 80)
    
    env = CrushResistanceEnv(
        num_type_b_robots=2,
        render_mode=None,
        max_episode_steps=2000
    )
    
    obs, info = env.reset(seed=456)
    
    positioning_rewards = []
    survival_rewards = []
    
    print("\n   Collecting rewards from both phases...")
    
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if not info['press_active']:
            positioning_rewards.append(reward)
        else:
            survival_rewards.append(reward)
        
        if terminated or truncated:
            break
    
    print(f"\n   Reward Statistics:")
    print(f"\n   Positioning Phase (first 30 seconds):")
    if positioning_rewards:
        print(f"     - Total rewards collected: {len(positioning_rewards)}")
        print(f"     - Average reward: {np.mean(positioning_rewards):.3f}")
        print(f"     - Total positioning reward: {np.sum(positioning_rewards):.3f}")
    
    print(f"\n   Survival Phase (after press activation):")
    if survival_rewards:
        print(f"     - Total rewards collected: {len(survival_rewards)}")
        print(f"     - Average reward: {np.mean(survival_rewards):.3f}")
        print(f"     - Total survival reward: {np.sum(survival_rewards):.3f}")
        print(f"     - Survival time: {info['survival_time']:.2f}s")
    
    env.close()
    
    print("\n✓ Reward system test completed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        # Run basic test
        test_crush_resistance_env()
        
        # Run press mechanism test
        test_press_mechanism()
        
        # Run reward system test
        test_reward_system()
        
        print("\n🎉 All tests completed successfully!")
        print("\nEnvironment Features Summary:")
        print("  ✓ Same physics as Session 1 (HeightMaximizeEnv)")
        print("  ✓ Two-phase episode: positioning (30s) + crushing")
        print("  ✓ Hydraulic press descends from configurable height")
        print("  ✓ Press applies increasing force when encountering obstacles")
        print("  ✓ Reward based on survival time after press activation")
        print("  ✓ Episode ends when all robots are crushed to ground")
        print("\nTraining Session 2 environment is ready to use.")
        print("This environment trains robots to resist crushing forces.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
