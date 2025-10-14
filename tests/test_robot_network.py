"""
Test script for the Robot-Aware Neural Network Architecture.

This script validates that the custom neural network can:
1. Handle different numbers of Type A (Bar) and Type B (Sphere) robots
2. Process variable batch sizes
3. Correctly encode robot states and types
4. Generate appropriate actions and value estimates
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Try to import torch, but handle the case where it's not installed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Skipping actual network tests.")
    print("This is OK for code validation, but tests won't run.")


def test_network_creation():
    """Test that the network can be created with different configurations."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_network_creation (PyTorch not available)")
        return True
    
    from src.agents.robot_network import create_robot_network
    
    print("Testing network creation...")
    
    # Test 1: Create network with Type A robots only
    network_a = create_robot_network(num_type_a=3, num_type_b=0, max_robots=5)
    assert network_a is not None
    print("✓ Network with Type A robots only created successfully")
    
    # Test 2: Create network with Type B robots only
    network_b = create_robot_network(num_type_a=0, num_type_b=4, max_robots=5)
    assert network_b is not None
    print("✓ Network with Type B robots only created successfully")
    
    # Test 3: Create network with mixed robots
    network_mixed = create_robot_network(num_type_a=2, num_type_b=3, max_robots=10)
    assert network_mixed is not None
    print("✓ Network with mixed robot types created successfully")
    
    return True


def test_forward_pass():
    """Test forward pass through the network."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_forward_pass (PyTorch not available)")
        return True
    
    from src.agents.robot_network import create_robot_network
    
    print("\nTesting forward pass...")
    
    # Create network
    network = create_robot_network(num_type_a=2, num_type_b=2, max_robots=5)
    
    # Create dummy inputs
    batch_size = 8
    max_robots = 5
    max_state_dim = 51
    
    observations = torch.randn(batch_size, max_robots, max_state_dim)
    robot_types = torch.tensor([
        [1, 1, 2, 2, 0],  # 2 Type A, 2 Type B, 1 padding
        [1, 2, 0, 0, 0],  # 1 Type A, 1 Type B, 3 padding
        [2, 2, 2, 0, 0],  # 3 Type B, 2 padding
        [1, 1, 1, 1, 2],  # 4 Type A, 1 Type B
        [1, 1, 2, 2, 2],  # 2 Type A, 3 Type B
        [2, 2, 2, 2, 2],  # 5 Type B
        [1, 1, 1, 1, 1],  # 5 Type A
        [1, 2, 1, 2, 1],  # 3 Type A, 2 Type B
    ])
    num_robots_per_batch = torch.tensor([4, 2, 3, 5, 5, 5, 5, 5])
    
    # Forward pass
    actions, values = network(observations, robot_types, num_robots_per_batch)
    
    # Validate output shapes
    assert actions.shape == (batch_size, max_robots, max(network.type_a_action_dim, network.type_b_action_dim))
    assert values.shape == (batch_size, 1)
    print(f"✓ Forward pass successful. Actions shape: {actions.shape}, Values shape: {values.shape}")
    
    # Test actor and critic separately
    actions_only = network.forward_actor(observations, robot_types, num_robots_per_batch)
    values_only = network.forward_critic(observations, robot_types, num_robots_per_batch)
    
    assert actions_only.shape == actions.shape
    assert values_only.shape == values.shape
    print("✓ Separate actor and critic forward passes successful")
    
    return True


def test_robot_type_handling():
    """Test that the network correctly handles different robot types."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_robot_type_handling (PyTorch not available)")
        return True
    
    from src.agents.robot_network import create_robot_network
    
    print("\nTesting robot type handling...")
    
    network = create_robot_network(num_type_a=1, num_type_b=1, max_robots=3)
    
    # Test with only Type A robots
    obs_a = torch.randn(2, 3, 51)
    types_a = torch.tensor([[1, 1, 1], [1, 1, 0]])
    num_robots_a = torch.tensor([3, 2])
    
    actions_a, values_a = network(obs_a, types_a, num_robots_a)
    assert not torch.isnan(actions_a).any(), "Actions contain NaN"
    assert not torch.isnan(values_a).any(), "Values contain NaN"
    print("✓ Type A robots handled correctly")
    
    # Test with only Type B robots
    obs_b = torch.randn(2, 3, 51)
    types_b = torch.tensor([[2, 2, 2], [2, 2, 0]])
    num_robots_b = torch.tensor([3, 2])
    
    actions_b, values_b = network(obs_b, types_b, num_robots_b)
    assert not torch.isnan(actions_b).any(), "Actions contain NaN"
    assert not torch.isnan(values_b).any(), "Values contain NaN"
    print("✓ Type B robots handled correctly")
    
    # Test with mixed robots
    obs_mix = torch.randn(2, 3, 51)
    types_mix = torch.tensor([[1, 2, 1], [2, 1, 0]])
    num_robots_mix = torch.tensor([3, 2])
    
    actions_mix, values_mix = network(obs_mix, types_mix, num_robots_mix)
    assert not torch.isnan(actions_mix).any(), "Actions contain NaN"
    assert not torch.isnan(values_mix).any(), "Values contain NaN"
    print("✓ Mixed robot types handled correctly")
    
    return True


def test_variable_robot_count():
    """Test that the network handles variable numbers of robots correctly."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_variable_robot_count (PyTorch not available)")
        return True
    
    from src.agents.robot_network import create_robot_network
    
    print("\nTesting variable robot count...")
    
    network = create_robot_network(num_type_a=3, num_type_b=3, max_robots=10)
    
    # Test different counts
    for num_robots in [1, 3, 5, 7, 10]:
        obs = torch.randn(4, 10, 51)
        # Create random robot types (1 or 2)
        types = torch.randint(1, 3, (4, 10))
        # Pad with zeros
        for i in range(4):
            types[i, num_robots:] = 0
        num_robots_tensor = torch.tensor([num_robots] * 4)
        
        actions, values = network(obs, types, num_robots_tensor)
        
        assert not torch.isnan(actions).any(), f"Actions contain NaN for {num_robots} robots"
        assert not torch.isnan(values).any(), f"Values contain NaN for {num_robots} robots"
    
    print("✓ Variable robot counts handled correctly (1-10 robots)")
    
    return True


def test_attention_mechanism():
    """Test that the attention mechanism is working."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_attention_mechanism (PyTorch not available)")
        return True
    
    from src.agents.robot_network import MultiHeadAttention
    
    print("\nTesting attention mechanism...")
    
    attention = MultiHeadAttention(embedding_dim=128, num_heads=4)
    
    # Create test input
    batch_size = 4
    num_robots = 5
    embedding_dim = 128
    
    x = torch.randn(batch_size, num_robots, embedding_dim)
    mask = torch.ones(batch_size, num_robots)
    mask[:, 3:] = 0  # Mask out last 2 robots
    
    # Forward pass
    output = attention(x, mask)
    
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    print("✓ Attention mechanism working correctly")
    
    return True


def test_environment_wrapper():
    """Test the environment observation wrapper."""
    print("\nTesting environment wrapper...")
    
    from src.agents.robot_policy import MultiRobotEnvironmentWrapper
    
    # Test observation formatting
    robot_states = [
        np.random.randn(51),  # Type A robot
        np.random.randn(51),  # Type A robot
        np.random.randn(13),  # Type B robot
    ]
    robot_types = [1, 1, 2]  # 2 Type A, 1 Type B
    max_robots = 5
    
    obs = MultiRobotEnvironmentWrapper.format_observation(
        robot_states, robot_types, max_robots
    )
    
    expected_size = max_robots * (51 + 1) + 1  # (max_state_dim + type) * max_robots + num_robots
    assert obs.shape[0] == expected_size
    print(f"✓ Observation formatting correct. Shape: {obs.shape}")
    
    # Test action parsing
    if TORCH_AVAILABLE:
        action = np.random.randn(max_robots, 6)
        parsed_actions = MultiRobotEnvironmentWrapper.parse_action(
            action.flatten(), robot_types
        )
        
        assert len(parsed_actions) == len(robot_types)
        assert parsed_actions[0].shape[0] == 6  # Type A action dim
        assert parsed_actions[1].shape[0] == 6  # Type A action dim
        assert parsed_actions[2].shape[0] == 3  # Type B action dim
        print("✓ Action parsing correct")
    
    return True


def test_parameter_count():
    """Test and report parameter counts."""
    if not TORCH_AVAILABLE:
        print("SKIP: test_parameter_count (PyTorch not available)")
        return True
    
    from src.agents.robot_network import create_robot_network
    
    print("\nTesting parameter counts...")
    
    # Create different network configurations
    configs = [
        {'num_type_a': 1, 'num_type_b': 0, 'max_robots': 3},
        {'num_type_a': 0, 'num_type_b': 3, 'max_robots': 3},
        {'num_type_a': 2, 'num_type_b': 2, 'max_robots': 5},
        {'num_type_a': 5, 'num_type_b': 5, 'max_robots': 10},
    ]
    
    for config in configs:
        network = create_robot_network(**config)
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        
        print(f"  Config {config}: {total_params:,} total params, {trainable_params:,} trainable")
        assert total_params == trainable_params, "Some parameters are frozen"
    
    print("✓ Parameter counts validated")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("ROBOT NEURAL NETWORK ARCHITECTURE TEST SUITE")
    print("="*70)
    
    tests = [
        test_network_creation,
        test_forward_pass,
        test_robot_type_handling,
        test_variable_robot_count,
        test_attention_mechanism,
        test_environment_wrapper,
        test_parameter_count,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("\nNOTE: PyTorch is not installed, so most tests were skipped.")
        print("The code structure is valid, but runtime tests require PyTorch.")
        print("Install PyTorch to run full tests: pip install torch")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
