"""Simple robot environment example using a cart-pole-like system."""

import gymnasium as gym
import numpy as np
import pybullet as p
from .base_robot_env import BaseRobotEnv


class SimpleRobotEnv(BaseRobotEnv):
    """
    Simple robot environment using a cart with a pole.
    This serves as an example that can be easily replaced with a custom robot.
    
    The goal is to balance the pole upright by applying forces to the cart.
    """
    
    def __init__(self, render_mode=None, **kwargs):
        """Initialize the simple robot environment."""
        super().__init__(render_mode=render_mode, **kwargs)
        
        # Define action space: force applied to cart [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )
        
        self.cart_id = None
        self.pole_id = None
        self.force_magnitude = 10.0
        
    def _load_robot(self) -> int:
        """
        Load the cart-pole robot into simulation.
        
        Returns:
            Cart body ID
        """
        # Create cart (box)
        cart_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.2, 0.2, 0.1]
        )
        cart_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.2, 0.2, 0.1],
            rgbaColor=[0.8, 0.2, 0.2, 1.0]
        )
        self.cart_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=cart_shape,
            baseVisualShapeIndex=cart_visual,
            basePosition=[0, 0, 0.1]
        )
        
        # Create pole (cylinder)
        pole_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.05,
            height=1.0
        )
        pole_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.05,
            length=1.0,
            rgbaColor=[0.2, 0.2, 0.8, 1.0]
        )
        self.pole_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=pole_shape,
            baseVisualShapeIndex=pole_visual,
            basePosition=[0, 0, 0.6]
        )
        
        # Create hinge joint between cart and pole
        p.createConstraint(
            self.cart_id, -1,
            self.pole_id, -1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0.1],
            childFramePosition=[0, 0, -0.5]
        )
        
        return self.cart_id
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation from the environment.
        
        Returns:
            Observation array [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        """
        # Get cart state
        cart_pos, cart_orn = p.getBasePositionAndOrientation(self.cart_id)
        cart_vel, cart_ang_vel = p.getBaseVelocity(self.cart_id)
        
        # Get pole orientation (angle from vertical)
        pole_pos, pole_orn = p.getBasePositionAndOrientation(self.pole_id)
        pole_vel, pole_ang_vel = p.getBaseVelocity(self.pole_id)
        
        # Convert quaternion to euler angles to get pole angle
        pole_euler = p.getEulerFromQuaternion(pole_orn)
        pole_angle = pole_euler[1]  # Pitch angle
        
        observation = np.array([
            cart_pos[0],           # Cart x position
            cart_vel[0],           # Cart x velocity
            pole_angle,            # Pole angle from vertical
            pole_ang_vel[1]        # Pole angular velocity
        ], dtype=np.float32)
        
        return observation
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on pole angle and cart position.
        
        Returns:
            Reward value
        """
        obs = self._get_observation()
        cart_pos = obs[0]
        pole_angle = obs[2]
        
        # Reward for keeping pole upright
        angle_reward = np.cos(pole_angle)
        
        # Penalty for cart moving too far from center
        position_penalty = -0.1 * abs(cart_pos)
        
        # Small step reward to encourage longer episodes
        step_reward = 0.1
        
        total_reward = angle_reward + position_penalty + step_reward
        
        return total_reward
    
    def _is_done(self) -> bool:
        """
        Check if episode should terminate.
        
        Returns:
            True if pole has fallen or cart is too far
        """
        obs = self._get_observation()
        cart_pos = obs[0]
        pole_angle = obs[2]
        
        # Episode ends if pole falls too far or cart moves too far
        pole_fallen = abs(pole_angle) > np.pi / 4  # 45 degrees
        cart_too_far = abs(cart_pos) > 2.0
        
        return pole_fallen or cart_too_far
    
    def _apply_action(self, action: np.ndarray):
        """
        Apply force to the cart.
        
        Args:
            action: Force to apply (normalized to [-1, 1])
        """
        force = action[0] * self.force_magnitude
        p.applyExternalForce(
            self.cart_id,
            -1,
            forceObj=[force, 0, 0],
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME
        )
