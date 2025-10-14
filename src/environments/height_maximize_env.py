"""
Training Session 1: Height Maximization Environment

This is the FIRST training session in a series of training sessions.
Goal: Get one of the robots as high (vertically) as possible.
- It doesn't matter which type or which robot
- The highest robot scores for the entire team
- This environment will be used in succession with other training sessions later
"""

import gymnasium as gym
import numpy as np
import pybullet as p
from typing import Optional, Dict, Tuple
from .base_robot_env import BaseRobotEnv


class HeightMaximizeEnv(BaseRobotEnv):
    """
    Training Session 1: Height Maximization Environment
    
    The objective is to train agents to maximize the vertical height of robots.
    This environment supports multiple robots, and the reward is based on the
    highest robot's vertical position (Z-coordinate).
    
    Features:
    - Multi-robot support (configurable number of robots)
    - Reward based on maximum height achieved by any robot
    - Simple box robots with controllable joints/forces
    - Encourages exploration of vertical movement strategies
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_robots: int = 3,
        **kwargs
    ):
        """
        Initialize the Height Maximization Environment.
        
        Args:
            render_mode: Render mode ('human', 'rgb_array', or None)
            num_robots: Number of robots in the environment (default: 3)
            **kwargs: Additional arguments passed to BaseRobotEnv
        """
        self.num_robots = num_robots
        self.robot_ids = []
        self.max_height_achieved = 0.0
        self.initial_height = 1.0  # Starting height of robots
        
        # Initialize base environment first
        super().__init__(render_mode=render_mode, **kwargs)
        
        # Each robot has 6 actuators (forces/torques in 3D space)
        # Action space: continuous control for all robots
        # [robot1_fx, robot1_fy, robot1_fz, robot1_tx, robot1_ty, robot1_tz, robot2_fx, ...]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_robots * 6,),  # 6 DOF per robot
            dtype=np.float32
        )
        
        # Observation space: position, velocity, orientation for each robot
        # For each robot: [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]
        # Total: 13 values per robot
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_robots * 13,),
            dtype=np.float32
        )
        
        self.force_magnitude = 50.0  # Maximum force that can be applied
        self.torque_magnitude = 20.0  # Maximum torque that can be applied
        
    def _load_robot(self) -> int:
        """
        Load multiple robots into the simulation.
        
        Returns:
            ID of the first robot (for compatibility with base class)
        """
        self.robot_ids = []
        
        # Create multiple robots in different positions
        for i in range(self.num_robots):
            # Position robots in a line with some spacing
            x_offset = (i - self.num_robots / 2) * 1.5
            
            # Create a simple box robot with some mass
            robot_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.3, 0.3, 0.5]  # Rectangular robot
            )
            
            # Different colors for different robots
            colors = [
                [0.8, 0.2, 0.2, 1.0],  # Red
                [0.2, 0.8, 0.2, 1.0],  # Green
                [0.2, 0.2, 0.8, 1.0],  # Blue
                [0.8, 0.8, 0.2, 1.0],  # Yellow
                [0.8, 0.2, 0.8, 1.0],  # Magenta
            ]
            color = colors[i % len(colors)]
            
            robot_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.3, 0.3, 0.5],
                rgbaColor=color
            )
            
            robot_id = p.createMultiBody(
                baseMass=2.0,  # 2 kg robot
                baseCollisionShapeIndex=robot_shape,
                baseVisualShapeIndex=robot_visual,
                basePosition=[x_offset, 0, self.initial_height]
            )
            
            self.robot_ids.append(robot_id)
        
        # Return the first robot ID for base class compatibility
        return self.robot_ids[0] if self.robot_ids else None
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation from all robots in the environment.
        
        Returns:
            Flattened observation array containing state of all robots
        """
        observations = []
        
        for robot_id in self.robot_ids:
            # Get position and orientation
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            
            # Get linear and angular velocity
            lin_vel, ang_vel = p.getBaseVelocity(robot_id)
            
            # Combine into single observation for this robot
            robot_obs = np.array([
                pos[0], pos[1], pos[2],           # Position (x, y, z)
                lin_vel[0], lin_vel[1], lin_vel[2],  # Linear velocity
                orn[0], orn[1], orn[2], orn[3],   # Orientation (quaternion)
                ang_vel[0], ang_vel[1], ang_vel[2]   # Angular velocity
            ], dtype=np.float32)
            
            observations.append(robot_obs)
        
        # Flatten all observations into single array
        return np.concatenate(observations)
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on the maximum height achieved by any robot.
        
        Training Session 1 Goal: Maximize vertical height
        - Primary reward: height of the highest robot
        - Bonus reward: improvements in max height
        - Small penalty for excessive movement/energy use
        
        Returns:
            Reward value
        """
        # Get heights of all robots
        heights = []
        for robot_id in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            heights.append(pos[2])  # Z-coordinate is height
        
        # Maximum height among all robots
        current_max_height = max(heights)
        
        # Primary reward: current maximum height (normalized)
        height_reward = current_max_height / self.initial_height
        
        # Bonus reward for achieving new maximum height
        height_improvement = 0.0
        if current_max_height > self.max_height_achieved:
            height_improvement = (current_max_height - self.max_height_achieved) * 10.0
            self.max_height_achieved = current_max_height
        
        # Small penalty for energy usage (encourage efficient solutions)
        # This is a placeholder - actual energy would be based on actions applied
        energy_penalty = -0.01
        
        # Total reward
        total_reward = height_reward + height_improvement + energy_penalty
        
        return total_reward
    
    def _is_done(self) -> bool:
        """
        Check if episode should terminate.
        
        Episode ends if:
        - Any robot falls below ground level
        - Any robot moves too far horizontally (out of bounds)
        
        Returns:
            True if episode should terminate
        """
        for robot_id in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            
            # Check if robot fell below ground
            if pos[2] < 0.0:
                return True
            
            # Check if robot moved too far horizontally
            horizontal_distance = np.sqrt(pos[0]**2 + pos[1]**2)
            if horizontal_distance > 10.0:
                return True
        
        return False
    
    def _apply_action(self, action: np.ndarray):
        """
        Apply forces and torques to all robots.
        
        Args:
            action: Action array with forces and torques for each robot
                   [robot1_fx, robot1_fy, robot1_fz, robot1_tx, robot1_ty, robot1_tz, ...]
        """
        for i, robot_id in enumerate(self.robot_ids):
            # Extract actions for this robot (6 values)
            start_idx = i * 6
            robot_action = action[start_idx:start_idx + 6]
            
            # Split into forces and torques
            force = robot_action[0:3] * self.force_magnitude
            torque = robot_action[3:6] * self.torque_magnitude
            
            # Apply external force at center of mass
            p.applyExternalForce(
                robot_id,
                -1,  # Link index (-1 for base)
                forceObj=force.tolist(),
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME
            )
            
            # Apply external torque
            p.applyExternalTorque(
                robot_id,
                -1,  # Link index (-1 for base)
                torqueObj=torque.tolist(),
                flags=p.LINK_FRAME
            )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        # Reset max height tracker
        self.max_height_achieved = self.initial_height
        
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        # Add session info
        info['training_session'] = 1
        info['session_goal'] = 'maximize_height'
        info['max_height'] = self.max_height_achieved
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Add training session specific info
        info['training_session'] = 1
        info['max_height'] = self.max_height_achieved
        
        # Track heights of all robots
        heights = []
        for robot_id in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            heights.append(pos[2])
        
        info['robot_heights'] = heights
        info['highest_robot'] = np.argmax(heights)
        
        return observation, reward, terminated, truncated, info
