"""Base environment class for robot simulation using PyBullet."""

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional


class BaseRobotEnv(gym.Env, ABC):
    """
    Base class for robot environments using PyBullet.
    
    This serves as a template for creating custom robot environments.
    Subclasses should implement the abstract methods to define specific
    robot behaviors and tasks.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        timestep: float = 1./240.,
        frame_skip: int = 4,
        max_episode_steps: int = 1000
    ):
        """
        Initialize the robot environment.
        
        Args:
            render_mode: Render mode ('human', 'rgb_array', or None)
            timestep: Physics simulation timestep
            frame_skip: Number of simulation steps per environment step
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        
        self._step_counter = 0
        self.physics_client = None
        self.robot_id = None
        
        # Connect to PyBullet
        self._connect()
        
        # Define action and observation spaces (to be set by subclass)
        self.action_space = None
        self.observation_space = None
        
    def _connect(self):
        """Connect to PyBullet physics server."""
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        
    @abstractmethod
    def _load_robot(self) -> int:
        """
        Load the robot into the simulation.
        
        Returns:
            Robot body ID
        """
        pass
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation from the environment.
        
        Returns:
            Observation array
        """
        pass
    
    @abstractmethod
    def _compute_reward(self) -> float:
        """
        Compute the reward for the current state.
        
        Returns:
            Reward value
        """
        pass
    
    @abstractmethod
    def _is_done(self) -> bool:
        """
        Check if the episode is done.
        
        Returns:
            True if episode should terminate
        """
        pass
    
    @abstractmethod
    def _apply_action(self, action: np.ndarray):
        """
        Apply action to the robot.
        
        Args:
            action: Action array
        """
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Load robot
        self.robot_id = self._load_robot()
        
        # Reset step counter
        self._step_counter = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        for _ in range(self.frame_skip):
            p.stepSimulation()
        
        # Get observation
        observation = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if done
        terminated = self._is_done()
        
        # Check if truncated (max steps reached)
        self._step_counter += 1
        truncated = self._step_counter >= self.max_episode_steps
        
        info = {
            'step': self._step_counter
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Get camera image
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=2.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width/height, nearVal=0.1, farVal=100.0
            )
            
            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (height, width, 4))[:, :, :3]
            return rgb_array
    
    def close(self):
        """Clean up resources."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
