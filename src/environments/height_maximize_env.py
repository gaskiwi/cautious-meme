"""
Training Session 1: Height Maximization Environment

This is the FIRST training session in a series of training sessions.
Goal: Get one of the robots as high (vertically) as possible.
- Type A: Bar robot with joints (bar_with_joint.urdf) - 2N robots
- Type B: Sphere robot (rolling_sphere.urdf) - N robots
- Type A endpoints can connect to Type B robots
- Reward based on highest z level at end of episode
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import os
from typing import Optional, Dict, Tuple, List
from .base_robot_env import BaseRobotEnv


class HeightMaximizeEnv(BaseRobotEnv):
    """
    Training Session 1: Height Maximization Environment
    
    The objective is to train agents to maximize the vertical height of robots.
    This environment supports multiple robots of two types:
    - Type A: Bar robots with joints (2N robots)
    - Type B: Sphere robots (N robots)
    
    Features:
    - Multi-robot support with proper URDF models
    - Type A robots can connect to Type B robots via endpoints
    - Joint motors configured to lift twice the robot's body weight
    - Reward based on maximum height achieved by any robot at end of episode
    - Random deployment on plane at start of each episode
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_type_b_robots: int = 2,  # N
        spawn_radius: float = 3.0,    # Radius for random spawn area
        **kwargs
    ):
        """
        Initialize the Height Maximization Environment.
        
        Args:
            render_mode: Render mode ('human', 'rgb_array', or None)
            num_type_b_robots: Number of Type B robots (N). Type A will be 2N
            spawn_radius: Radius of circular area for random robot spawning
            **kwargs: Additional arguments passed to BaseRobotEnv
        """
        self.num_type_b = num_type_b_robots
        self.num_type_a = 2 * num_type_b_robots
        self.total_robots = self.num_type_a + self.num_type_b
        self.spawn_radius = spawn_radius
        
        # Robot tracking
        self.type_a_robots = []  # List of (robot_id, joint_indices) tuples
        self.type_b_robots = []  # List of robot_ids
        self.all_robot_ids = []  # All robot IDs in order
        
        # Connection tracking (Type A endpoint to Type B sphere)
        self.connections = []  # List of (constraint_id, type_a_idx, type_b_idx, link_idx)
        self.connection_distance = 0.2  # Max distance for connection (in meters)
        
        # Physics parameters
        self.type_a_mass = 1.1  # Total mass (base 0.1 + 2 bars @ 0.5 each)
        self.type_b_mass = 1.0  # Sphere mass
        self.gravity = 9.81
        
        # Joint force: enough to lift twice body weight
        # Force = 2 * mass * g
        self.type_a_joint_force = 2.0 * self.type_a_mass * self.gravity  # ~21.6 N
        
        # Get path to URDF models
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        self.type_a_urdf = os.path.join(self.models_dir, 'bar_with_joint.urdf')
        self.type_b_urdf = os.path.join(self.models_dir, 'rolling_sphere.urdf')
        
        # Track max height for reward
        self.max_height_achieved = 0.0
        
        # Initialize base environment first
        super().__init__(render_mode=render_mode, **kwargs)
        
        # Action space: 
        # - Type A robots: control 2 spherical joints (each has 3 DOF for torques)
        #   But PyBullet spherical joints are controlled as 3 separate motors
        #   So we need position targets for each joint's 3 axes
        # - Type B robots: apply forces (3 DOF)
        # Total: type_a * 6 (2 joints * 3 DOF) + type_b * 3 (force x,y,z)
        action_dim = self.num_type_a * 6 + self.num_type_b * 3
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Observation space:
        # For Type A: position (3), orientation (4), velocity (3), angular_vel (3), joint_states (6)
        # For Type B: position (3), orientation (4), velocity (3), angular_vel (3)
        # Type A: 19 values per robot
        # Type B: 13 values per robot
        obs_dim = self.num_type_a * 19 + self.num_type_b * 13
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Force magnitude for Type B robots
        self.type_b_force_magnitude = 20.0
        
    def _load_robot(self) -> int:
        """
        Load multiple robots (2N Type A, N Type B) into the simulation.
        Randomly position them on the plane.
        
        Returns:
            ID of the first robot (for compatibility with base class)
        """
        self.type_a_robots = []
        self.type_b_robots = []
        self.all_robot_ids = []
        self.connections = []
        
        # Random positions within spawn radius
        positions = self._generate_random_positions(self.total_robots)
        pos_idx = 0
        
        # Load Type A robots (bar with joints)
        for i in range(self.num_type_a):
            x, y = positions[pos_idx]
            pos_idx += 1
            
            # Random orientation around Z axis
            yaw = np.random.uniform(0, 2 * np.pi)
            orientation = p.getQuaternionFromEuler([0, 0, yaw])
            
            robot_id = p.loadURDF(
                self.type_a_urdf,
                basePosition=[x, y, 0.5],  # Start slightly above ground
                baseOrientation=orientation,
                useFixedBase=False
            )
            
            # Get joint information
            num_joints = p.getNumJoints(robot_id)
            joint_indices = []
            
            for joint_idx in range(num_joints):
                joint_info = p.getJointInfo(robot_id, joint_idx)
                joint_type = joint_info[2]
                
                # Spherical joints in PyBullet (type 5)
                if joint_type == p.JOINT_SPHERICAL:
                    joint_indices.append(joint_idx)
                    
                    # Enable motors with force limit
                    # Spherical joints have 3 DOF, controlled separately
                    p.setJointMotorControlMultiDof(
                        robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=[0, 0, 0, 1],  # Quaternion for neutral position
                        force=[self.type_a_joint_force] * 3
                    )
            
            self.type_a_robots.append((robot_id, joint_indices))
            self.all_robot_ids.append(robot_id)
        
        # Load Type B robots (spheres)
        for i in range(self.num_type_b):
            x, y = positions[pos_idx]
            pos_idx += 1
            
            robot_id = p.loadURDF(
                self.type_b_urdf,
                basePosition=[x, y, 0.5],  # Start slightly above ground
                useFixedBase=False
            )
            
            self.type_b_robots.append(robot_id)
            self.all_robot_ids.append(robot_id)
        
        # Return the first robot ID for base class compatibility
        return self.all_robot_ids[0] if self.all_robot_ids else None
    
    def _generate_random_positions(self, num_positions: int) -> List[Tuple[float, float]]:
        """
        Generate random (x, y) positions within spawn radius.
        Ensures minimum separation between robots.
        
        Args:
            num_positions: Number of positions to generate
            
        Returns:
            List of (x, y) tuples
        """
        positions = []
        min_separation = 0.4  # Minimum distance between robots
        max_attempts = 100
        
        for _ in range(num_positions):
            attempts = 0
            while attempts < max_attempts:
                # Random position in circle
                r = np.random.uniform(0, self.spawn_radius)
                theta = np.random.uniform(0, 2 * np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                # Check separation from existing positions
                too_close = False
                for px, py in positions:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < min_separation:
                        too_close = True
                        break
                
                if not too_close:
                    positions.append((x, y))
                    break
                    
                attempts += 1
            
            # If we couldn't find a good position, just place it anyway
            if attempts >= max_attempts:
                positions.append((x, y))
        
        return positions
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation from all robots in the environment.
        
        Returns:
            Flattened observation array containing state of all robots
        """
        observations = []
        
        # Type A robots
        for robot_id, joint_indices in self.type_a_robots:
            # Base state
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(robot_id)
            
            # Joint states
            joint_states = []
            for joint_idx in joint_indices:
                joint_state = p.getJointStateMultiDof(robot_id, joint_idx)
                # joint_state[0] is position (quaternion for spherical)
                # joint_state[1] is velocity
                # For spherical joints, we get quaternion and angular velocity
                joint_pos = joint_state[0]  # Quaternion (4 values)
                joint_vel = joint_state[1]  # Angular velocity (3 values)
                
                # We'll use first 3 values of position and velocity
                joint_states.extend(joint_pos[:3])
            
            robot_obs = np.array([
                pos[0], pos[1], pos[2],           # Position (3)
                orn[0], orn[1], orn[2], orn[3],   # Orientation quaternion (4)
                lin_vel[0], lin_vel[1], lin_vel[2],  # Linear velocity (3)
                ang_vel[0], ang_vel[1], ang_vel[2],  # Angular velocity (3)
                *joint_states                      # Joint states (6)
            ], dtype=np.float32)
            
            observations.append(robot_obs)
        
        # Type B robots
        for robot_id in self.type_b_robots:
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(robot_id)
            
            robot_obs = np.array([
                pos[0], pos[1], pos[2],           # Position (3)
                orn[0], orn[1], orn[2], orn[3],   # Orientation quaternion (4)
                lin_vel[0], lin_vel[1], lin_vel[2],  # Linear velocity (3)
                ang_vel[0], ang_vel[1], ang_vel[2]   # Angular velocity (3)
            ], dtype=np.float32)
            
            observations.append(robot_obs)
        
        # Flatten all observations into single array
        return np.concatenate(observations)
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on the maximum height achieved by any robot.
        
        Training Session 1 Goal: Maximize vertical height
        According to issue: reward is highest z level at END of episode
        We'll track during episode and give final reward at end
        
        Returns:
            Reward value
        """
        # Get heights of all robots
        heights = []
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            heights.append(pos[2])  # Z-coordinate is height
        
        # Maximum height among all robots
        current_max_height = max(heights)
        
        # Track the best height achieved
        if current_max_height > self.max_height_achieved:
            self.max_height_achieved = current_max_height
        
        # Small per-step reward based on current max height
        # This helps guide learning during episode
        step_reward = current_max_height * 0.1
        
        # We'll add a large bonus at episode end (in step method)
        return step_reward
    
    def _is_done(self) -> bool:
        """
        Check if episode should terminate.
        
        Episode ends if:
        - Any robot falls below ground level significantly
        - Any robot moves too far horizontally (out of bounds)
        
        Returns:
            True if episode should terminate
        """
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            
            # Check if robot fell below ground significantly
            if pos[2] < -1.0:
                return True
            
            # Check if robot moved too far horizontally
            horizontal_distance = np.sqrt(pos[0]**2 + pos[1]**2)
            if horizontal_distance > 20.0:
                return True
        
        return False
    
    def _apply_action(self, action: np.ndarray):
        """
        Apply actions to all robots.
        
        Args:
            action: Action array
                Type A robots: joint position targets (6 values per robot: 2 joints * 3 DOF)
                Type B robots: forces (3 values per robot: fx, fy, fz)
        """
        action_idx = 0
        
        # Type A robots: control joints
        for robot_id, joint_indices in self.type_a_robots:
            for joint_idx in joint_indices:
                # Get 3 action values for this joint's 3 DOF
                joint_action = action[action_idx:action_idx + 3]
                action_idx += 3
                
                # Map from [-1, 1] to joint angle range (e.g., -pi to pi)
                target_angles = joint_action * np.pi
                
                # Convert to quaternion for spherical joint
                # Simplified: use the 3 values as Euler angles
                quat = p.getQuaternionFromEuler(target_angles.tolist())
                
                # Set joint target position
                p.setJointMotorControlMultiDof(
                    robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=quat,
                    force=[self.type_a_joint_force] * 3,
                    positionGain=0.3,
                    velocityGain=0.1
                )
        
        # Type B robots: apply forces
        for robot_id in self.type_b_robots:
            force_action = action[action_idx:action_idx + 3]
            action_idx += 3
            
            # Scale action to force magnitude
            force = force_action * self.type_b_force_magnitude
            
            # Apply external force at center of mass
            p.applyExternalForce(
                robot_id,
                -1,  # Link index (-1 for base)
                forceObj=force.tolist(),
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME
            )
    
    def _get_bar_endpoint_position(self, robot_id: int, link_idx: int) -> np.ndarray:
        """
        Get the world position of a bar endpoint.
        
        Args:
            robot_id: ID of the Type A robot
            link_idx: Link index (1 for bar_1, 2 for bar_2)
            
        Returns:
            3D position of the endpoint
        """
        # Get link state
        link_state = p.getLinkState(robot_id, link_idx)
        link_pos = np.array(link_state[0])  # World position of link COM
        link_orn = link_state[1]  # World orientation
        
        # The bar extends 0.5m along its length, so endpoint is 0.25m from COM
        # Bar is oriented along x-axis in local frame
        # We need to transform the endpoint offset by link orientation
        local_endpoint_offset = [0.5, 0, 0]  # End of 0.5m bar
        
        # Convert quaternion to rotation matrix and apply
        rotation_matrix = p.getMatrixFromQuaternion(link_orn)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        
        endpoint_offset = rotation_matrix @ local_endpoint_offset
        endpoint_pos = link_pos + endpoint_offset
        
        return endpoint_pos
    
    def _check_and_create_connections(self):
        """
        Check for Type A endpoints near Type B spheres and create connections.
        
        Type A robots have 2 bars (bar_1 and bar_2), each with an endpoint.
        When an endpoint comes close to a Type B sphere center, create a constraint.
        
        This enables Type A robots to attach to Type B robots for collaborative height building.
        """
        # Check each Type A robot
        for type_a_idx, (robot_id, joint_indices) in enumerate(self.type_a_robots):
            # Each Type A robot has 2 bars (links 1 and 2)
            # Skip link 0 (base)
            num_links = p.getNumJoints(robot_id)
            
            for link_idx in range(1, num_links + 1):
                # Get endpoint position
                try:
                    endpoint_pos = self._get_bar_endpoint_position(robot_id, link_idx)
                except:
                    continue  # Skip if we can't get position
                
                # Check distance to each Type B robot
                for type_b_idx, type_b_id in enumerate(self.type_b_robots):
                    # Get Type B position (center of sphere)
                    type_b_pos, _ = p.getBasePositionAndOrientation(type_b_id)
                    type_b_pos = np.array(type_b_pos)
                    
                    # Calculate distance
                    distance = np.linalg.norm(endpoint_pos - type_b_pos)
                    
                    # Check if close enough and not already connected
                    if distance < self.connection_distance:
                        # Check if this connection already exists
                        already_connected = False
                        for conn in self.connections:
                            conn_a_idx, conn_b_idx, conn_link = conn[1], conn[2], conn[3]
                            if (conn_a_idx == type_a_idx and 
                                conn_b_idx == type_b_idx and 
                                conn_link == link_idx):
                                already_connected = True
                                break
                        
                        if not already_connected:
                            # Create point-to-point constraint
                            # This connects the endpoint of Type A to the center of Type B
                            constraint_id = p.createConstraint(
                                parentBodyUniqueId=robot_id,
                                parentLinkIndex=link_idx,
                                childBodyUniqueId=type_b_id,
                                childLinkIndex=-1,  # Base of Type B
                                jointType=p.JOINT_POINT2POINT,
                                jointAxis=[0, 0, 0],
                                parentFramePosition=[0.5, 0, 0],  # Endpoint in link frame
                                childFramePosition=[0, 0, 0]  # Center of sphere
                            )
                            
                            # Track the connection
                            self.connections.append((constraint_id, type_a_idx, type_b_idx, link_idx))
    
    def _remove_weak_connections(self):
        """
        Remove connections where the robots have moved too far apart.
        This allows dynamic connection/disconnection.
        """
        connections_to_remove = []
        
        for i, (constraint_id, type_a_idx, type_b_idx, link_idx) in enumerate(self.connections):
            robot_id = self.type_a_robots[type_a_idx][0]
            type_b_id = self.type_b_robots[type_b_idx]
            
            try:
                # Get current endpoint position
                endpoint_pos = self._get_bar_endpoint_position(robot_id, link_idx)
                
                # Get Type B position
                type_b_pos, _ = p.getBasePositionAndOrientation(type_b_id)
                type_b_pos = np.array(type_b_pos)
                
                # Calculate distance
                distance = np.linalg.norm(endpoint_pos - type_b_pos)
                
                # If too far apart, mark for removal
                # Use 2x connection distance as threshold
                if distance > self.connection_distance * 2.5:
                    connections_to_remove.append(i)
            except:
                # If we can't check, remove the connection
                connections_to_remove.append(i)
        
        # Remove constraints in reverse order to maintain indices
        for i in reversed(connections_to_remove):
            constraint_id = self.connections[i][0]
            p.removeConstraint(constraint_id)
            del self.connections[i]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset max height tracker
        self.max_height_achieved = 0.0
        
        # Clear connections
        for constraint_id in self.connections:
            p.removeConstraint(constraint_id)
        self.connections = []
        
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        # Add session info
        info['training_session'] = 1
        info['session_goal'] = 'maximize_height'
        info['max_height'] = self.max_height_achieved
        info['num_type_a'] = self.num_type_a
        info['num_type_b'] = self.num_type_b
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Check and create new connections
        self._check_and_create_connections()
        
        # Remove weak connections
        self._remove_weak_connections()
        
        observation, reward, terminated, truncated, info = super().step(action)
        
        # At episode end, give reward based on max height achieved
        if terminated or truncated:
            # Final reward is the maximum height achieved
            reward += self.max_height_achieved * 10.0  # Large bonus for final height
        
        # Add training session specific info
        info['training_session'] = 1
        info['max_height'] = self.max_height_achieved
        
        # Track heights of all robots
        heights = []
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            heights.append(pos[2])
        
        info['robot_heights'] = heights
        info['highest_robot_idx'] = np.argmax(heights)
        info['num_connections'] = len(self.connections)
        
        return observation, reward, terminated, truncated, info
