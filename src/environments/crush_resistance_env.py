"""
Training Session 2: Crush Resistance Environment

This is the SECOND training session in a series of training sessions.
Goal: Resist a descending hydraulic press for as long as possible.
- Type A: Bar robot with joints (bar_with_joint.urdf) - 2N robots
- Type B: Sphere robot (rolling_sphere.urdf) - N robots
- Type A endpoints can connect to Type B robots
- First 30 seconds: robots can position themselves
- After 30 seconds: a plane descends from above like a hydraulic press
- Press applies increasing force when encountering obstacles until crushed
- Reward based on time elapsed after press starts until all robots touch ground
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import os
from typing import Optional, Dict, Tuple, List
from .base_robot_env import BaseRobotEnv


class CrushResistanceEnv(BaseRobotEnv):
    """
    Training Session 2: Crush Resistance Environment
    
    The objective is to train agents to resist a descending hydraulic press.
    This environment supports the same robots as Session 1:
    - Type A: Bar robots with joints (2N robots)
    - Type B: Sphere robots (N robots)
    
    Features:
    - Same physics and robots as HeightMaximizeEnv
    - Episode runs for 30 seconds of free movement
    - After 30 seconds, a descending plane acts like a hydraulic press
    - Press starts from a height above the max height from session 1
    - Press applies increasing force to crush obstacles
    - Reward based on survival time after press activation
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_type_b_robots: int = 2,  # N
        spawn_radius: float = 3.0,    # Radius for random spawn area
        reference_height: float = 5.0,  # Height above which press starts (from session 1 max)
        press_descent_speed: float = 0.05,  # Speed of press descent (m/s)
        press_force_increment: float = 50.0,  # Force increment when pressing (N per step)
        **kwargs
    ):
        """
        Initialize the Crush Resistance Environment.
        
        Args:
            render_mode: Render mode ('human', 'rgb_array', or None)
            num_type_b_robots: Number of Type B robots (N). Type A will be 2N
            spawn_radius: Radius of circular area for random robot spawning
            reference_height: Starting height of press (should be above session 1 max height)
            press_descent_speed: Speed at which press descends (m/s)
            press_force_increment: Force increment when press encounters resistance (N)
            **kwargs: Additional arguments passed to BaseRobotEnv
        """
        self.num_type_b = num_type_b_robots
        self.num_type_a = 2 * num_type_b_robots
        self.total_robots = self.num_type_a + self.num_type_b
        self.spawn_radius = spawn_radius
        
        # Press parameters
        self.reference_height = reference_height
        self.press_descent_speed = press_descent_speed
        self.press_force_increment = press_force_increment
        
        # Robot tracking
        self.type_a_robots = []  # List of (robot_id, joint_indices) tuples
        self.type_b_robots = []  # List of robot_ids
        self.all_robot_ids = []  # All robot IDs in order
        
        # Connection tracking (Type A endpoint to Type B sphere)
        self.connections = []  # List of (constraint_id, type_a_idx, type_b_idx, link_idx)
        self.connection_distance = 0.2  # Max distance for connection (in meters)
        
        # Physics parameters (same as HeightMaximizeEnv)
        self.type_a_mass = 1.1  # Total mass (base 0.1 + 2 bars @ 0.5 each)
        self.type_b_mass = 1.0  # Sphere mass
        self.gravity = 9.81
        
        # Joint force: enough to lift twice body weight
        self.type_a_joint_force = 2.0 * self.type_a_mass * self.gravity  # ~21.6 N
        
        # Get path to URDF models
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        self.type_a_urdf = os.path.join(self.models_dir, 'bar_with_joint.urdf')
        self.type_b_urdf = os.path.join(self.models_dir, 'rolling_sphere.urdf')
        
        # Press tracking
        self.press_id = None
        self.press_active = False
        self.press_start_step = 1800  # 30 seconds at 60 FPS (1/60s per step)
        self.press_current_height = reference_height
        self.press_current_force = 0.0
        
        # Survival tracking
        self.survival_time = 0.0  # Time survived after press activation
        self.all_robots_grounded = False
        
        # Initialize base environment first
        super().__init__(render_mode=render_mode, **kwargs)
        
        # Action space: same as HeightMaximizeEnv
        # Type A robots: control 2 spherical joints (each has 3 DOF)
        # Type B robots: apply forces (3 DOF)
        action_dim = self.num_type_a * 6 + self.num_type_b * 3
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Observation space: same as HeightMaximizeEnv
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
        Also creates the hydraulic press plane.
        
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
        
        # Create the hydraulic press (initially invisible/inactive)
        self._create_press()
        
        # Return the first robot ID for base class compatibility
        return self.all_robot_ids[0] if self.all_robot_ids else None
    
    def _create_press(self):
        """
        Create the hydraulic press plane.
        This is a large horizontal plane that will descend from above.
        """
        # Create a large horizontal plane (box with small height)
        press_size = [10.0, 10.0, 0.1]  # Large plane, 10x10 meters, 0.1m thick
        
        press_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=press_size
        )
        
        press_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=press_size,
            rgbaColor=[0.8, 0.2, 0.2, 0.8]  # Semi-transparent red
        )
        
        # Start the press high above the reference height (inactive)
        self.press_current_height = self.reference_height
        
        self.press_id = p.createMultiBody(
            baseMass=100.0,  # Heavy press
            baseCollisionShapeIndex=press_collision,
            baseVisualShapeIndex=press_visual,
            basePosition=[0, 0, self.press_current_height]
        )
        
        # Initially disable collisions by setting it as a static object high up
        # We'll enable it when the press activates
        p.changeDynamics(
            self.press_id,
            -1,
            mass=100.0,
            lateralFriction=1.0,
            spinningFriction=0.1,
            rollingFriction=0.1
        )
    
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
            if attempts == max_attempts:
                positions.append((x, y))
        
        return positions
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation from all robots in the environment.
        Same as HeightMaximizeEnv.
        
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
                joint_pos = joint_state[0]  # Quaternion (4 values)
                # We'll use first 3 values of position
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
    
    def _update_press(self):
        """
        Update the hydraulic press position and apply forces.
        
        The press:
        1. Descends at a constant speed
        2. When it encounters resistance, applies increasing force
        3. Continues until all robots are crushed to the ground
        """
        if not self.press_active:
            return
        
        # Get current press position
        press_pos, press_orn = p.getBasePositionAndOrientation(self.press_id)
        
        # Target height: descend at constant speed
        # Speed is in m/s, timestep is 1/60s per step
        descent_per_step = self.press_descent_speed / 60.0
        target_height = self.press_current_height - descent_per_step
        
        # Check for contacts with robots or ground
        contact_points = p.getContactPoints(bodyA=self.press_id)
        
        if len(contact_points) > 0:
            # Press is in contact with something
            # Apply downward force that increases over time
            self.press_current_force += self.press_force_increment
            
            # Apply the force to push through obstacles
            downward_force = [0, 0, -self.press_current_force]
            p.applyExternalForce(
                self.press_id,
                -1,  # Base link
                downward_force,
                [0, 0, 0],
                p.LINK_FRAME
            )
        else:
            # No contact, reset force and descend normally
            self.press_current_force = 0.0
        
        # Move press downward (using velocity control for smooth descent)
        p.resetBaseVelocity(
            self.press_id,
            linearVelocity=[0, 0, -self.press_descent_speed]
        )
        
        # Update tracked height
        self.press_current_height = press_pos[2]
    
    def _check_all_robots_grounded(self) -> bool:
        """
        Check if all robots are touching the ground plane.
        Episode ends when all robots are grounded.
        
        Returns:
            True if all robots are on the ground
        """
        ground_threshold = 0.3  # Height below which robot is considered grounded
        
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            
            # Check if robot is above ground threshold
            if pos[2] > ground_threshold:
                return False
        
        return True
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on survival time after press activation.
        
        Training Session 2 Goal: Resist the press as long as possible
        - Small reward during positioning phase (first 30 seconds)
        - Large reward for each second survived after press starts
        - Bonus for keeping robots off the ground
        
        Returns:
            Reward value
        """
        if not self.press_active:
            # During positioning phase: small reward for building height
            # This encourages robots to position themselves well
            heights = []
            for robot_id in self.all_robot_ids:
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                heights.append(pos[2])
            
            max_height = max(heights)
            positioning_reward = max_height * 0.01  # Small reward
            
            return positioning_reward
        else:
            # After press starts: reward for survival time
            # Each step survived is valuable
            survival_reward = 1.0  # 1 point per step survived
            
            # Bonus for keeping robots higher (resisting the press)
            heights = []
            for robot_id in self.all_robot_ids:
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                heights.append(pos[2])
            
            avg_height = np.mean(heights)
            height_bonus = avg_height * 0.5  # Bonus for maintaining height
            
            total_reward = survival_reward + height_bonus
            
            # Track survival time
            self.survival_time += 1.0 / 60.0  # Add time in seconds
            
            return total_reward
    
    def _is_done(self) -> bool:
        """
        Check if episode should terminate.
        
        Episode ends if:
        - All robots are grounded (pressed to the ground)
        - Any robot falls below ground level significantly
        - Any robot moves too far horizontally (out of bounds)
        
        Returns:
            True if episode should terminate
        """
        # Check if all robots are grounded (main termination condition)
        if self.press_active and self._check_all_robots_grounded():
            self.all_robots_grounded = True
            return True
        
        # Safety checks
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            
            # Check if robot fell through ground
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
        Same as HeightMaximizeEnv.
        
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
                
                # Map from [-1, 1] to joint angle range
                target_angles = joint_action * np.pi
                
                # Convert to quaternion for spherical joint
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
        Same as HeightMaximizeEnv.
        
        Args:
            robot_id: ID of the Type A robot
            link_idx: Link index (1 for bar_1, 2 for bar_2)
            
        Returns:
            3D position of the endpoint
        """
        link_state = p.getLinkState(robot_id, link_idx)
        link_pos = np.array(link_state[0])
        link_orn = link_state[1]
        
        local_endpoint_offset = [0.5, 0, 0]
        
        rotation_matrix = p.getMatrixFromQuaternion(link_orn)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        
        endpoint_offset = rotation_matrix @ local_endpoint_offset
        endpoint_pos = link_pos + endpoint_offset
        
        return endpoint_pos
    
    def _check_and_create_connections(self):
        """
        Check for Type A endpoints near Type B spheres and create connections.
        Same as HeightMaximizeEnv.
        """
        # Check each Type A robot
        for type_a_idx, (robot_id, joint_indices) in enumerate(self.type_a_robots):
            num_links = p.getNumJoints(robot_id)
            
            for link_idx in range(1, num_links + 1):
                try:
                    endpoint_pos = self._get_bar_endpoint_position(robot_id, link_idx)
                except:
                    continue
                
                # Check distance to each Type B robot
                for type_b_idx, type_b_id in enumerate(self.type_b_robots):
                    type_b_pos, _ = p.getBasePositionAndOrientation(type_b_id)
                    type_b_pos = np.array(type_b_pos)
                    
                    distance = np.linalg.norm(endpoint_pos - type_b_pos)
                    
                    if distance < self.connection_distance:
                        # Check if already connected
                        already_connected = False
                        for conn in self.connections:
                            conn_a_idx, conn_b_idx, conn_link = conn[1], conn[2], conn[3]
                            if (conn_a_idx == type_a_idx and 
                                conn_b_idx == type_b_idx and 
                                conn_link == link_idx):
                                already_connected = True
                                break
                        
                        if not already_connected:
                            constraint_id = p.createConstraint(
                                parentBodyUniqueId=robot_id,
                                parentLinkIndex=link_idx,
                                childBodyUniqueId=type_b_id,
                                childLinkIndex=-1,
                                jointType=p.JOINT_POINT2POINT,
                                jointAxis=[0, 0, 0],
                                parentFramePosition=[0.5, 0, 0],
                                childFramePosition=[0, 0, 0]
                            )
                            
                            self.connections.append((constraint_id, type_a_idx, type_b_idx, link_idx))
    
    def _remove_weak_connections(self):
        """
        Remove connections where the robots have moved too far apart.
        Same as HeightMaximizeEnv.
        """
        connections_to_remove = []
        
        for i, (constraint_id, type_a_idx, type_b_idx, link_idx) in enumerate(self.connections):
            robot_id = self.type_a_robots[type_a_idx][0]
            type_b_id = self.type_b_robots[type_b_idx]
            
            try:
                endpoint_pos = self._get_bar_endpoint_position(robot_id, link_idx)
                type_b_pos, _ = p.getBasePositionAndOrientation(type_b_id)
                type_b_pos = np.array(type_b_pos)
                
                distance = np.linalg.norm(endpoint_pos - type_b_pos)
                
                if distance > self.connection_distance * 2.5:
                    connections_to_remove.append(i)
            except:
                connections_to_remove.append(i)
        
        # Remove constraints in reverse order
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
        
        # Reset press tracking
        self.press_active = False
        self.press_current_force = 0.0
        self.press_current_height = self.reference_height
        self.survival_time = 0.0
        self.all_robots_grounded = False
        
        # Clear connections
        for constraint_id, _, _, _ in self.connections:
            p.removeConstraint(constraint_id)
        self.connections = []
        
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        # Add session info
        info['training_session'] = 2
        info['session_goal'] = 'resist_crush'
        info['press_active'] = self.press_active
        info['survival_time'] = self.survival_time
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
        # Activate press after 30 seconds (1800 steps)
        if not self.press_active and self._step_counter >= self.press_start_step:
            self.press_active = True
        
        # Update press if active
        if self.press_active:
            self._update_press()
        
        # Check and create new connections
        self._check_and_create_connections()
        
        # Remove weak connections
        self._remove_weak_connections()
        
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Add training session specific info
        info['training_session'] = 2
        info['press_active'] = self.press_active
        info['survival_time'] = self.survival_time
        info['press_height'] = self.press_current_height
        info['press_force'] = self.press_current_force
        
        # Track heights of all robots
        heights = []
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            heights.append(pos[2])
        
        info['robot_heights'] = heights
        info['avg_height'] = np.mean(heights)
        info['max_height'] = np.max(heights)
        info['num_connections'] = len(self.connections)
        info['all_grounded'] = self.all_robots_grounded
        
        # Final bonus for surviving
        if terminated or truncated:
            if self.press_active:
                # Bonus based on total survival time
                reward += self.survival_time * 10.0
        
        return observation, reward, terminated, truncated, info
