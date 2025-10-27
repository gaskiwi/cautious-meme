"""
Training Session 3: Obstacle Navigation Environment

This is the THIRD training session in a series of training sessions.
Goal: Navigate through an obstacle course to reach target positions.
- Type A: Bar robot with joints (bar_with_joint.urdf) - 2N robots
- Type B: Sphere robot (rolling_sphere.urdf) - N robots
- Type A endpoints can connect to Type B robots
- Random obstacles (walls, barriers) spawn in the environment
- Target position(s) spawn at various locations
- Reward based on reaching targets and avoiding collisions
- Penalty for collisions with obstacles
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import os
from typing import Optional, Dict, Tuple, List
from .base_robot_env import BaseRobotEnv


class ObstacleNavigationEnv(BaseRobotEnv):
    """
    Training Session 3: Obstacle Navigation Environment
    
    The objective is to train agents to navigate through obstacles to reach targets.
    This environment supports the same robots as previous sessions:
    - Type A: Bar robots with joints (2N robots)
    - Type B: Sphere robots (N robots)
    
    Features:
    - Random obstacle placement (walls, boxes, cylinders)
    - Target zones that robots must reach
    - Dynamic target repositioning after reaching
    - Reward for progress toward target and reaching it
    - Penalty for collisions with obstacles
    - Navigation requires coordination between robots
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_type_b_robots: int = 2,
        spawn_radius: float = 2.0,
        num_obstacles: int = 8,
        obstacle_types: List[str] = None,
        arena_size: float = 20.0,
        target_radius: float = 1.0,
        **kwargs
    ):
        """
        Initialize the Obstacle Navigation Environment.
        
        Args:
            render_mode: Render mode ('human', 'rgb_array', or None)
            num_type_b_robots: Number of Type B robots (N). Type A will be 2N
            spawn_radius: Radius of circular area for random robot spawning
            num_obstacles: Number of obstacles to spawn
            obstacle_types: List of obstacle types ('wall', 'box', 'cylinder')
            arena_size: Size of the square arena
            target_radius: Radius of target zones
            **kwargs: Additional arguments passed to BaseRobotEnv
        """
        self.num_type_b = num_type_b_robots
        self.num_type_a = 2 * num_type_b_robots
        self.total_robots = self.num_type_a + self.num_type_b
        self.spawn_radius = spawn_radius
        self.num_obstacles = num_obstacles
        self.obstacle_types = obstacle_types or ['wall', 'box', 'cylinder']
        self.arena_size = arena_size
        self.target_radius = target_radius
        
        # Robot tracking
        self.type_a_robots = []
        self.type_b_robots = []
        self.all_robot_ids = []
        self.connections = []
        self.connection_distance = 0.2
        
        # Physics parameters
        self.type_a_mass = 1.1
        self.type_b_mass = 1.0
        self.gravity = 9.81
        self.type_a_joint_force = 2.0 * self.type_a_mass * self.gravity
        
        # Get path to URDF models
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        self.type_a_urdf = os.path.join(self.models_dir, 'bar_with_joint.urdf')
        self.type_b_urdf = os.path.join(self.models_dir, 'rolling_sphere.urdf')
        
        # Obstacle and target tracking
        self.obstacle_ids = []
        self.target_id = None
        self.target_position = np.array([0.0, 0.0, 0.5])
        self.previous_distance_to_target = None
        self.targets_reached = 0
        self.collision_count = 0
        
        # Initialize base environment
        super().__init__(render_mode=render_mode, **kwargs)
        
        # Action space
        action_dim = self.num_type_a * 6 + self.num_type_b * 3
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Observation space includes target position
        # For Type A: 19 values per robot
        # For Type B: 13 values per robot
        # Plus: target position (3) and vector to target from each robot (3 * total_robots)
        obs_dim = self.num_type_a * 19 + self.num_type_b * 13 + 3 + 3 * self.total_robots
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.type_b_force_magnitude = 20.0
        
    def _load_robot(self) -> int:
        """
        Load multiple robots, obstacles, and target into the simulation.
        
        Returns:
            ID of the first robot
        """
        self.type_a_robots = []
        self.type_b_robots = []
        self.all_robot_ids = []
        self.connections = []
        self.obstacle_ids = []
        
        # Load robots (similar to previous environments)
        positions = self._generate_random_positions(self.total_robots)
        pos_idx = 0
        
        # Load Type A robots
        for i in range(self.num_type_a):
            x, y = positions[pos_idx]
            pos_idx += 1
            
            yaw = np.random.uniform(0, 2 * np.pi)
            orientation = p.getQuaternionFromEuler([0, 0, yaw])
            
            robot_id = p.loadURDF(
                self.type_a_urdf,
                basePosition=[x, y, 0.5],
                baseOrientation=orientation,
                useFixedBase=False
            )
            
            num_joints = p.getNumJoints(robot_id)
            joint_indices = []
            
            for joint_idx in range(num_joints):
                joint_info = p.getJointInfo(robot_id, joint_idx)
                joint_type = joint_info[2]
                
                if joint_type == p.JOINT_SPHERICAL:
                    joint_indices.append(joint_idx)
                    p.setJointMotorControlMultiDof(
                        robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=[0, 0, 0, 1],
                        force=[self.type_a_joint_force] * 3
                    )
            
            self.type_a_robots.append((robot_id, joint_indices))
            self.all_robot_ids.append(robot_id)
        
        # Load Type B robots
        for i in range(self.num_type_b):
            x, y = positions[pos_idx]
            pos_idx += 1
            
            robot_id = p.loadURDF(
                self.type_b_urdf,
                basePosition=[x, y, 0.5],
                useFixedBase=False
            )
            
            self.type_b_robots.append(robot_id)
            self.all_robot_ids.append(robot_id)
        
        # Create obstacles
        self._create_obstacles()
        
        # Create target
        self._create_target()
        
        return self.all_robot_ids[0] if self.all_robot_ids else None
    
    def _create_obstacles(self):
        """Create random obstacles in the arena."""
        for i in range(self.num_obstacles):
            obstacle_type = np.random.choice(self.obstacle_types)
            
            # Random position avoiding spawn area
            valid_position = False
            attempts = 0
            max_attempts = 50
            
            while not valid_position and attempts < max_attempts:
                x = np.random.uniform(-self.arena_size/2 + 2, self.arena_size/2 - 2)
                y = np.random.uniform(-self.arena_size/2 + 2, self.arena_size/2 - 2)
                
                # Check distance from spawn area (origin)
                if np.sqrt(x**2 + y**2) > self.spawn_radius + 2.0:
                    valid_position = True
                
                attempts += 1
            
            if not valid_position:
                continue
            
            if obstacle_type == 'wall':
                # Vertical wall
                height = np.random.uniform(1.0, 3.0)
                width = np.random.uniform(2.0, 5.0)
                thickness = 0.2
                
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[width/2, thickness/2, height/2]
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[width/2, thickness/2, height/2],
                    rgbaColor=[0.6, 0.6, 0.6, 1.0]
                )
                
                yaw = np.random.uniform(0, 2 * np.pi)
                orientation = p.getQuaternionFromEuler([0, 0, yaw])
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,  # Static obstacle
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x, y, height/2],
                    baseOrientation=orientation
                )
                
            elif obstacle_type == 'box':
                # Box obstacle
                size = np.random.uniform(0.5, 1.5)
                
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[size/2, size/2, size/2]
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[size/2, size/2, size/2],
                    rgbaColor=[0.5, 0.3, 0.1, 1.0]
                )
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x, y, size/2]
                )
                
            else:  # cylinder
                # Cylindrical pillar
                radius = np.random.uniform(0.3, 0.8)
                height = np.random.uniform(1.0, 3.0)
                
                collision_shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    height=height
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=[0.4, 0.4, 0.5, 1.0]
                )
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x, y, height/2]
                )
            
            self.obstacle_ids.append(obstacle_id)
    
    def _create_target(self):
        """Create a target zone for robots to reach."""
        # Random target position away from spawn
        valid_target = False
        attempts = 0
        
        while not valid_target and attempts < 100:
            x = np.random.uniform(-self.arena_size/2 + 3, self.arena_size/2 - 3)
            y = np.random.uniform(-self.arena_size/2 + 3, self.arena_size/2 - 3)
            
            # Check distance from spawn
            dist_from_spawn = np.sqrt(x**2 + y**2)
            if dist_from_spawn > 5.0:  # At least 5m from spawn
                # Check distance from obstacles
                too_close = False
                for obs_id in self.obstacle_ids:
                    obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
                    dist = np.sqrt((x - obs_pos[0])**2 + (y - obs_pos[1])**2)
                    if dist < 2.0:
                        too_close = True
                        break
                
                if not too_close:
                    valid_target = True
            
            attempts += 1
        
        if not valid_target:
            # Fallback position
            x, y = 8.0, 8.0
        
        self.target_position = np.array([x, y, 0.5])
        
        # Create visual marker for target (semi-transparent sphere)
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.target_radius,
            rgbaColor=[0.0, 1.0, 0.0, 0.3]
        )
        
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_position.tolist()
        )
    
    def _generate_random_positions(self, num_positions: int) -> List[Tuple[float, float]]:
        """Generate random spawn positions for robots."""
        positions = []
        min_separation = 0.4
        max_attempts = 100
        
        for _ in range(num_positions):
            attempts = 0
            while attempts < max_attempts:
                r = np.random.uniform(0, self.spawn_radius)
                theta = np.random.uniform(0, 2 * np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
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
            
            if attempts == max_attempts:
                positions.append((x, y))
        
        return positions
    
    def _get_observation(self) -> np.ndarray:
        """Get observation including robot states and target information."""
        observations = []
        
        # Type A robots
        for robot_id, joint_indices in self.type_a_robots:
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(robot_id)
            
            joint_states = []
            for joint_idx in joint_indices:
                joint_state = p.getJointStateMultiDof(robot_id, joint_idx)
                joint_pos = joint_state[0]
                joint_states.extend(joint_pos[:3])
            
            robot_obs = np.array([
                pos[0], pos[1], pos[2],
                orn[0], orn[1], orn[2], orn[3],
                lin_vel[0], lin_vel[1], lin_vel[2],
                ang_vel[0], ang_vel[1], ang_vel[2],
                *joint_states
            ], dtype=np.float32)
            
            observations.append(robot_obs)
        
        # Type B robots
        for robot_id in self.type_b_robots:
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(robot_id)
            
            robot_obs = np.array([
                pos[0], pos[1], pos[2],
                orn[0], orn[1], orn[2], orn[3],
                lin_vel[0], lin_vel[1], lin_vel[2],
                ang_vel[0], ang_vel[1], ang_vel[2]
            ], dtype=np.float32)
            
            observations.append(robot_obs)
        
        # Target position (global)
        target_obs = self.target_position.copy()
        
        # Vector to target from each robot
        target_vectors = []
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            vec_to_target = self.target_position - np.array(pos)
            target_vectors.extend(vec_to_target.tolist())
        
        # Combine all observations
        all_obs = np.concatenate([
            np.concatenate(observations),
            target_obs,
            np.array(target_vectors, dtype=np.float32)
        ])
        
        return all_obs
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on navigation progress and target reaching.
        
        Reward components:
        - Progress toward target
        - Reaching target zone
        - Penalty for collisions with obstacles
        - Staying within arena bounds
        """
        reward = 0.0
        
        # Calculate minimum distance from any robot to target
        min_distance = float('inf')
        closest_robot_pos = None
        
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            distance = np.linalg.norm(self.target_position - np.array(pos))
            if distance < min_distance:
                min_distance = distance
                closest_robot_pos = np.array(pos)
        
        # Reward for progress toward target
        if self.previous_distance_to_target is not None:
            progress = self.previous_distance_to_target - min_distance
            reward += progress * 2.0  # Scale progress reward
        
        self.previous_distance_to_target = min_distance
        
        # Large reward for reaching target
        if min_distance < self.target_radius:
            reward += 100.0
            self.targets_reached += 1
            # Reposition target
            self._reposition_target()
        
        # Small penalty for being far from target (encourages exploration)
        reward -= min_distance * 0.01
        
        # Penalty for collisions with obstacles
        collision_penalty = self._check_collisions()
        reward -= collision_penalty
        
        # Small step reward to encourage longer episodes
        reward += 0.1
        
        return reward
    
    def _check_collisions(self) -> float:
        """Check for collisions with obstacles and return penalty."""
        penalty = 0.0
        
        for robot_id in self.all_robot_ids:
            for obs_id in self.obstacle_ids:
                contacts = p.getContactPoints(bodyA=robot_id, bodyB=obs_id)
                if len(contacts) > 0:
                    penalty += 1.0  # Penalty per collision
                    self.collision_count += 1
        
        return penalty
    
    def _reposition_target(self):
        """Move target to a new random position after being reached."""
        valid_target = False
        attempts = 0
        
        while not valid_target and attempts < 100:
            x = np.random.uniform(-self.arena_size/2 + 3, self.arena_size/2 - 3)
            y = np.random.uniform(-self.arena_size/2 + 3, self.arena_size/2 - 3)
            
            # Check distance from current robot positions
            min_robot_dist = float('inf')
            for robot_id in self.all_robot_ids:
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                min_robot_dist = min(min_robot_dist, dist)
            
            if min_robot_dist > 3.0:
                valid_target = True
            
            attempts += 1
        
        self.target_position = np.array([x, y, 0.5])
        
        # Update target visual position
        p.resetBasePositionAndOrientation(
            self.target_id,
            self.target_position.tolist(),
            [0, 0, 0, 1]
        )
        
        # Reset distance tracking
        self.previous_distance_to_target = None
    
    def _is_done(self) -> bool:
        """Check termination conditions."""
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            
            # Robot fell below ground
            if pos[2] < -1.0:
                return True
            
            # Robot left arena
            if abs(pos[0]) > self.arena_size/2 or abs(pos[1]) > self.arena_size/2:
                return True
        
        return False
    
    def _apply_action(self, action: np.ndarray):
        """Apply actions to all robots (same as previous environments)."""
        action_idx = 0
        
        # Type A robots
        for robot_id, joint_indices in self.type_a_robots:
            for joint_idx in joint_indices:
                joint_action = action[action_idx:action_idx + 3]
                action_idx += 3
                
                target_angles = joint_action * np.pi
                quat = p.getQuaternionFromEuler(target_angles.tolist())
                
                p.setJointMotorControlMultiDof(
                    robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=quat,
                    force=[self.type_a_joint_force] * 3,
                    positionGain=0.3,
                    velocityGain=0.1
                )
        
        # Type B robots
        for robot_id in self.type_b_robots:
            force_action = action[action_idx:action_idx + 3]
            action_idx += 3
            
            force = force_action * self.type_b_force_magnitude
            
            p.applyExternalForce(
                robot_id,
                -1,
                forceObj=force.tolist(),
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME
            )
    
    def _check_and_create_connections(self):
        """Check and create connections between Type A endpoints and Type B spheres."""
        for type_a_idx, (robot_id, joint_indices) in enumerate(self.type_a_robots):
            num_links = p.getNumJoints(robot_id)
            
            for link_idx in range(1, num_links + 1):
                try:
                    endpoint_pos = self._get_bar_endpoint_position(robot_id, link_idx)
                except:
                    continue
                
                for type_b_idx, type_b_id in enumerate(self.type_b_robots):
                    type_b_pos, _ = p.getBasePositionAndOrientation(type_b_id)
                    type_b_pos = np.array(type_b_pos)
                    
                    distance = np.linalg.norm(endpoint_pos - type_b_pos)
                    
                    if distance < self.connection_distance:
                        already_connected = False
                        for conn in self.connections:
                            if (conn[1] == type_a_idx and conn[2] == type_b_idx and conn[3] == link_idx):
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
        """Remove connections where robots have moved too far apart."""
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
        
        for i in reversed(connections_to_remove):
            constraint_id = self.connections[i][0]
            p.removeConstraint(constraint_id)
            del self.connections[i]
    
    def _get_bar_endpoint_position(self, robot_id: int, link_idx: int) -> np.ndarray:
        """Get world position of a bar endpoint."""
        link_state = p.getLinkState(robot_id, link_idx)
        link_pos = np.array(link_state[0])
        link_orn = link_state[1]
        
        local_endpoint_offset = [0.5, 0, 0]
        
        rotation_matrix = p.getMatrixFromQuaternion(link_orn)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        
        endpoint_offset = rotation_matrix @ local_endpoint_offset
        endpoint_pos = link_pos + endpoint_offset
        
        return endpoint_pos
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset tracking variables
        self.previous_distance_to_target = None
        self.targets_reached = 0
        self.collision_count = 0
        
        # Clear connections
        for constraint_id, _, _, _ in self.connections:
            p.removeConstraint(constraint_id)
        self.connections = []
        
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        info['training_session'] = 3
        info['session_goal'] = 'navigate_obstacles'
        info['targets_reached'] = self.targets_reached
        info['num_obstacles'] = len(self.obstacle_ids)
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Check and manage connections
        self._check_and_create_connections()
        self._remove_weak_connections()
        
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Add session-specific info
        info['training_session'] = 3
        info['targets_reached'] = self.targets_reached
        info['collision_count'] = self.collision_count
        info['distance_to_target'] = self.previous_distance_to_target
        info['target_position'] = self.target_position.tolist()
        info['num_connections'] = len(self.connections)
        
        return observation, reward, terminated, truncated, info
