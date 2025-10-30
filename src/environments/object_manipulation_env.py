"""
Training Session 4: Object Manipulation Environment

This is the FOURTH training session in a series of training sessions.
Goal: Push/pull objects to target locations.
- Type A: Bar robot with joints (bar_with_joint.urdf) - 2N robots
- Type B: Sphere robot (rolling_sphere.urdf) - N robots
- Type A endpoints can connect to Type B robots
- Various objects (boxes, cylinders) spawn in environment
- Target zones indicate where objects should be moved
- Reward based on successfully moving objects to targets
- Teaches robots to apply coordinated forces to manipulate objects
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import os
from typing import Optional, Dict, Tuple, List
from .base_robot_env import BaseRobotEnv


class ObjectManipulationEnv(BaseRobotEnv):
    """
    Training Session 4: Object Manipulation Environment
    
    The objective is to train agents to push/pull objects to target locations.
    This environment supports the same robots as previous sessions:
    - Type A: Bar robots with joints (2N robots)
    - Type B: Sphere robots (N robots)
    
    Features:
    - Multiple objects that need to be moved
    - Target zones where objects should be placed
    - Reward for moving objects closer to targets
    - Large reward for placing objects in target zones
    - Objects have varying mass and friction properties
    - Requires coordinated pushing/pulling strategies
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_type_b_robots: int = 2,
        spawn_radius: float = 2.0,
        num_objects: int = 3,
        arena_size: float = 15.0,
        target_radius: float = 0.8,
        **kwargs
    ):
        """
        Initialize the Object Manipulation Environment.
        
        Args:
            render_mode: Render mode ('human', 'rgb_array', or None)
            num_type_b_robots: Number of Type B robots (N). Type A will be 2N
            spawn_radius: Radius of circular area for random robot spawning
            num_objects: Number of objects to manipulate
            arena_size: Size of the square arena
            target_radius: Radius of target zones
            **kwargs: Additional arguments passed to BaseRobotEnv
        """
        self.num_type_b = num_type_b_robots
        self.num_type_a = 2 * num_type_b_robots
        self.total_robots = self.num_type_a + self.num_type_b
        self.spawn_radius = spawn_radius
        self.num_objects = num_objects
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
        
        # Object and target tracking
        self.objects = []  # List of (object_id, target_id, target_position, initial_position)
        self.previous_object_distances = []
        self.objects_in_target = 0
        self.total_push_distance = 0.0
        
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
        
        # Observation space includes object and target information
        # Robot observations + object positions (3 per object) + target positions (3 per object) + vectors (3 per object)
        robot_obs_dim = self.num_type_a * 19 + self.num_type_b * 13
        object_obs_dim = num_objects * (3 + 3 + 3)  # position, target, vector
        obs_dim = robot_obs_dim + object_obs_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.type_b_force_magnitude = 20.0
        
    def _load_robot(self) -> int:
        """
        Load robots and objects into the simulation.
        
        Returns:
            ID of the first robot
        """
        self.type_a_robots = []
        self.type_b_robots = []
        self.all_robot_ids = []
        self.connections = []
        self.objects = []
        
        # Load robots
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
        
        # Create objects and targets
        self._create_objects_and_targets()
        
        return self.all_robot_ids[0] if self.all_robot_ids else None
    
    def _create_objects_and_targets(self):
        """Create objects to manipulate and their target zones."""
        for i in range(self.num_objects):
            # Random object position (away from spawn area)
            valid_position = False
            attempts = 0
            
            while not valid_position and attempts < 100:
                x_obj = np.random.uniform(-self.arena_size/2 + 2, self.arena_size/2 - 2)
                y_obj = np.random.uniform(-self.arena_size/2 + 2, self.arena_size/2 - 2)
                
                # Check distance from spawn
                if np.sqrt(x_obj**2 + y_obj**2) > self.spawn_radius + 2.0:
                    valid_position = True
                
                attempts += 1
            
            # Random object type and properties
            object_type = np.random.choice(['box', 'cylinder'])
            mass = np.random.uniform(2.0, 8.0)  # Heavier than robots
            
            if object_type == 'box':
                size = np.random.uniform(0.5, 1.0)
                
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[size/2, size/2, size/2]
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[size/2, size/2, size/2],
                    rgbaColor=[0.8, 0.4, 0.1, 1.0]
                )
                
                object_id = p.createMultiBody(
                    baseMass=mass,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x_obj, y_obj, size/2]
                )
            else:  # cylinder
                radius = np.random.uniform(0.3, 0.6)
                height = np.random.uniform(0.5, 1.0)
                
                collision_shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    height=height
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=[0.6, 0.3, 0.2, 1.0]
                )
                
                object_id = p.createMultiBody(
                    baseMass=mass,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x_obj, y_obj, height/2]
                )
            
            # Set friction properties
            p.changeDynamics(
                object_id,
                -1,
                lateralFriction=np.random.uniform(0.5, 1.5),
                spinningFriction=0.1,
                rollingFriction=0.05
            )
            
            # Create target position (away from object and spawn)
            valid_target = False
            attempts = 0
            
            while not valid_target and attempts < 100:
                x_target = np.random.uniform(-self.arena_size/2 + 2, self.arena_size/2 - 2)
                y_target = np.random.uniform(-self.arena_size/2 + 2, self.arena_size/2 - 2)
                
                # Target should be at least 4m from object
                dist_to_obj = np.sqrt((x_target - x_obj)**2 + (y_target - y_obj)**2)
                
                if dist_to_obj > 4.0:
                    valid_target = True
                
                attempts += 1
            
            target_position = np.array([x_target, y_target, 0.5])
            
            # Create visual marker for target (flat cylinder, semi-transparent)
            target_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=self.target_radius,
                length=0.05,
                rgbaColor=[0.0, 0.8, 0.0, 0.3]
            )
            
            target_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=target_visual,
                basePosition=target_position.tolist()
            )
            
            # Store object info
            initial_position = np.array([x_obj, y_obj, 0.5])
            self.objects.append((object_id, target_id, target_position, initial_position))
            
            # Initialize distance tracking
            self.previous_object_distances.append(dist_to_obj)
    
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
        """Get observation including robot states and object information."""
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
        
        # Object information
        object_observations = []
        for object_id, target_id, target_pos, _ in self.objects:
            obj_pos, _ = p.getBasePositionAndOrientation(object_id)
            obj_pos = np.array(obj_pos)
            
            # Object position
            object_observations.extend(obj_pos.tolist())
            
            # Target position
            object_observations.extend(target_pos.tolist())
            
            # Vector from object to target
            vec_to_target = target_pos - obj_pos
            object_observations.extend(vec_to_target.tolist())
        
        # Combine all observations
        all_obs = np.concatenate([
            np.concatenate(observations),
            np.array(object_observations, dtype=np.float32)
        ])
        
        return all_obs
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on object manipulation progress.
        
        Reward components:
        - Progress in moving objects toward targets
        - Large reward for placing object in target zone
        - Small penalty for distance from objects (encourages engagement)
        - Bonus for coordinated team effort
        """
        reward = 0.0
        objects_in_target = 0
        
        for i, (object_id, target_id, target_pos, initial_pos) in enumerate(self.objects):
            obj_pos, _ = p.getBasePositionAndOrientation(object_id)
            obj_pos = np.array(obj_pos)
            
            # Distance to target
            distance_to_target = np.linalg.norm(target_pos - obj_pos)
            
            # Reward for progress
            if self.previous_object_distances[i] is not None:
                progress = self.previous_object_distances[i] - distance_to_target
                reward += progress * 5.0  # Significant reward for moving object
                
                if progress > 0:
                    self.total_push_distance += progress
            
            self.previous_object_distances[i] = distance_to_target
            
            # Check if object is in target zone
            if distance_to_target < self.target_radius:
                reward += 50.0  # Large reward for placing in target
                objects_in_target += 1
            else:
                # Small penalty proportional to distance
                reward -= distance_to_target * 0.02
        
        self.objects_in_target = objects_in_target
        
        # Bonus if all objects in targets
        if objects_in_target == self.num_objects:
            reward += 200.0
        
        # Encourage robots to be near objects (to interact with them)
        min_robot_to_object_dist = float('inf')
        for robot_id in self.all_robot_ids:
            robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
            robot_pos = np.array(robot_pos)
            
            for object_id, _, _, _ in self.objects:
                obj_pos, _ = p.getBasePositionAndOrientation(object_id)
                obj_pos = np.array(obj_pos)
                
                dist = np.linalg.norm(robot_pos - obj_pos)
                min_robot_to_object_dist = min(min_robot_to_object_dist, dist)
        
        # Small reward for being close to objects
        if min_robot_to_object_dist < 2.0:
            reward += (2.0 - min_robot_to_object_dist) * 0.1
        
        # Small step reward
        reward += 0.1
        
        return reward
    
    def _is_done(self) -> bool:
        """Check termination conditions."""
        # Check if all objects are in target zones
        all_in_target = True
        for object_id, target_id, target_pos, _ in self.objects:
            obj_pos, _ = p.getBasePositionAndOrientation(object_id)
            obj_pos = np.array(obj_pos)
            
            distance = np.linalg.norm(target_pos - obj_pos)
            if distance >= self.target_radius:
                all_in_target = False
                break
        
        if all_in_target:
            return True  # Success termination
        
        # Standard termination checks
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            
            if pos[2] < -1.0:
                return True
            
            if abs(pos[0]) > self.arena_size/2 or abs(pos[1]) > self.arena_size/2:
                return True
        
        return False
    
    def _apply_action(self, action: np.ndarray):
        """Apply actions to all robots."""
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
        self.previous_object_distances = []
        self.objects_in_target = 0
        self.total_push_distance = 0.0
        
        # Clear connections
        for constraint_id, _, _, _ in self.connections:
            p.removeConstraint(constraint_id)
        self.connections = []
        
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        info['training_session'] = 4
        info['session_goal'] = 'manipulate_objects'
        info['num_objects'] = len(self.objects)
        info['objects_in_target'] = self.objects_in_target
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Check and manage connections
        self._check_and_create_connections()
        self._remove_weak_connections()
        
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Add session-specific info
        info['training_session'] = 4
        info['objects_in_target'] = self.objects_in_target
        info['total_push_distance'] = self.total_push_distance
        info['num_connections'] = len(self.connections)
        
        # Calculate object positions for info
        object_positions = []
        for object_id, _, target_pos, _ in self.objects:
            obj_pos, _ = p.getBasePositionAndOrientation(object_id)
            object_positions.append(obj_pos)
        info['object_positions'] = object_positions
        
        return observation, reward, terminated, truncated, info
