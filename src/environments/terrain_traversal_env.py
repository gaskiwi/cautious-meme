"""
Training Session 5: Terrain Traversal Environment

This is the FIFTH training session in a series of training sessions.
Goal: Navigate across challenging terrain (slopes, stairs, rough surfaces).
- Type A: Bar robot with joints (bar_with_joint.urdf) - 2N robots
- Type B: Sphere robot (rolling_sphere.urdf) - N robots
- Type A endpoints can connect to Type B robots
- Procedurally generated terrain with varying difficulty
- Includes slopes, stairs, gaps, and uneven surfaces
- Reward based on forward progress and maintaining stability
- Teaches adaptability to different surface conditions
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import os
from typing import Optional, Dict, Tuple, List
from .base_robot_env import BaseRobotEnv


class TerrainTraversalEnv(BaseRobotEnv):
    """
    Training Session 5: Terrain Traversal Environment
    
    The objective is to train agents to traverse challenging terrain.
    This environment supports the same robots as previous sessions:
    - Type A: Bar robots with joints (2N robots)
    - Type B: Sphere robots (N robots)
    
    Features:
    - Procedurally generated terrain sections
    - Multiple terrain types: slopes, stairs, platforms, rough surfaces
    - Target distance to travel forward
    - Reward for forward progress and stability
    - Penalty for falling or getting stuck
    - Requires adaptive locomotion strategies
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_type_b_robots: int = 2,
        spawn_radius: float = 1.5,
        target_distance: float = 20.0,
        terrain_difficulty: float = 0.5,  # 0.0 to 1.0
        num_terrain_sections: int = 5,
        **kwargs
    ):
        """
        Initialize the Terrain Traversal Environment.
        
        Args:
            render_mode: Render mode ('human', 'rgb_array', or None)
            num_type_b_robots: Number of Type B robots (N). Type A will be 2N
            spawn_radius: Radius of circular area for random robot spawning
            target_distance: Target distance to travel in X direction
            terrain_difficulty: Difficulty level (0-1), affects slope angles and gap sizes
            num_terrain_sections: Number of terrain sections to generate
            **kwargs: Additional arguments passed to BaseRobotEnv
        """
        self.num_type_b = num_type_b_robots
        self.num_type_a = 2 * num_type_b_robots
        self.total_robots = self.num_type_a + self.num_type_b
        self.spawn_radius = spawn_radius
        self.target_distance = target_distance
        self.terrain_difficulty = np.clip(terrain_difficulty, 0.0, 1.0)
        self.num_terrain_sections = num_terrain_sections
        
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
        
        # Terrain tracking
        self.terrain_ids = []
        self.max_forward_distance = 0.0
        self.falls_count = 0
        self.stuck_counter = 0
        self.previous_center_position = None
        
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
        
        # Observation space includes terrain information
        # Robot observations + forward distance (1) + terrain type ahead (num_terrain_sections)
        robot_obs_dim = self.num_type_a * 19 + self.num_type_b * 13
        terrain_obs_dim = 1 + num_terrain_sections  # distance + terrain type encoding
        obs_dim = robot_obs_dim + terrain_obs_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.type_b_force_magnitude = 20.0
        
    def _load_robot(self) -> int:
        """
        Load robots and terrain into the simulation.
        
        Returns:
            ID of the first robot
        """
        self.type_a_robots = []
        self.type_b_robots = []
        self.all_robot_ids = []
        self.connections = []
        self.terrain_ids = []
        
        # Create terrain first
        self._create_terrain()
        
        # Load robots at starting platform
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
                basePosition=[x, y, 1.5],  # Higher start for rough terrain
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
                basePosition=[x, y, 1.5],
                useFixedBase=False
            )
            
            self.type_b_robots.append(robot_id)
            self.all_robot_ids.append(robot_id)
        
        return self.all_robot_ids[0] if self.all_robot_ids else None
    
    def _create_terrain(self):
        """Create procedurally generated terrain sections."""
        current_x = 0.0
        
        # Starting platform
        self._create_flat_platform(current_x, 0.0, width=5.0, length=4.0, height=1.0)
        current_x += 4.0
        
        # Generate terrain sections
        terrain_types = ['slope_up', 'slope_down', 'stairs_up', 'stairs_down', 'rough', 'gap', 'platform']
        
        for i in range(self.num_terrain_sections):
            terrain_type = np.random.choice(terrain_types)
            
            if terrain_type == 'slope_up':
                length = 4.0
                max_angle = 20.0 + (self.terrain_difficulty * 25.0)  # 20-45 degrees
                angle = np.random.uniform(10.0, max_angle)
                self._create_slope(current_x, angle, length, width=5.0)
                current_x += length * np.cos(np.radians(angle))
                
            elif terrain_type == 'slope_down':
                length = 4.0
                max_angle = 20.0 + (self.terrain_difficulty * 25.0)
                angle = np.random.uniform(10.0, max_angle)
                self._create_slope(current_x, -angle, length, width=5.0)
                current_x += length * np.cos(np.radians(angle))
                
            elif terrain_type == 'stairs_up':
                num_steps = int(3 + self.terrain_difficulty * 5)  # 3-8 steps
                step_height = 0.15 + self.terrain_difficulty * 0.15  # 0.15-0.3m
                step_depth = 0.5
                self._create_stairs(current_x, num_steps, step_height, step_depth, width=5.0, ascending=True)
                current_x += num_steps * step_depth
                
            elif terrain_type == 'stairs_down':
                num_steps = int(3 + self.terrain_difficulty * 5)
                step_height = 0.15 + self.terrain_difficulty * 0.15
                step_depth = 0.5
                self._create_stairs(current_x, num_steps, step_height, step_depth, width=5.0, ascending=False)
                current_x += num_steps * step_depth
                
            elif terrain_type == 'rough':
                length = 4.0
                roughness = 0.1 + self.terrain_difficulty * 0.3  # 0.1-0.4m variation
                self._create_rough_terrain(current_x, length, width=5.0, roughness=roughness)
                current_x += length
                
            elif terrain_type == 'gap':
                gap_size = 0.3 + self.terrain_difficulty * 0.7  # 0.3-1.0m gap
                platform_height = 1.0
                self._create_gap(current_x, gap_size, width=5.0, height=platform_height)
                current_x += gap_size
                
            else:  # platform
                length = 3.0
                height = 1.0
                self._create_flat_platform(current_x, 0.0, width=5.0, length=length, height=height)
                current_x += length
        
        # Final platform
        self._create_flat_platform(current_x, 0.0, width=5.0, length=4.0, height=1.0)
    
    def _create_flat_platform(self, x_start: float, y_pos: float, width: float, length: float, height: float):
        """Create a flat platform."""
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length/2, width/2, height/2]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length/2, width/2, height/2],
            rgbaColor=[0.7, 0.7, 0.7, 1.0]
        )
        
        terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x_start + length/2, y_pos, height/2]
        )
        
        p.changeDynamics(terrain_id, -1, lateralFriction=1.0)
        self.terrain_ids.append(terrain_id)
    
    def _create_slope(self, x_start: float, angle_deg: float, length: float, width: float):
        """Create a sloped surface."""
        angle_rad = np.radians(angle_deg)
        
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length/2, width/2, 0.1]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length/2, width/2, 0.1],
            rgbaColor=[0.6, 0.6, 0.5, 1.0]
        )
        
        # Calculate position and orientation
        height_change = length * np.sin(angle_rad)
        x_center = x_start + (length/2) * np.cos(angle_rad)
        z_center = 1.0 + height_change/2
        
        orientation = p.getQuaternionFromEuler([0, angle_rad, 0])
        
        terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x_center, 0.0, z_center],
            baseOrientation=orientation
        )
        
        p.changeDynamics(terrain_id, -1, lateralFriction=1.2)
        self.terrain_ids.append(terrain_id)
    
    def _create_stairs(self, x_start: float, num_steps: int, step_height: float, step_depth: float, width: float, ascending: bool):
        """Create a staircase."""
        current_x = x_start
        current_height = 1.0
        
        for i in range(num_steps):
            if ascending:
                current_height += step_height
            
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[step_depth/2, width/2, current_height/2]
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[step_depth/2, width/2, current_height/2],
                rgbaColor=[0.65, 0.65, 0.6, 1.0]
            )
            
            terrain_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[current_x + step_depth/2, 0.0, current_height/2]
            )
            
            p.changeDynamics(terrain_id, -1, lateralFriction=1.0)
            self.terrain_ids.append(terrain_id)
            
            current_x += step_depth
            
            if not ascending:
                current_height -= step_height
    
    def _create_rough_terrain(self, x_start: float, length: float, width: float, roughness: float):
        """Create rough/uneven terrain with random bumps."""
        num_blocks = int(length / 0.3)  # Blocks every 0.3m
        block_size = length / num_blocks
        
        for i in range(num_blocks):
            for j in range(int(width / 0.5)):
                height_variation = np.random.uniform(-roughness, roughness)
                block_height = max(0.8, 1.0 + height_variation)
                
                x_pos = x_start + i * block_size + block_size/2
                y_pos = -width/2 + j * 0.5 + 0.25
                
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[block_size/2, 0.25, block_height/2]
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[block_size/2, 0.25, block_height/2],
                    rgbaColor=[0.55 + np.random.uniform(-0.1, 0.1), 
                              0.55 + np.random.uniform(-0.1, 0.1), 
                              0.45, 1.0]
                )
                
                terrain_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x_pos, y_pos, block_height/2]
                )
                
                p.changeDynamics(terrain_id, -1, lateralFriction=0.8)
                self.terrain_ids.append(terrain_id)
    
    def _create_gap(self, x_start: float, gap_size: float, width: float, height: float):
        """Create a gap that robots must cross."""
        # Gap is just empty space - platforms are created before and after by other sections
        pass  # The gap is naturally created between sections
    
    def _generate_random_positions(self, num_positions: int) -> List[Tuple[float, float]]:
        """Generate random spawn positions for robots on starting platform."""
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
        """Get observation including robot states and terrain information."""
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
        
        # Calculate center position for forward distance
        center_x = np.mean([p.getBasePositionAndOrientation(rid)[0][0] for rid in self.all_robot_ids])
        forward_distance = center_x
        
        # Terrain encoding (simplified - just use forward distance sections)
        terrain_features = np.zeros(self.num_terrain_sections, dtype=np.float32)
        section_length = self.target_distance / self.num_terrain_sections
        current_section = min(int(forward_distance / section_length), self.num_terrain_sections - 1)
        if current_section >= 0:
            terrain_features[current_section] = 1.0
        
        # Combine all observations
        all_obs = np.concatenate([
            np.concatenate(observations),
            np.array([forward_distance], dtype=np.float32),
            terrain_features
        ])
        
        return all_obs
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on forward progress and stability.
        
        Reward components:
        - Forward progress in X direction
        - Maintaining stability (not flipping or falling)
        - Reaching target distance
        - Penalty for falling off terrain
        """
        reward = 0.0
        
        # Calculate center of mass of all robots
        positions = []
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            positions.append(np.array(pos))
        
        center_pos = np.mean(positions, axis=0)
        
        # Reward for forward progress
        if self.previous_center_position is not None:
            forward_progress = center_pos[0] - self.previous_center_position[0]
            reward += forward_progress * 10.0  # Significant reward for moving forward
            
            # Update max distance
            if center_pos[0] > self.max_forward_distance:
                self.max_forward_distance = center_pos[0]
        
        self.previous_center_position = center_pos.copy()
        
        # Check for stuck condition (no forward progress)
        if self.previous_center_position is not None:
            if abs(center_pos[0] - self.previous_center_position[0]) < 0.01:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        # Penalty for being stuck
        if self.stuck_counter > 50:
            reward -= 0.5
        
        # Reward for stability (staying upright)
        avg_height = np.mean([pos[2] for pos in positions])
        if avg_height > 0.3:  # Robots are above minimum height
            reward += 0.1
        
        # Large reward for reaching target distance
        if center_pos[0] >= self.target_distance:
            reward += 500.0
        
        # Small continuous reward for distance covered
        reward += center_pos[0] * 0.1
        
        # Small step reward
        reward += 0.1
        
        return reward
    
    def _is_done(self) -> bool:
        """Check termination conditions."""
        # Check if target distance reached
        center_x = np.mean([p.getBasePositionAndOrientation(rid)[0][0] for rid in self.all_robot_ids])
        
        if center_x >= self.target_distance:
            return True  # Success!
        
        # Check if any robot fell too far
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            
            if pos[2] < -2.0:  # Fell below terrain
                self.falls_count += 1
                return True
            
            if abs(pos[1]) > 10.0:  # Moved too far sideways
                return True
        
        # Check if stuck for too long
        if self.stuck_counter > 200:
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
        self.max_forward_distance = 0.0
        self.falls_count = 0
        self.stuck_counter = 0
        self.previous_center_position = None
        
        # Clear connections
        for constraint_id, _, _, _ in self.connections:
            p.removeConstraint(constraint_id)
        self.connections = []
        
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        info['training_session'] = 5
        info['session_goal'] = 'traverse_terrain'
        info['target_distance'] = self.target_distance
        info['terrain_difficulty'] = self.terrain_difficulty
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Check and manage connections
        self._check_and_create_connections()
        self._remove_weak_connections()
        
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Calculate current forward distance
        center_x = np.mean([p.getBasePositionAndOrientation(rid)[0][0] for rid in self.all_robot_ids])
        
        # Add session-specific info
        info['training_session'] = 5
        info['forward_distance'] = center_x
        info['max_forward_distance'] = self.max_forward_distance
        info['falls_count'] = self.falls_count
        info['stuck_counter'] = self.stuck_counter
        info['num_connections'] = len(self.connections)
        info['progress_percentage'] = (center_x / self.target_distance) * 100.0
        
        return observation, reward, terminated, truncated, info
