"""
Training Session 6: Narrow Passage Environment

This is the SIXTH training session in a series of training sessions.
Goal: Navigate through narrow passages and tight spaces requiring precision.
- Type A: Bar robot with joints (bar_with_joint.urdf) - 2N robots
- Type B: Sphere robot (rolling_sphere.urdf) - N robots
- Type A endpoints can connect to Type B robots
- Narrow corridors, tight turns, and constrained spaces
- Requires precise control and sometimes reconfiguration
- Reward based on successfully navigating passages without collisions
- Teaches fine motor control and spatial awareness
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import os
from typing import Optional, Dict, Tuple, List
from .base_robot_env import BaseRobotEnv


class NarrowPassageEnv(BaseRobotEnv):
    """
    Training Session 6: Narrow Passage Environment
    
    The objective is to train agents to navigate through narrow passages.
    This environment supports the same robots as previous sessions:
    - Type A: Bar robots with joints (2N robots)
    - Type B: Sphere robots (N robots)
    
    Features:
    - Series of narrow corridors and tight spaces
    - Checkpoints that must be passed through
    - Penalty for colliding with walls
    - Reward for passing checkpoints and reaching the end
    - Requires precise movement and possibly shape adaptation
    - Tests spatial awareness and fine control
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_type_b_robots: int = 2,
        spawn_radius: float = 1.0,
        num_passages: int = 4,
        passage_width: float = 2.0,  # Width of passages
        passage_difficulty: float = 0.5,  # 0.0 to 1.0, affects narrowness
        **kwargs
    ):
        """
        Initialize the Narrow Passage Environment.
        
        Args:
            render_mode: Render mode ('human', 'rgb_array', or None)
            num_type_b_robots: Number of Type B robots (N). Type A will be 2N
            spawn_radius: Radius of circular area for random robot spawning
            num_passages: Number of passage sections to navigate
            passage_width: Base width of passages (gets narrower with difficulty)
            passage_difficulty: Difficulty level (0-1), affects passage narrowness
            **kwargs: Additional arguments passed to BaseRobotEnv
        """
        self.num_type_b = num_type_b_robots
        self.num_type_a = 2 * num_type_b_robots
        self.total_robots = self.num_type_a + self.num_type_b
        self.spawn_radius = spawn_radius
        self.num_passages = num_passages
        self.base_passage_width = passage_width
        self.passage_difficulty = np.clip(passage_difficulty, 0.0, 1.0)
        
        # Calculate effective passage width based on difficulty
        # At difficulty 0: full width, at difficulty 1: 40% of width
        self.passage_width = passage_width * (1.0 - 0.6 * self.passage_difficulty)
        
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
        
        # Passage tracking
        self.wall_ids = []
        self.checkpoint_ids = []
        self.checkpoint_positions = []
        self.checkpoints_passed = []
        self.collision_count = 0
        self.max_checkpoint_reached = -1
        
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
        
        # Observation space includes checkpoint information
        # Robot observations + next checkpoint position (3) + distance to checkpoint (1) + checkpoints passed (num_passages)
        robot_obs_dim = self.num_type_a * 19 + self.num_type_b * 13
        checkpoint_obs_dim = 3 + 1 + num_passages
        obs_dim = robot_obs_dim + checkpoint_obs_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.type_b_force_magnitude = 20.0
        
    def _load_robot(self) -> int:
        """
        Load robots and create passage maze.
        
        Returns:
            ID of the first robot
        """
        self.type_a_robots = []
        self.type_b_robots = []
        self.all_robot_ids = []
        self.connections = []
        self.wall_ids = []
        self.checkpoint_ids = []
        self.checkpoint_positions = []
        self.checkpoints_passed = [False] * self.num_passages
        
        # Create passage maze first
        self._create_passage_maze()
        
        # Load robots at starting area
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
        
        return self.all_robot_ids[0] if self.all_robot_ids else None
    
    def _create_passage_maze(self):
        """Create a maze of narrow passages with checkpoints."""
        # Starting area (open space)
        self._create_room(0.0, 0.0, 5.0, 5.0)
        
        current_x = 5.0
        current_y = 0.0
        
        passage_types = ['straight', 'turn_left', 'turn_right', 's_curve', 'zigzag']
        
        for i in range(self.num_passages):
            passage_type = np.random.choice(passage_types)
            
            if passage_type == 'straight':
                length = 4.0 + np.random.uniform(0, 2.0)
                next_x, next_y = self._create_straight_passage(current_x, current_y, length, 0.0)
                checkpoint_pos = np.array([current_x + length/2, current_y, 0.5])
                
            elif passage_type == 'turn_left':
                length = 3.0
                next_x, next_y = self._create_turn_passage(current_x, current_y, length, left=True)
                checkpoint_pos = np.array([current_x + length, current_y + length, 0.5])
                
            elif passage_type == 'turn_right':
                length = 3.0
                next_x, next_y = self._create_turn_passage(current_x, current_y, length, left=False)
                checkpoint_pos = np.array([current_x + length, current_y - length, 0.5])
                
            elif passage_type == 's_curve':
                length = 5.0
                next_x, next_y = self._create_s_curve_passage(current_x, current_y, length)
                checkpoint_pos = np.array([current_x + length, current_y, 0.5])
                
            else:  # zigzag
                length = 4.0
                next_x, next_y = self._create_zigzag_passage(current_x, current_y, length)
                checkpoint_pos = np.array([current_x + length, current_y, 0.5])
            
            # Create checkpoint marker
            self._create_checkpoint(checkpoint_pos)
            self.checkpoint_positions.append(checkpoint_pos)
            
            current_x = next_x
            current_y = next_y
        
        # Final goal area
        self._create_room(current_x, current_y, 4.0, 4.0)
        final_checkpoint = np.array([current_x + 2.0, current_y, 0.5])
        self._create_checkpoint(final_checkpoint, is_final=True)
        self.checkpoint_positions.append(final_checkpoint)
    
    def _create_room(self, x_center: float, y_center: float, width: float, length: float):
        """Create an open room area."""
        wall_thickness = 0.2
        wall_height = 2.0
        
        # Create walls around perimeter
        # Bottom wall
        self._create_wall(x_center, y_center - width/2, length, wall_thickness, wall_height)
        # Top wall
        self._create_wall(x_center, y_center + width/2, length, wall_thickness, wall_height)
        # Left wall
        self._create_wall(x_center - length/2, y_center, wall_thickness, width, wall_height)
        # Right wall
        self._create_wall(x_center + length/2, y_center, wall_thickness, width, wall_height)
    
    def _create_straight_passage(self, x_start: float, y_center: float, length: float, angle: float) -> Tuple[float, float]:
        """Create a straight narrow passage."""
        wall_thickness = 0.2
        wall_height = 2.0
        
        # Side walls
        self._create_wall(x_start + length/2, y_center - self.passage_width/2, length, wall_thickness, wall_height)
        self._create_wall(x_start + length/2, y_center + self.passage_width/2, length, wall_thickness, wall_height)
        
        return x_start + length, y_center
    
    def _create_turn_passage(self, x_start: float, y_start: float, length: float, left: bool) -> Tuple[float, float]:
        """Create a 90-degree turn passage."""
        wall_thickness = 0.2
        wall_height = 2.0
        direction = 1.0 if left else -1.0
        
        # Create walls for L-shaped turn
        # Horizontal section
        self._create_wall(x_start + length/2, y_start - direction * self.passage_width/2, length, wall_thickness, wall_height)
        
        # Vertical section
        self._create_wall(x_start + length - self.passage_width/2, y_start + direction * length/2, wall_thickness, length, wall_height)
        
        # Outer corner
        self._create_wall(x_start + length + self.passage_width/2, y_start + direction * length/2, wall_thickness, length, wall_height)
        self._create_wall(x_start + length/2, y_start + direction * length + self.passage_width/2, length, wall_thickness, wall_height)
        
        return x_start + length, y_start + direction * length
    
    def _create_s_curve_passage(self, x_start: float, y_center: float, length: float) -> Tuple[float, float]:
        """Create an S-curve passage."""
        wall_thickness = 0.2
        wall_height = 2.0
        curve_amplitude = 2.0
        
        # Create walls along S-curve (simplified as segments)
        num_segments = 6
        segment_length = length / num_segments
        
        for i in range(num_segments):
            x = x_start + i * segment_length
            y_offset = curve_amplitude * np.sin(2 * np.pi * i / num_segments)
            
            self._create_wall(x + segment_length/2, y_center + y_offset - self.passage_width/2, segment_length, wall_thickness, wall_height)
            self._create_wall(x + segment_length/2, y_center + y_offset + self.passage_width/2, segment_length, wall_thickness, wall_height)
        
        return x_start + length, y_center
    
    def _create_zigzag_passage(self, x_start: float, y_center: float, length: float) -> Tuple[float, float]:
        """Create a zigzag passage."""
        wall_thickness = 0.2
        wall_height = 2.0
        num_zigs = 3
        zig_length = length / num_zigs
        zig_offset = 1.5
        
        current_y = y_center
        for i in range(num_zigs):
            x = x_start + i * zig_length
            offset = zig_offset if i % 2 == 0 else -zig_offset
            
            # Create angled walls
            self._create_wall(x + zig_length/2, current_y + offset/2 - self.passage_width/2, zig_length, wall_thickness, wall_height)
            self._create_wall(x + zig_length/2, current_y + offset/2 + self.passage_width/2, zig_length, wall_thickness, wall_height)
            
            current_y += offset
        
        return x_start + length, current_y
    
    def _create_wall(self, x: float, y: float, length: float, thickness: float, height: float):
        """Create a wall obstacle."""
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length/2, thickness/2, height/2]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length/2, thickness/2, height/2],
            rgbaColor=[0.5, 0.5, 0.5, 1.0]
        )
        
        wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, height/2]
        )
        
        p.changeDynamics(wall_id, -1, lateralFriction=0.8)
        self.wall_ids.append(wall_id)
    
    def _create_checkpoint(self, position: np.ndarray, is_final: bool = False):
        """Create a checkpoint marker (visual indicator)."""
        color = [0.0, 1.0, 0.0, 0.4] if is_final else [0.0, 0.5, 1.0, 0.3]
        
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.5,
            length=2.0,
            rgbaColor=color
        )
        
        checkpoint_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position.tolist()
        )
        
        self.checkpoint_ids.append(checkpoint_id)
    
    def _generate_random_positions(self, num_positions: int) -> List[Tuple[float, float]]:
        """Generate random spawn positions for robots in starting area."""
        positions = []
        min_separation = 0.3
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
        """Get observation including robot states and checkpoint information."""
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
        
        # Get next checkpoint to reach
        next_checkpoint_idx = self.max_checkpoint_reached + 1
        if next_checkpoint_idx < len(self.checkpoint_positions):
            next_checkpoint = self.checkpoint_positions[next_checkpoint_idx]
        else:
            next_checkpoint = self.checkpoint_positions[-1]  # Final checkpoint
        
        # Calculate distance to next checkpoint
        center_pos = np.mean([p.getBasePositionAndOrientation(rid)[0] for rid in self.all_robot_ids], axis=0)
        distance_to_checkpoint = np.linalg.norm(next_checkpoint - center_pos)
        
        # Checkpoints passed encoding
        checkpoints_encoding = np.array([1.0 if passed else 0.0 for passed in self.checkpoints_passed], dtype=np.float32)
        
        # Combine all observations
        all_obs = np.concatenate([
            np.concatenate(observations),
            next_checkpoint,
            np.array([distance_to_checkpoint], dtype=np.float32),
            checkpoints_encoding
        ])
        
        return all_obs
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on checkpoint progress and collision avoidance.
        
        Reward components:
        - Large reward for passing checkpoints
        - Progress toward next checkpoint
        - Penalty for wall collisions
        - Bonus for completing all checkpoints
        """
        reward = 0.0
        
        # Calculate center position of robots
        center_pos = np.mean([p.getBasePositionAndOrientation(rid)[0] for rid in self.all_robot_ids], axis=0)
        
        # Check if any new checkpoints have been passed
        for i, checkpoint_pos in enumerate(self.checkpoint_positions):
            if not self.checkpoints_passed[i] if i < len(self.checkpoints_passed) else False:
                distance = np.linalg.norm(checkpoint_pos - center_pos)
                
                if distance < 1.0:  # Within checkpoint radius
                    self.checkpoints_passed[i] = True
                    self.max_checkpoint_reached = i
                    reward += 100.0  # Large reward for passing checkpoint
        
        # Reward for progress toward next checkpoint
        next_checkpoint_idx = self.max_checkpoint_reached + 1
        if next_checkpoint_idx < len(self.checkpoint_positions):
            next_checkpoint = self.checkpoint_positions[next_checkpoint_idx]
            distance_to_next = np.linalg.norm(next_checkpoint - center_pos)
            reward -= distance_to_next * 0.05  # Small penalty for distance
        
        # Check for collisions with walls
        collision_penalty = 0.0
        for robot_id in self.all_robot_ids:
            for wall_id in self.wall_ids:
                contacts = p.getContactPoints(bodyA=robot_id, bodyB=wall_id)
                if len(contacts) > 0:
                    collision_penalty += 2.0  # Significant penalty for collision
                    self.collision_count += 1
        
        reward -= collision_penalty
        
        # Bonus for completing all checkpoints
        if all(self.checkpoints_passed):
            reward += 500.0
        
        # Small step reward
        reward += 0.1
        
        return reward
    
    def _is_done(self) -> bool:
        """Check termination conditions."""
        # Success: all checkpoints passed
        if all(self.checkpoints_passed):
            return True
        
        # Check if any robot fell or went out of bounds
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            
            if pos[2] < -1.0:
                return True
            
            if abs(pos[0]) > 100.0 or abs(pos[1]) > 100.0:  # Very far out
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
        self.checkpoints_passed = [False] * self.num_passages
        self.collision_count = 0
        self.max_checkpoint_reached = -1
        
        # Clear connections
        for constraint_id, _, _, _ in self.connections:
            p.removeConstraint(constraint_id)
        self.connections = []
        
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        info['training_session'] = 6
        info['session_goal'] = 'navigate_passages'
        info['num_passages'] = self.num_passages
        info['passage_width'] = self.passage_width
        info['passage_difficulty'] = self.passage_difficulty
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Check and manage connections
        self._check_and_create_connections()
        self._remove_weak_connections()
        
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Add session-specific info
        info['training_session'] = 6
        info['checkpoints_passed'] = sum(self.checkpoints_passed)
        info['total_checkpoints'] = len(self.checkpoints_passed)
        info['collision_count'] = self.collision_count
        info['max_checkpoint_reached'] = self.max_checkpoint_reached
        info['num_connections'] = len(self.connections)
        info['completion_percentage'] = (sum(self.checkpoints_passed) / len(self.checkpoints_passed)) * 100.0
        
        return observation, reward, terminated, truncated, info
