"""
Training Session 3: Object Displacement Environment

This is the THIRD training session in a series of training sessions.
Goal: Displace a randomly shaped object as far as possible in a random direction.
- Type A: Bar robot with joints (bar_with_joint.urdf) - 2N robots
- Type B: Sphere robot (rolling_sphere.urdf) - N robots
- Random object with varying shapes (sphere, cube, cylinder, coffee mug, etc.)
- Object and agents placed randomly on the plane
- Random direction chosen (North, East, South, West)
- Reward based on distance object moves in the chosen direction
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import os
from typing import Optional, Dict, Tuple, List
from .base_robot_env import BaseRobotEnv


class DisplacementEnv(BaseRobotEnv):
    """
    Training Session 3: Object Displacement Environment
    
    The objective is to train agents to push/move an object in a specific direction.
    This environment supports the same robots as previous sessions:
    - Type A: Bar robots with joints (2N robots)
    - Type B: Sphere robots (N robots)
    
    Features:
    - Random object shape each episode (sphere, cube, cylinder, torus, coffee mug, etc.)
    - Random object placement on the plane
    - Random agent placement on the plane
    - Random cardinal direction (N/E/S/W) chosen each episode
    - Reward based on object displacement in the chosen direction
    - Object mass varies by shape for realistic interaction
    """
    
    # Define available object shapes with their parameters
    OBJECT_SHAPES = {
        'sphere': {
            'geom': p.GEOM_SPHERE,
            'params': {'radius': 0.3},
            'mass': 2.0,
            'color': [0.8, 0.2, 0.2, 1.0]
        },
        'cube': {
            'geom': p.GEOM_BOX,
            'params': {'halfExtents': [0.25, 0.25, 0.25]},
            'mass': 2.5,
            'color': [0.2, 0.8, 0.2, 1.0]
        },
        'cylinder': {
            'geom': p.GEOM_CYLINDER,
            'params': {'radius': 0.2, 'height': 0.6},
            'mass': 2.2,
            'color': [0.2, 0.2, 0.8, 1.0]
        },
        'capsule': {
            'geom': p.GEOM_CAPSULE,
            'params': {'radius': 0.15, 'height': 0.5},
            'mass': 1.8,
            'color': [0.8, 0.8, 0.2, 1.0]
        },
        'coffee_mug': {
            'geom': 'mesh',
            'params': {'scale': 2.0},  # Scale up for visibility
            'mass': 1.5,
            'color': [0.6, 0.4, 0.2, 1.0]
        },
        'torus': {
            'geom': 'compound',  # We'll create a torus-like shape with multiple spheres
            'params': {'major_radius': 0.3, 'minor_radius': 0.1, 'segments': 12},
            'mass': 2.0,
            'color': [0.8, 0.2, 0.8, 1.0]
        },
        'rectangular_prism': {
            'geom': p.GEOM_BOX,
            'params': {'halfExtents': [0.4, 0.2, 0.15]},
            'mass': 2.3,
            'color': [0.2, 0.8, 0.8, 1.0]
        },
        'tall_cylinder': {
            'geom': p.GEOM_CYLINDER,
            'params': {'radius': 0.15, 'height': 0.8},
            'mass': 2.0,
            'color': [0.9, 0.5, 0.1, 1.0]
        }
    }
    
    # Cardinal directions
    DIRECTIONS = {
        'north': np.array([0, 1]),   # +Y
        'south': np.array([0, -1]),  # -Y
        'east': np.array([1, 0]),    # +X
        'west': np.array([-1, 0])    # -X
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_type_b_robots: int = 2,  # N
        spawn_radius: float = 4.0,    # Radius for random spawn area
        max_object_distance: float = 15.0,  # Max distance from origin for object spawn
        **kwargs
    ):
        """
        Initialize the Object Displacement Environment.
        
        Args:
            render_mode: Render mode ('human', 'rgb_array', or None)
            num_type_b_robots: Number of Type B robots (N). Type A will be 2N
            spawn_radius: Radius of circular area for random robot spawning
            max_object_distance: Maximum distance from origin for object placement
            **kwargs: Additional arguments passed to BaseRobotEnv
        """
        self.num_type_b = num_type_b_robots
        self.num_type_a = 2 * num_type_b_robots
        self.total_robots = self.num_type_a + self.num_type_b
        self.spawn_radius = spawn_radius
        self.max_object_distance = max_object_distance
        
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
        self.type_a_joint_force = 2.0 * self.type_a_mass * self.gravity  # ~21.6 N
        
        # Get path to URDF models
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        self.type_a_urdf = os.path.join(self.models_dir, 'bar_with_joint.urdf')
        self.type_b_urdf = os.path.join(self.models_dir, 'rolling_sphere.urdf')
        
        # Object tracking
        self.object_id = None
        self.object_shape_name = None
        self.object_initial_pos = None
        self.target_direction_name = None
        self.target_direction_vec = None
        self.max_displacement = 0.0  # Track maximum displacement achieved
        
        # Initialize base environment first
        super().__init__(render_mode=render_mode, **kwargs)
        
        # Action space: same as previous sessions
        # Type A robots: control 2 spherical joints (each has 3 DOF)
        # Type B robots: apply forces (3 DOF)
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
        # Object: position (3), orientation (4), velocity (3), angular_vel (3)
        # Target direction: (2) - unit vector in XY plane
        # Type A: 19 values per robot
        # Type B: 13 values per robot
        # Object: 13 values
        # Direction: 2 values
        obs_dim = self.num_type_a * 19 + self.num_type_b * 13 + 13 + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Force magnitude for Type B robots
        self.type_b_force_magnitude = 20.0
        
    def _create_object(self, shape_name: str, position: List[float]) -> int:
        """
        Create an object with the specified shape at the given position.
        
        Args:
            shape_name: Name of shape from OBJECT_SHAPES
            position: [x, y, z] position for the object
            
        Returns:
            Object body ID
        """
        shape_info = self.OBJECT_SHAPES[shape_name]
        
        if shape_name == 'coffee_mug':
            # Try to create a coffee mug-like shape using compound geometry
            # We'll create it as a cylinder with a handle (simplified)
            # Main body (cylinder)
            body_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.2,
                height=0.4
            )
            body_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.2,
                length=0.4,
                rgbaColor=shape_info['color']
            )
            
            # Create the mug body
            object_id = p.createMultiBody(
                baseMass=shape_info['mass'],
                baseCollisionShapeIndex=body_collision,
                baseVisualShapeIndex=body_visual,
                basePosition=position
            )
            
            # Add handle as a visual-only torus (simplified as box for collision)
            handle_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.05, 0.15, 0.1]
            )
            handle_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.05, 0.15, 0.1],
                rgbaColor=shape_info['color']
            )
            
            # Position handle to the side
            handle_offset = [0.25, 0, 0]
            
            # We can't add links dynamically, so we'll just use the main body
            # The "handle" is conceptual in this simplified version
            
        elif shape_name == 'torus':
            # Create a torus-like shape using a compound of spheres arranged in a circle
            params = shape_info['params']
            major_r = params['major_radius']
            minor_r = params['minor_radius']
            segments = params['segments']
            
            # Create collision and visual shapes for the torus approximation
            # Use a flat cylinder with a hole (approximation)
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=major_r,
                height=minor_r * 2
            )
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=major_r,
                length=minor_r * 2,
                rgbaColor=shape_info['color']
            )
            
            object_id = p.createMultiBody(
                baseMass=shape_info['mass'],
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position
            )
            
        else:
            # Standard geometric shapes
            geom_type = shape_info['geom']
            params = shape_info['params']
            
            if geom_type == p.GEOM_SPHERE:
                collision_shape = p.createCollisionShape(geom_type, radius=params['radius'])
                visual_shape = p.createVisualShape(geom_type, radius=params['radius'], 
                                                   rgbaColor=shape_info['color'])
            elif geom_type == p.GEOM_BOX:
                collision_shape = p.createCollisionShape(geom_type, halfExtents=params['halfExtents'])
                visual_shape = p.createVisualShape(geom_type, halfExtents=params['halfExtents'], 
                                                   rgbaColor=shape_info['color'])
            elif geom_type == p.GEOM_CYLINDER:
                collision_shape = p.createCollisionShape(geom_type, radius=params['radius'], 
                                                        height=params['height'])
                visual_shape = p.createVisualShape(geom_type, radius=params['radius'], 
                                                   length=params['height'], 
                                                   rgbaColor=shape_info['color'])
            elif geom_type == p.GEOM_CAPSULE:
                collision_shape = p.createCollisionShape(geom_type, radius=params['radius'], 
                                                        height=params['height'])
                visual_shape = p.createVisualShape(geom_type, radius=params['radius'], 
                                                   length=params['height'], 
                                                   rgbaColor=shape_info['color'])
            
            object_id = p.createMultiBody(
                baseMass=shape_info['mass'],
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position
            )
        
        # Set friction properties for realistic pushing
        p.changeDynamics(
            object_id,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.1,
            rollingFriction=0.05
        )
        
        return object_id
        
    def _load_robot(self) -> int:
        """
        Load multiple robots (2N Type A, N Type B) and a random object into the simulation.
        Randomly position them on the plane.
        Choose a random target direction.
        
        Returns:
            ID of the first robot (for compatibility with base class)
        """
        self.type_a_robots = []
        self.type_b_robots = []
        self.all_robot_ids = []
        self.connections = []
        
        # Choose random object shape
        self.object_shape_name = np.random.choice(list(self.OBJECT_SHAPES.keys()))
        
        # Choose random target direction
        self.target_direction_name = np.random.choice(list(self.DIRECTIONS.keys()))
        self.target_direction_vec = self.DIRECTIONS[self.target_direction_name]
        
        # Random object position (somewhere on the plane, not too close to origin)
        object_distance = np.random.uniform(2.0, self.max_object_distance)
        object_angle = np.random.uniform(0, 2 * np.pi)
        object_x = object_distance * np.cos(object_angle)
        object_y = object_distance * np.sin(object_angle)
        object_z = 0.5  # Start slightly above ground
        
        self.object_initial_pos = np.array([object_x, object_y, object_z])
        
        # Create the object
        self.object_id = self._create_object(
            self.object_shape_name,
            self.object_initial_pos.tolist()
        )
        
        # Random positions for robots (separate from object)
        # Place robots near the object but with some randomness
        robot_positions = self._generate_random_positions_near_object(
            self.total_robots,
            self.object_initial_pos[:2]  # Only x, y
        )
        
        pos_idx = 0
        
        # Load Type A robots (bar with joints)
        for i in range(self.num_type_a):
            x, y = robot_positions[pos_idx]
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
            x, y = robot_positions[pos_idx]
            pos_idx += 1
            
            robot_id = p.loadURDF(
                self.type_b_urdf,
                basePosition=[x, y, 0.5],  # Start slightly above ground
                useFixedBase=False
            )
            
            self.type_b_robots.append(robot_id)
            self.all_robot_ids.append(robot_id)
        
        # Reset max displacement
        self.max_displacement = 0.0
        
        # Return the first robot ID for base class compatibility
        return self.all_robot_ids[0] if self.all_robot_ids else None
    
    def _generate_random_positions_near_object(
        self, 
        num_positions: int,
        object_pos: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Generate random (x, y) positions near the object.
        Robots spawn in a circle around the object.
        
        Args:
            num_positions: Number of positions to generate
            object_pos: [x, y] position of the object
            
        Returns:
            List of (x, y) tuples
        """
        positions = []
        min_separation = 0.4  # Minimum distance between robots
        robot_spawn_radius = self.spawn_radius  # Radius around object
        max_attempts = 100
        
        for _ in range(num_positions):
            attempts = 0
            while attempts < max_attempts:
                # Random position in circle around object
                r = np.random.uniform(1.0, robot_spawn_radius)
                theta = np.random.uniform(0, 2 * np.pi)
                
                # Position relative to object
                x = object_pos[0] + r * np.cos(theta)
                y = object_pos[1] + r * np.sin(theta)
                
                # Check separation from existing positions
                too_close = False
                for px, py in positions:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < min_separation:
                        too_close = True
                        break
                
                # Also check distance from object
                dist_to_object = np.sqrt((x - object_pos[0])**2 + (y - object_pos[1])**2)
                if dist_to_object < 0.8:  # Don't spawn too close to object
                    too_close = True
                
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
        Get observation from all robots and the object in the environment.
        
        Returns:
            Flattened observation array containing state of all robots, object, and direction
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
        
        # Object state
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.object_id)
        obj_lin_vel, obj_ang_vel = p.getBaseVelocity(self.object_id)
        
        object_obs = np.array([
            obj_pos[0], obj_pos[1], obj_pos[2],           # Position (3)
            obj_orn[0], obj_orn[1], obj_orn[2], obj_orn[3],   # Orientation quaternion (4)
            obj_lin_vel[0], obj_lin_vel[1], obj_lin_vel[2],  # Linear velocity (3)
            obj_ang_vel[0], obj_ang_vel[1], obj_ang_vel[2]   # Angular velocity (3)
        ], dtype=np.float32)
        
        observations.append(object_obs)
        
        # Target direction
        direction_obs = self.target_direction_vec.astype(np.float32)
        observations.append(direction_obs)
        
        # Flatten all observations into single array
        return np.concatenate(observations)
    
    def _compute_displacement(self) -> float:
        """
        Compute the displacement of the object in the target direction.
        
        Returns:
            Displacement distance (can be negative if moved backward)
        """
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        obj_pos_2d = np.array([obj_pos[0], obj_pos[1]])
        initial_pos_2d = self.object_initial_pos[:2]
        
        # Vector from initial position to current position
        displacement_vec = obj_pos_2d - initial_pos_2d
        
        # Project onto target direction
        displacement = np.dot(displacement_vec, self.target_direction_vec)
        
        return displacement
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on object displacement in the target direction.
        
        Training Session 3 Goal: Displace object as far as possible in chosen direction
        - Reward for displacement in target direction
        - Penalty for displacement in opposite direction
        - Small bonus for object velocity in target direction
        
        Returns:
            Reward value
        """
        # Get current displacement
        displacement = self._compute_displacement()
        
        # Track maximum displacement
        if displacement > self.max_displacement:
            self.max_displacement = displacement
        
        # Reward is based on current displacement
        displacement_reward = displacement * 1.0
        
        # Bonus for object velocity in target direction
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        obj_lin_vel, _ = p.getBaseVelocity(self.object_id)
        obj_vel_2d = np.array([obj_lin_vel[0], obj_lin_vel[1]])
        
        # Project velocity onto target direction
        velocity_in_direction = np.dot(obj_vel_2d, self.target_direction_vec)
        velocity_bonus = velocity_in_direction * 0.1
        
        total_reward = displacement_reward + velocity_bonus
        
        return total_reward
    
    def _is_done(self) -> bool:
        """
        Check if episode should terminate.
        
        Episode ends if:
        - Object falls below ground level
        - Object moves too far from origin (out of bounds)
        - Any robot falls below ground level significantly
        - Any robot moves too far horizontally (out of bounds)
        
        Returns:
            True if episode should terminate
        """
        # Check object
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        # Check if object fell below ground
        if obj_pos[2] < -0.5:
            return True
        
        # Check if object moved too far from origin
        obj_distance = np.sqrt(obj_pos[0]**2 + obj_pos[1]**2)
        if obj_distance > 50.0:  # Large boundary
            return True
        
        # Check robots
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            
            # Check if robot fell below ground significantly
            if pos[2] < -1.0:
                return True
            
            # Check if robot moved too far horizontally
            horizontal_distance = np.sqrt(pos[0]**2 + pos[1]**2)
            if horizontal_distance > 50.0:
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
        Same as previous sessions - enables collaborative pushing.
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
        
        # Reset displacement tracking
        self.max_displacement = 0.0
        
        # Clear connections
        for constraint_id, _, _, _ in self.connections:
            p.removeConstraint(constraint_id)
        self.connections = []
        
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        # Add session info
        info['training_session'] = 3
        info['session_goal'] = 'displace_object'
        info['object_shape'] = self.object_shape_name
        info['target_direction'] = self.target_direction_name
        info['object_initial_pos'] = self.object_initial_pos.tolist()
        info['max_displacement'] = self.max_displacement
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
        
        # At episode end, give bonus based on max displacement achieved
        if terminated or truncated:
            # Final reward is the maximum displacement achieved
            reward += self.max_displacement * 10.0  # Large bonus for final displacement
        
        # Add training session specific info
        info['training_session'] = 3
        info['object_shape'] = self.object_shape_name
        info['target_direction'] = self.target_direction_name
        info['max_displacement'] = self.max_displacement
        
        # Get current object position
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        info['object_pos'] = [obj_pos[0], obj_pos[1], obj_pos[2]]
        
        # Current displacement
        current_displacement = self._compute_displacement()
        info['current_displacement'] = current_displacement
        
        # Track robot positions
        robot_positions = []
        for robot_id in self.all_robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            robot_positions.append([pos[0], pos[1], pos[2]])
        
        info['robot_positions'] = robot_positions
        info['num_connections'] = len(self.connections)
        
        return observation, reward, terminated, truncated, info
