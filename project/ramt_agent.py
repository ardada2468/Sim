

"""
RAMT (Reliability-Aware Multi-modal Transformer) Training System
Improved version with:
- Middle lane starting position
- More training data collection
- Better data collection parameters
- Apple Silicon MPS GPU support
"""

import argparse
import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.lidar import Lidar
from metadrive.component.vehicle.vehicle_type import SVehicle, DefaultVehicle, XLVehicle, LVehicle, MVehicle
import metadrive.manager.traffic_manager as traffic_manager_module
import metadrive.policy.idm_policy as idm_policy_module
from panda3d.core import Vec3, loadPrcFileData

from metadrive.policy.base_policy import BasePolicy
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.utils.math import not_zero, wrap_to_pi, norm
from metadrive.policy.idm_policy import FrontBackObjects

# --- 1. SIMULATION CLASSES ---

class Motorcycle(SVehicle):
    """Custom motorcycle vehicle type"""
    DEFAULT_LENGTH = 2.0
    DEFAULT_WIDTH = 0.8
    DEFAULT_HEIGHT = 1.5
    
    @property
    def LENGTH(self): return self.DEFAULT_LENGTH
    @property
    def WIDTH(self): return self.DEFAULT_WIDTH
    @property
    def HEIGHT(self): return self.DEFAULT_HEIGHT
    @property
    def path(self): return ['bicycle/scene.gltf', (0.15, 0.15, 0.15), (0, 0, 0.3), (90, 0, 0)]

    def _add_wheel(self, pos: Vec3, radius: float, front: bool, left):
        wheel_np = self.origin.attachNewNode("wheel")
        self._node_path_list.append(wheel_np)
        wheel = self.system.createWheel()
        wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))
        wheel.setWheelRadius(radius)
        wheel.setMaxSuspensionTravelCm(self.SUSPENSION_LENGTH)
        wheel.setSuspensionStiffness(self.SUSPENSION_STIFFNESS)
        wheel.setWheelsDampingRelaxation(4.8)
        wheel.setWheelsDampingCompression(1.2)
        wheel_friction = self.config["wheel_friction"] if not self.config["no_wheel_friction"] else 0
        wheel.setFrictionSlip(wheel_friction)
        wheel.setRollInfluence(0.5)
        return wheel

class ChaoticIDMPolicy(BasePolicy):
    """Chaotic IDM Policy for generating diverse training data"""
    TAU_ACC = 0.6
    TAU_HEADING = 0.5
    TAU_LATERAL = 0.8
    TAU_PURSUIT = 0.5 * TAU_HEADING
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL
    MAX_STEERING_ANGLE = np.pi / 2
    DELTA_SPEED = 10
    DELTA = 10.0
    DELTA_RANGE = [3.5, 4.5]
    LANE_CHANGE_SPEED_INCREASE = 5
    MAX_LONG_DIST = 30
    MAX_SPEED = 100
    CREEP_SPEED = 10
    DEACC_FACTOR = -5

    def __init__(self, control_object, random_seed):
        super(ChaoticIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.DISTANCE_WANTED = self.np_random.uniform(0.5, 5.0) 
        self.ACC_FACTOR = self.np_random.uniform(1.0, 3.0)
        self.TIME_WANTED = self.np_random.uniform(0.1, 1.0)
        self.LANE_CHANGE_FREQ = self.np_random.randint(10, 50)
        self.SAFE_LANE_CHANGE_DISTANCE = self.np_random.uniform(5.0, 10.0)
        self.NORMAL_SPEED = self.np_random.uniform(30, 60)
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = self.engine.global_config.get("enable_idm_lane_change", True)
        self.disable_idm_deceleration = self.engine.global_config.get("disable_idm_deceleration", False)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

    def act(self, *args, **kwargs):
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            if success and self.enable_lane_change:
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects, 
                    self.routing_target_lane, 
                    self.control_object.position, 
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except Exception as e:
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane

        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        action = [steering, acc]
        self.action_info["action"] = action
        return action

    def move_to_next_road(self):
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            return True if self.routing_target_lane in current_lanes else False
        routing_network = self.control_object.navigation.map.road_network
        if self.routing_target_lane not in current_lanes:
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane) or \
                   routing_network.has_connection(self.routing_target_lane.index, lane.index):
                    self.routing_target_lane = lane
                    return True
            return False
        elif self.control_object.lane in current_lanes and \
             self.routing_target_lane is not self.control_object.lane:
            self.routing_target_lane = self.control_object.lane
            self.overtake_timer = self.np_random.randint(0, int(self.LANE_CHANGE_FREQ / 2))
            return True
        else:
            return True

    def steering_control(self, target_lane) -> float:
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def acceleration(self, front_obj, dist_to_front) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed_km_h, 0) / ego_target_speed, self.DELTA))
        if front_obj and (not self.disable_idm_deceleration):
            d = dist_to_front
            speed_diff = self.desired_gap(ego_vehicle, front_obj) / not_zero(d)
            acceleration -= self.ACC_FACTOR * (speed_diff**2)
        return acceleration

    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.ACC_FACTOR * self.DEACC_FACTOR
        dv = np.dot(ego_vehicle.velocity_km_h - front_obj.velocity_km_h, ego_vehicle.heading) \
             if projected else ego_vehicle.speed_km_h - front_obj.speed_km_h
        d_star = d0 + ego_vehicle.speed_km_h * tau + ego_vehicle.speed_km_h * dv / (2 * np.sqrt(ab))
        return d_star

    def reset(self):
        self.heading_pid.reset()
        self.lateral_pid.reset()
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)

    def lane_change_policy(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects, 
            self.routing_target_lane, 
            self.control_object.position, 
            self.MAX_LONG_DIST, 
            current_lanes
        )
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0
        
        if lane_num_diff > 0:
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            
            if self.routing_target_lane.index[-1] not in index_range:
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    if surrounding_objects.left_back_min_distance() < self.SAFE_LANE_CHANGE_DISTANCE or \
                       surrounding_objects.left_front_min_distance() < 5:
                        self.target_speed = self.CREEP_SPEED
                        return (surrounding_objects.front_object(), 
                                surrounding_objects.front_min_distance(), 
                                self.routing_target_lane)
                    else:
                        self.target_speed = self.NORMAL_SPEED
                        return (surrounding_objects.left_front_object(), 
                                surrounding_objects.left_front_min_distance(), 
                                current_lanes[self.routing_target_lane.index[-1] - 1])
                else:
                    if surrounding_objects.right_back_min_distance() < self.SAFE_LANE_CHANGE_DISTANCE or \
                       surrounding_objects.right_front_min_distance() < 5:
                        self.target_speed = self.CREEP_SPEED
                        return (surrounding_objects.front_object(), 
                                surrounding_objects.front_min_distance(), 
                                self.routing_target_lane)
                    else:
                        self.target_speed = self.NORMAL_SPEED
                        return (surrounding_objects.right_front_object(), 
                                surrounding_objects.right_front_min_distance(), 
                                current_lanes[self.routing_target_lane.index[-1] + 1])
        
        if abs(self.control_object.speed_km_h - self.NORMAL_SPEED) > 3 and \
           surrounding_objects.has_front_object() and \
           abs(surrounding_objects.front_object().speed_km_h - self.NORMAL_SPEED) > 3 and \
           self.overtake_timer > self.LANE_CHANGE_FREQ:
            
            right_front_speed = surrounding_objects.right_front_object().speed_km_h \
                if surrounding_objects.has_right_front_object() \
                else self.MAX_SPEED if surrounding_objects.right_lane_exist() and \
                     surrounding_objects.right_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and \
                     surrounding_objects.right_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE \
                else None
            
            front_speed = surrounding_objects.front_object().speed_km_h \
                if surrounding_objects.has_front_object() else self.MAX_SPEED
            
            left_front_speed = surrounding_objects.left_front_object().speed_km_h \
                if surrounding_objects.has_left_front_object() \
                else self.MAX_SPEED if surrounding_objects.left_lane_exist() and \
                     surrounding_objects.left_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and \
                     surrounding_objects.left_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE \
                else None
            
            if left_front_speed is not None and \
               left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                if expect_lane_idx in self.available_routing_index_range:
                    return (surrounding_objects.left_front_object(), 
                            surrounding_objects.left_front_min_distance(), 
                            current_lanes[expect_lane_idx])
            
            if right_front_speed is not None and \
               right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                if expect_lane_idx in self.available_routing_index_range:
                    return (surrounding_objects.right_front_object(), 
                            surrounding_objects.right_front_min_distance(), 
                            current_lanes[expect_lane_idx])
        
        self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return (surrounding_objects.front_object(), 
                surrounding_objects.front_min_distance(), 
                self.routing_target_lane)

# Monkeypatching
original_random_vehicle_type = traffic_manager_module.PGTrafficManager.random_vehicle_type

def chaotic_random_vehicle_type(self):
    if self.np_random.random() < 0.3:
        return Motorcycle
    else:
        return self.np_random.choice([SVehicle, MVehicle, LVehicle, XLVehicle, DefaultVehicle])

traffic_manager_module.PGTrafficManager.random_vehicle_type = chaotic_random_vehicle_type
idm_policy_module.IDMPolicy = ChaoticIDMPolicy

from metadrive.component.vehicle import vehicle_type as vehicle_type_module
vehicle_type_module.vehicle_type["motorcycle"] = Motorcycle
vehicle_type_module.vehicle_class_to_type[Motorcycle] = "motorcycle"

# --- 2. RAMT MODEL ARCHITECTURE ---

class CameraEncoder(nn.Module):
    """Encodes camera images into feature vectors"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc(x))
        return x

class LidarEncoder(nn.Module):
    """Encodes LiDAR data into feature vectors"""
    def __init__(self, input_dim=240):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

    def forward(self, x):
        return self.net(x)

class TelemetryEncoder(nn.Module):
    """Encodes vehicle telemetry into feature vectors"""
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x):
        return self.net(x)

class ReliabilityEstimator(nn.Module):
    """Estimates reliability weights for each sensor modality"""
    def __init__(self, feature_dim=256+256+128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Weights for Camera, Lidar, Telemetry
        )

    def forward(self, features):
        logits = self.net(features)
        weights = torch.softmax(logits, dim=1)
        return weights

class RAMT(nn.Module):
    """Reliability-Aware Multi-modal Transformer for autonomous driving"""
    def __init__(self, lidar_input_dim=240):
        super().__init__()
        self.camera_encoder = CameraEncoder()
        self.lidar_encoder = LidarEncoder(input_dim=lidar_input_dim)
        self.telemetry_encoder = TelemetryEncoder()
        
        self.reliability_estimator = ReliabilityEstimator(feature_dim=256+256+128)
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 256 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Steering, Acceleration
            nn.Tanh()
        )

    def forward(self, img, lidar, telemetry):
        f_cam = self.camera_encoder(img)
        f_lidar = self.lidar_encoder(lidar)
        f_tel = self.telemetry_encoder(telemetry)
        
        concat_features = torch.cat([f_cam, f_lidar, f_tel], dim=1)
        
        reliability_weights = self.reliability_estimator(concat_features)
        
        w_cam = reliability_weights[:, 0:1]
        w_lidar = reliability_weights[:, 1:2]
        w_tel = reliability_weights[:, 2:3]
        
        f_cam_weighted = f_cam * w_cam
        f_lidar_weighted = f_lidar * w_lidar
        f_tel_weighted = f_tel * w_tel
        
        weighted_features = torch.cat([f_cam_weighted, f_lidar_weighted, f_tel_weighted], dim=1)
        
        fused = self.fusion_fc(weighted_features)
        action = self.action_head(fused)
        
        return action, reliability_weights

# --- 3. DATA & TRAINING ---

class DrivingDataset(Dataset):
    """Dataset for driving data"""
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = cv2.resize(sample['image'], (64, 64))
        img_tensor = self.transform(img).float()
        
        lidar_tensor = torch.tensor(sample['lidar'], dtype=torch.float32)
        telemetry_tensor = torch.tensor(sample['telemetry'], dtype=torch.float32)
        action_tensor = torch.tensor(sample['action'], dtype=torch.float32)
        
        return img_tensor, lidar_tensor, telemetry_tensor, action_tensor

def get_env_config(lidar_num=240, for_training=False, seed=None):
    """Get consistent environment configuration"""
    start_seed = seed if seed is not None else 32
    if not for_training:
        print(f"Test Seed: {start_seed}")

    config = {
        "use_render": not for_training,
        #out_of_road_done: False
        "out_of_road_done": False  if for_training else False,
        "manual_control": False,
        "traffic_density": 0.3 if for_training else 0.4,
        "map": "SCrR",
        "start_seed": start_seed,
        "num_scenarios": 30 if for_training else 10,
        "image_observation": False,
        "sensors": {
            "rgb_camera": (RGBCamera, 256, 256),
            "lidar": (Lidar, ),
        },
        "vehicle_config": {
            "image_source": "rgb_camera",
            "lidar": {
                "num_lasers": lidar_num,
                "distance": 50,
                "num_others": 4
            },
        }
    }
    print(config)
    return config

import pickle
import math

def collect_data(num_episodes=20, steps_per_episode=500, lidar_num=240, data_file="training_data.pkl"):
    """Collect training data - Persistent & Incremental"""
    print(f"Collecting data with {lidar_num} lidar points...")
    target_samples = num_episodes * steps_per_episode
    print(f"Target: {num_episodes} episodes × {steps_per_episode} steps = {target_samples} samples")
    
    buffer = []
    
    # 1. Try to load existing data
    if os.path.exists(data_file):
        try:
            with open(data_file, 'rb') as f:
                buffer = pickle.load(f)
            print(f"✓ Loaded {len(buffer)} samples from {data_file}")
            
            # Check consistency (optional, but good for safety)
            if len(buffer) > 0:
                sample_lidar_dim = buffer[0]['lidar'].shape[0]
                if sample_lidar_dim != lidar_num:
                    print(f"⚠ Warning: Loaded data has lidar_dim={sample_lidar_dim}, requested={lidar_num}")
                    print("  Discarding existing data to avoid mismatch.")
                    buffer = []
        except Exception as e:
            print(f"⚠ Error loading data file: {e}")
            buffer = []
            
    # 2. Check if we have enough data
    current_samples = len(buffer)
    if current_samples >= target_samples:
        print(f"✓ Sufficient data available ({current_samples} >= {target_samples}). Skipping generation.")
        return buffer[:target_samples]
    
    # 3. Generate missing data
    needed_samples = target_samples - current_samples
    needed_episodes = math.ceil(needed_samples / steps_per_episode)
    print(f"Need {needed_samples} more samples. Running {needed_episodes} additional episodes...")
    
    config = get_env_config(lidar_num=lidar_num, for_training=True)
    env = MetaDriveEnv(config)
    
    new_buffer = []
    
    # Calculate starting episode index to avoid repeating seeds
    start_episode_idx = math.ceil(current_samples / steps_per_episode)
    
    try:
        for i in range(needed_episodes):
            ep_idx = start_episode_idx + i
            # Ensure we use a unique seed for the environment scenario
            # MetaDrive uses start_seed + (seed % num_scenarios)
            scenario_seed = 42 + ep_idx
            obs, _ = env.reset(seed=scenario_seed)
            
            # Vary policy seed as well
            expert_policy = ChaoticIDMPolicy(env.agent, random_seed=32 + ep_idx) 
            
            # START IN MIDDLE LANE (IMPROVED)
            vehicle = env.agent
            if vehicle.navigation and vehicle.navigation.current_ref_lanes:
                try:
                    lanes = vehicle.navigation.current_ref_lanes
                    middle_lane_idx = len(lanes) // 2  # Get middle lane
                    middle_lane = lanes[middle_lane_idx]
                    long, lat = middle_lane.local_coordinates(vehicle.position)
                    new_pos = middle_lane.position(long, 0)
                    vehicle.set_position(new_pos, height=vehicle.HEIGHT/2)
                    vehicle.set_heading_theta(middle_lane.heading_theta_at(long))
                    print(f"  Episode {ep_idx+1}: Starting in lane {middle_lane_idx+1}/{len(lanes)} (middle)")
                except Exception as e:
                    print(f"  Warning: Could not set initial lane: {e}")
                
            for step in range(steps_per_episode):
                try:
                    action = expert_policy.act()
                    action = np.clip(action, -1.0, 1.0)
                except Exception as e:
                    action = [0.0, 0.0]
                
                if "image" in obs:
                    img = obs["image"][..., 0] if len(obs["image"].shape) == 4 else obs["image"]
                    img = (img * 255).astype(np.uint8)
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.shape[-1] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = np.zeros((256, 256, 3), dtype=np.uint8)
                
                if isinstance(obs, dict):
                    lidar = obs.get("lidar", np.zeros(lidar_num, dtype=np.float32))
                else:
                    lidar = obs if len(obs.shape) == 1 else obs.flatten()
                
                if len(lidar.shape) != 1:
                    lidar = lidar.flatten()
                if lidar.shape[0] < lidar_num:
                    lidar = np.pad(lidar, (0, lidar_num - lidar.shape[0]))
                elif lidar.shape[0] > lidar_num:
                    lidar = lidar[:lidar_num]
                
                v = env.agent
                telemetry = np.array([
                    v.speed if hasattr(v, 'speed') else 0.0,
                    v.heading_theta if hasattr(v, 'heading_theta') else 0.0,
                    v.position[0] if hasattr(v, 'position') else 0.0,
                    v.position[1] if hasattr(v, 'position') else 0.0,
                    v.steering if hasattr(v, 'steering') else 0.0,
                    v.throttle_brake if hasattr(v, 'throttle_brake') else 0.0,
                    0.0, 0.0
                ], dtype=np.float32)
                
                new_buffer.append({
                    'image': img,
                    'lidar': lidar.astype(np.float32),
                    'telemetry': telemetry,
                    'action': np.array(action, dtype=np.float32)
                })
                
                obs, r, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
                
            print(f"  Episode {ep_idx+1} (Batch {i+1}/{needed_episodes}) complete - New samples: {len(new_buffer)}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    
    # 4. Merge and Save
    if len(new_buffer) > 0:
        buffer.extend(new_buffer)
        print(f"✓ Generated {len(new_buffer)} new samples.")
        try:
            with open(data_file, 'wb') as f:
                pickle.dump(buffer, f)
            print(f"✓ Saved total {len(buffer)} samples to {data_file}")
        except Exception as e:
            print(f"⚠ Failed to save data: {e}")
    
    if len(buffer) == 0:
        raise RuntimeError("No data collected!")
    
    print(f"\n✓ Data collection complete!")
    print(f"  Total samples: {len(buffer)}")
    print(f"  Lidar dimension: {lidar_num}")
    return buffer[:target_samples]

def train_model(model, data_buffer, epochs=100, batch_size=32, validation_split=0.2, device='cpu'):
    """Train the RAMT model"""
    model = model.to(device)
    
    split_idx = int(len(data_buffer) * (1 - validation_split))
    train_data = data_buffer[:split_idx]
    val_data = data_buffer[split_idx:]
    
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Training samples:   {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Batch size:         {batch_size}")
    print(f"Epochs:             {epochs}")
    print(f"Device:             {device}")
    print(f"{'='*60}\n")
    
    train_dataset = DrivingDataset(train_data)
    val_dataset = DrivingDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience =90
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        
        for imgs, lidars, tels, actions in train_loader:
            imgs = imgs.to(device)
            lidars = lidars.to(device)
            tels = tels.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            
            # Sensor dropout for robustness
            if random.random() < 0.2:
                imgs = torch.zeros_like(imgs)
            if random.random() < 0.2:
                lidars = torch.zeros_like(lidars)
            
            pred_actions, reliability = model(imgs, lidars, tels)
            loss = criterion(pred_actions, actions)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for imgs, lidars, tels, actions in val_loader:
                imgs = imgs.to(device)
                lidars = lidars.to(device)
                tels = tels.to(device)
                actions = actions.to(device)
                
                pred_actions, reliability = model(imgs, lidars, tels)
                loss = criterion(pred_actions, actions)
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", end="")
        
        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "ramt_best_model.pth")
            print(" ← Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            print()
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break
                
    print(f"\n{'='*60}")
    print(f"✓ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

def visualize_dashboard(img, lidar, telemetry, action_pred, reliability, step):
    """Create and display a real-time dashboard - IMPROVED UI"""
    # Canvas setup
    H, W = 800, 1200
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30) # Dark grey background
    
    # Title Bar
    cv2.rectangle(canvas, (0, 0), (W, 60), (20, 20, 20), -1)
    cv2.putText(canvas, "RAMT AGENT - REAL-TIME TELEMETRY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(canvas, f"Step: {step}", (W-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # 1. Camera View (Top Left)
    cam_size = 500
    if img is not None:
        cam_view = cv2.resize(img, (cam_size, cam_size))
        # Add border
        cv2.rectangle(canvas, (45, 95), (45+cam_size+10, 95+cam_size+10), (100, 100, 100), -1)
        canvas[100:100+cam_size, 50:50+cam_size] = cam_view
    
    cv2.putText(canvas, "CAMERA FEED", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # 2. Lidar View (Top Right)
    lidar_size = 500
    lidar_center = (850, 350)
    lidar_radius = 230
    
    # Draw Radar Circle
    cv2.circle(canvas, lidar_center, lidar_radius, (50, 50, 50), -1)
    cv2.circle(canvas, lidar_center, lidar_radius, (100, 100, 100), 2)
    cv2.circle(canvas, lidar_center, int(lidar_radius*0.66), (70, 70, 70), 1)
    cv2.circle(canvas, lidar_center, int(lidar_radius*0.33), (70, 70, 70), 1)
    
    # Draw Ego Vehicle
    ego_len = 20
    ego_wid = 10
    cv2.rectangle(canvas, (lidar_center[0]-ego_wid//2, lidar_center[1]-ego_len//2), 
                  (lidar_center[0]+ego_wid//2, lidar_center[1]+ego_len//2), (0, 0, 255), -1)
    
    # Draw lidar points
    if lidar is not None:
        num_points = len(lidar)
        # Logic from lidar_thingi.py:
        # angles = np.linspace(0, 2 * np.pi, num_lasers, endpoint=False)
        # x_rel = dist_m * np.cos(angle) (Forward)
        # y_rel = dist_m * np.sin(angle) (Left)
        # px = cx - int(y_rel * scale)
        # py = cy - int(x_rel * scale)
        
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        
        for i, dist in enumerate(lidar):
            if dist < 1.0: # Only draw if within range
                dist_m = dist * 50 # Max distance 50m
                angle = angles[i]
                
                x_rel = dist_m * np.cos(angle)
                y_rel = dist_m * np.sin(angle)
                
                # Map to screen: Forward -> Up (y-), Left -> Left (x-)
                # Scale: 50m -> 230px => scale = 4.6
                scale = 4.6
                
                px = int(lidar_center[0] - y_rel * scale)
                py = int(lidar_center[1] - x_rel * scale)
                
                # Color based on distance (Green -> Red)
                color = (0, int(255 * dist), int(255 * (1-dist)))
                cv2.circle(canvas, (px, py), 2, color, -1)
            
    cv2.putText(canvas, "LIDAR SENSOR", (600, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # 3. Telemetry & Prediction (Bottom)
    speed = telemetry[0]
    steering = telemetry[4]
    throttle = telemetry[5]
    
    pred_steering = action_pred[0]
    pred_acc = action_pred[1]
    
    rel_cam = reliability[0]
    rel_lidar = reliability[1]
    rel_tel = reliability[2]

    panel_y = 630
    
    # Speedometer (Text)
    cv2.putText(canvas, f"{speed:.1f}", (100, panel_y+50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    cv2.putText(canvas, "km/h", (120, panel_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Steering Bar
    bar_w = 300
    bar_x = 350
    bar_y = panel_y + 40
    
    cv2.putText(canvas, "STEERING", (bar_x + 110, bar_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+20), (50, 50, 50), -1)
    cv2.line(canvas, (bar_x+bar_w//2, bar_y-5), (bar_x+bar_w//2, bar_y+25), (150, 150, 150), 2) # Center
    
    # Actual Steering (Blue)
    s_pos = int(bar_x + bar_w//2 + steering * bar_w//2)
    cv2.circle(canvas, (s_pos, bar_y+10), 8, (255, 100, 0), -1)
    
    # Pred Steering (Yellow)
    ps_pos = int(bar_x + bar_w//2 + pred_steering * bar_w//2)
    cv2.circle(canvas, (ps_pos, bar_y+10), 6, (0, 255, 255), -1)
    
    # Legend
    cv2.putText(canvas, "Actual", (bar_x, bar_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    cv2.putText(canvas, "Pred", (bar_x+250, bar_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Throttle/Brake
    t_bar_x = 750
    cv2.putText(canvas, "THROTTLE / BRAKE", (t_bar_x, bar_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Throttle
    cv2.rectangle(canvas, (t_bar_x, bar_y), (t_bar_x+100, bar_y+20), (50, 50, 50), -1)
    val_t = max(0, throttle)
    cv2.rectangle(canvas, (t_bar_x, bar_y), (t_bar_x+int(val_t*100), bar_y+20), (0, 255, 0), -1)
    cv2.putText(canvas, "Thr", (t_bar_x, bar_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Brake
    cv2.rectangle(canvas, (t_bar_x+120, bar_y), (t_bar_x+220, bar_y+20), (50, 50, 50), -1)
    val_b = max(0, -throttle) # Assuming negative throttle is brake
    cv2.rectangle(canvas, (t_bar_x+120, bar_y), (t_bar_x+120+int(val_b*100), bar_y+20), (0, 0, 255), -1)
    cv2.putText(canvas, "Brk", (t_bar_x+120, bar_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Reliability
    r_x = 1000
    r_y = panel_y
    cv2.putText(canvas, "CONFIDENCE", (r_x, r_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def draw_conf(label, val, y, col):
        cv2.putText(canvas, label, (r_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(canvas, (r_x+60, y-10), (r_x+160, y+5), (50, 50, 50), -1)
        cv2.rectangle(canvas, (r_x+60, y-10), (r_x+60+int(val*100), y+5), col, -1)
        
    draw_conf("CAM", rel_cam, r_y+10, (0, 255, 0))
    draw_conf("LID", rel_lidar, r_y+35, (0, 255, 255))
    draw_conf("TEL", rel_tel, r_y+60, (255, 100, 100))

    cv2.imshow("RAMT Agent Dashboard", canvas)
    cv2.waitKey(1)

def test_model(model_path="ramt_best_model.pth", render=True, episodes=10, device='cpu', lidar_num=None, seed=None):
    """Test the trained model"""
    print(f"\n{'='*60}")
    print(f"TESTING MODEL: {model_path}")
    print(f"{'='*60}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        lidar_weight_key = 'lidar_encoder.net.0.weight'
        if lidar_weight_key in checkpoint:
            detected_lidar_dim = checkpoint[lidar_weight_key].shape[1]
            print(f"✓ Model expects lidar_dim={detected_lidar_dim}")
            if lidar_num is not None and lidar_num != detected_lidar_dim:
                print(f"⚠ Requested {lidar_num} but model uses {detected_lidar_dim}")
            lidar_num = detected_lidar_dim
        else:
            lidar_num = lidar_num or 240
            print(f"Using default lidar_dim={lidar_num}")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        print("Please train the model first!")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        lidar_num = lidar_num or 240
    
    config = get_env_config(lidar_num=lidar_num, for_training=False, seed=seed)
    config["use_render"] = render
    env = MetaDriveEnv(config)
    
    model = RAMT(lidar_input_dim=lidar_num)
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        env.close()
        return

    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    success_count = 0
    collision_count = 0
    total_steps = 0
    
    print(f"\nRunning {episodes} test episodes...\n")
    
    try:
        for ep in range(episodes):
            current_seed = (seed if seed is not None else 32) + ep
            print(f"Episode {ep+1} starting with seed: {current_seed}")
            obs, _ = env.reset(seed=current_seed)
            
            # START IN MIDDLE LANE (same as training)
            vehicle = env.agent
            if vehicle.navigation and vehicle.navigation.current_ref_lanes:
                try:
                    lanes = vehicle.navigation.current_ref_lanes
                    middle_lane_idx = len(lanes) // 2
                    middle_lane = lanes[middle_lane_idx]
                    long, lat = middle_lane.local_coordinates(vehicle.position)
                    new_pos = middle_lane.position(long, 0)
                    vehicle.set_position(new_pos, height=vehicle.HEIGHT/2)
                    vehicle.set_heading_theta(middle_lane.heading_theta_at(long))
                except:
                    pass
                
            episode_steps = 0
            

            for i in range(1000):
                # Manual Camera Capture
                raw_img = None
                try:
                    # Access the RGB camera directly
                    cam = env.engine.get_sensor("rgb_camera")
                    if cam:
                        # perceive() returns the image. 
                        # Note: check if we need to pass arguments. 
                        # Usually cam.perceive(to_float=False) returns uint8 array.
                        # But MetaDrive API varies. Let's try getting the buffer.
                        # Or use the fact that we are rendering.
                        # If image_observation is False, we might need to force render?
                        # Actually, if use_render is True, the window is open.
                        # We can grab the image from the buffer or sensor.
                        
                        # Let's try the standard way if sensor is available
                        raw_img = cam.perceive(to_float=False)
                        
                        if len(raw_img.shape) == 2:
                            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
                        elif raw_img.shape[-1] == 3:
                            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
                        elif raw_img.shape[-1] == 4:
                            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2BGR)
                            
                        img = cv2.resize(raw_img, (64, 64))
                    else:
                        img = np.zeros((64, 64, 3), dtype=np.uint8)
                except Exception as e:
                    # print(f"Cam Error: {e}")
                    img = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                img_tensor = transform(img).float().unsqueeze(0).to(device)
                
                # Debug Lidar Extraction
                # print(f"Obs type: {type(obs)}")
                if isinstance(obs, dict):
                    # print(f"Obs keys: {obs.keys()}")
                    if "lidar" in obs:
                        lidar_data = obs["lidar"]
                    else:
                        # If no 'lidar' key, maybe it's the whole obs if it's not a dict of sensors?
                        # But we checked isinstance(obs, dict).
                        # Sometimes MetaDrive returns a dict with other keys.
                        lidar_data = np.zeros(lidar_num, dtype=np.float32)
                else:
                    # If not dict, it's likely the lidar array itself
                    lidar_data = obs
                
                # Ensure numpy array
                if not isinstance(lidar_data, np.ndarray):
                    lidar_data = np.array(lidar_data)
                
                if len(lidar_data.shape) != 1:
                    lidar_data = lidar_data.flatten()
                if lidar_data.shape[0] < lidar_num:
                    lidar_data = np.pad(lidar_data, (0, lidar_num - lidar_data.shape[0]))
                elif lidar_data.shape[0] > lidar_num:
                    lidar_data = lidar_data[:lidar_num]
                
                lidar_tensor = torch.tensor(lidar_data, dtype=torch.float32).unsqueeze(0).to(device)
                
                v = env.agent
                telemetry = np.array([
                    v.speed if hasattr(v, 'speed') else 0.0,
                    v.heading_theta if hasattr(v, 'heading_theta') else 0.0,
                    v.position[0] if hasattr(v, 'position') else 0.0,
                    v.position[1] if hasattr(v, 'position') else 0.0,
                    v.steering if hasattr(v, 'steering') else 0.0,
                    v.throttle_brake if hasattr(v, 'throttle_brake') else 0.0,
                    0.0, 0.0
                ], dtype=np.float32)
                telemetry_tensor = torch.tensor(telemetry, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action_pred, reliability = model(img_tensor, lidar_tensor, telemetry_tensor)
                    action = action_pred.squeeze().cpu().numpy()
                    
                action = np.clip(action, -1.0, 1.0)
                
                obs, r, terminated, truncated, info = env.step(action)
                if render:
                    env.render()
                
                # Visualization
                try:
                    rel_weights = reliability.squeeze().cpu().numpy()
                    visualize_dashboard(raw_img, lidar_data, telemetry, action, rel_weights, episode_steps)
                except Exception as e:
                    print(f"Vis Error: {e}")
                
                episode_steps += 1
                total_steps += 1
                
                if terminated or truncated:
                    if info.get("arrive_dest", False):
                        print(f"Episode {ep+1:2d}: ✓ SUCCESS ({episode_steps:3d} steps)")
                        success_count += 1
                    elif info.get("crash", False):
                        print(f"Episode {ep+1:2d}: ✗ CRASH   ({episode_steps:3d} steps)")
                        collision_count += 1
                    else:
                        print(f"Episode {ep+1:2d}: ○ TIMEOUT ({episode_steps:3d} steps)")
                    break
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        cv2.destroyAllWindows()
        
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Episodes:     {episodes}")
    print(f"Success Rate:       {success_count/episodes*100:.1f}% ({success_count}/{episodes})")
    print(f"Collision Rate:     {collision_count/episodes*100:.1f}% ({collision_count}/{episodes})")
    print(f"Average Steps:      {total_steps/episodes:.1f}")
    print(f"{'='*60}\n")

def get_device():
    """Automatically detect and return the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAMT Training System')
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes for data collection/testing (default: 20)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Steps per episode for data collection (default: 500)")
    parser.add_argument("--lidar-num", type=int, default=240,
                        help="Number of lidar points (default: 240)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering in test mode")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"],
                        help="Device to use for training (default: auto - automatically detects best device)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Specific seed for testing (default: None)")
    args = parser.parse_args()
    
    # Automatically detect device if set to auto
    if args.device == "auto":
        args.device = get_device()
        print(f"Auto-detected device: {args.device}")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU instead")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠ MPS not available, using CPU instead")
        args.device = "cpu"
    
    print(f"\n{'='*60}")
    print(f"RAMT Training System - IMPROVED VERSION (Apple Silicon Support)")
    print(f"{'='*60}")
    print(f"Mode:          {args.mode}")
    print(f"Device:        {args.device.upper()}")
    if args.device == "mps":
        print(f"               (Apple Silicon GPU)")
    print(f"Lidar points:  {args.lidar_num}")
    if args.mode == "train":
        print(f"Episodes:      {args.episodes}")
        print(f"Steps/episode: {args.steps}")
        print(f"Training epochs: {args.epochs}")
        print(f"Expected samples: ~{args.episodes * args.steps}")
    else:
        print(f"Test episodes: {args.episodes}")
        print(f"Render:        {not args.no_render}")
    print(f"{'='*60}\n")
    
    if args.mode == "train":
        print("Starting data collection with MIDDLE LANE start position...\n")
        data = collect_data(
            num_episodes=args.episodes, 
            steps_per_episode=args.steps,
            lidar_num=args.lidar_num,
            data_file="training_data.pkl"
        )
        
        if len(data) > 0:
            lidar_dim = data[0]['lidar'].shape[0]
            print(f"\nInitializing RAMT model with lidar_dim={lidar_dim}")
            model = RAMT(lidar_input_dim=lidar_dim)
            train_model(model, data, epochs=args.epochs, device=args.device)
            print(f"\n✓ Training complete!")
            print(f"  Model saved to: ramt_best_model.pth")
            print(f"\nTo test the model, run:")
            print(f"  python {os.path.basename(__file__)} --mode test --lidar-num {lidar_dim} --device {args.device}")
    else:
        test_model(
            render=not args.no_render, 
            episodes=args.episodes, 
            device=args.device,
            lidar_num=args.lidar_num,
            seed=args.seed
        )




