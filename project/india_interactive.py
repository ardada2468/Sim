import argparse
import os
import cv2
import csv
import numpy as np
import random
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.lidar import Lidar
from metadrive.constants import HELP_MESSAGE
from metadrive.component.vehicle.vehicle_type import SVehicle, DefaultVehicle, XLVehicle, LVehicle, MVehicle
import metadrive.manager.traffic_manager as traffic_manager_module
import metadrive.policy.idm_policy as idm_policy_module
from panda3d.core import Vec3
from metadrive.engine.asset_loader import AssetLoader

# Imports for ChaoticIDMPolicy
from metadrive.policy.base_policy import BasePolicy
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.utils.math import not_zero, wrap_to_pi, norm
from metadrive.policy.idm_policy import FrontBackObjects

# --- 1. DEFINE CUSTOM CLASSES ---

class Motorcycle(SVehicle):
    """
    A motorcycle class derived from SVehicle (Beetle).
    We scale it down to look more like a bike and adjust dimensions.
    """
    # Override dimensions to be smaller
    DEFAULT_LENGTH = 2.0
    DEFAULT_WIDTH = 0.8
    DEFAULT_HEIGHT = 1.5
    
    @property
    def LENGTH(self):
        return self.DEFAULT_LENGTH

    @property
    def WIDTH(self):
        return self.DEFAULT_WIDTH

    @property
    def HEIGHT(self):
        return self.DEFAULT_HEIGHT

    @property
    def path(self):
        # Use the requested bicycle model
        # Scale down significantly (0.15) and rotate 90 degrees to face forward
        # Raise by 0.3m to prevent being buried in the ground
        return ['bicycle/scene.gltf', (0.15, 0.15, 0.15), (0, 0, 0.3), (90, 0, 0)]

    def _add_wheel(self, pos: Vec3, radius: float, front: bool, left):
        wheel_np = self.origin.attachNewNode("wheel")
        self._node_path_list.append(wheel_np)

        # VISUALS DISABLED: We don't want to show the beetle tires
        # if self.render:
        #     model = 'right_tire_front.gltf' if front else 'right_tire_back.gltf'
        #     # Use beetle tires since bicycle model doesn't have them
        #     model_path = AssetLoader.file_path("models", "beetle", model)
        #     wheel_model = self.loader.loadModel(model_path)
        #     wheel_model.setTwoSided(self.TIRE_TWO_SIDED)
        #     wheel_model.reparentTo(wheel_np)
        #     wheel_model.set_scale(1 * self.TIRE_MODEL_CORRECT if left else -1 * self.TIRE_MODEL_CORRECT)
        
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
    """
    A more aggressive and unpredictable IDM policy.
    Copied and modified from IDMPolicy to avoid recursion issues with monkeypatching.
    """
    DEBUG_MARK_COLOR = (219, 3, 252, 255)

    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.5  # [s]
    TAU_LATERAL = 0.8  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 2  # [rad]
    DELTA_SPEED = 10  # [m/s]

    # DISTANCE_WANTED = 10.0
    # TIME_WANTED = 1.5  # [s]
    DELTA = 10.0  # []
    DELTA_RANGE = [3.5, 4.5]

    # Lateral policy parameters
    # LANE_CHANGE_FREQ = 50  # [step]
    LANE_CHANGE_SPEED_INCREASE = 5
    # SAFE_LANE_CHANGE_DISTANCE = 15
    MAX_LONG_DIST = 30
    MAX_SPEED = 100  # km/h

    # Normal speed
    # NORMAL_SPEED = 30  # km/h

    # Creep Speed
    CREEP_SPEED = 10

    # acc factor
    # ACC_FACTOR = 1.0
    DEACC_FACTOR = -5

    def __init__(self, control_object, random_seed):
        super(ChaoticIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        
        # --- CHAOTIC PARAMETERS ---
        self.DISTANCE_WANTED = self.np_random.uniform(0.5, 5.0) 
        self.ACC_FACTOR = self.np_random.uniform(1.0, 3.0)
        self.TIME_WANTED = self.np_random.uniform(0.1, 1.0)
        self.LANE_CHANGE_FREQ = self.np_random.randint(10, 50)
        self.SAFE_LANE_CHANGE_DISTANCE = self.np_random.uniform(5.0, 10.0)
        self.NORMAL_SPEED = self.np_random.uniform(30, 60) # km/h
        
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = self.engine.global_config.get("enable_idm_lane_change", True)
        self.disable_idm_deceleration = self.engine.global_config.get("disable_idm_deceleration", False)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

    def act(self, *args, **kwargs):
        # concat lane
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            if success and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except:
            # error fallback
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane
            # logging.warning("IDM bug! fall back")
            # print("IDM bug! fall back")

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        action = [steering, acc]
        self.action_info["action"] = action
        return action

    def move_to_next_road(self):
        # routing target lane is in current ref lanes
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            return True if self.routing_target_lane in current_lanes else False
        routing_network = self.control_object.navigation.map.road_network
        if self.routing_target_lane not in current_lanes:
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane) or \
                        routing_network.has_connection(self.routing_target_lane.index, lane.index):
                    # two lanes connect
                    self.routing_target_lane = lane
                    return True
                    # lane change for lane num change
            return False
        elif self.control_object.lane in current_lanes and self.routing_target_lane is not self.control_object.lane:
            # lateral routing lane change
            self.routing_target_lane = self.control_object.lane
            self.overtake_timer = self.np_random.randint(0, int(self.LANE_CHANGE_FREQ / 2))
            return True
        else:
            return True

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance control
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
        dv = np.dot(ego_vehicle.velocity_km_h - front_obj.velocity_km_h, ego_vehicle.heading) if projected \
            else ego_vehicle.speed_km_h - front_obj.speed_km_h
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
            all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
        )
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0

        # We have to perform lane changing because the number of lanes in next road is less than current road
        if lane_num_diff > 0:
            # lane num decreasing happened in left road or right road
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                # not on suitable lane do lane change !!!
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    # change to left
                    if surrounding_objects.left_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                        # creep to wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane
                    else:
                        # it is time to change lane!
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] - 1]
                else:
                    # change to right
                    if surrounding_objects.right_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                        # unsafe, creep and wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane,
                    else:
                        # change lane
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] + 1]

        # lane follow or active change lane/overtake for high driving speed
        if abs(self.control_object.speed_km_h - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
        ) and abs(surrounding_objects.front_object().speed_km_h -
                  self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
            # may lane change
            right_front_speed = surrounding_objects.right_front_object().speed_km_h if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_lane_exist() and surrounding_objects.right_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.right_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
            front_speed = surrounding_objects.front_object().speed_km_h if surrounding_objects.has_front_object(
            ) else self.MAX_SPEED
            left_front_speed = surrounding_objects.left_front_object().speed_km_h if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_lane_exist() and surrounding_objects.left_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.left_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
            if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                # left overtake has a high priority
                expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                if expect_lane_idx in self.available_routing_index_range:
                    return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                           current_lanes[expect_lane_idx]
            if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                if expect_lane_idx in self.available_routing_index_range:
                    return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                           current_lanes[expect_lane_idx]

        # fall back to lane follow
        self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane


# --- 2. MONKEYPATCHING ---

# Patch the random_vehicle_type function to include Motorcycles
original_random_vehicle_type = traffic_manager_module.PGTrafficManager.random_vehicle_type

def chaotic_random_vehicle_type(self):
    # 30% chance of Motorcycle, rest distributed among others
    if self.np_random.random() < 0.3:
        return Motorcycle
    else:
        # Fallback to standard types
        return self.np_random.choice([SVehicle, MVehicle, LVehicle, XLVehicle, DefaultVehicle])

# Apply the patch to the class method
traffic_manager_module.PGTrafficManager.random_vehicle_type = chaotic_random_vehicle_type

# Patch the IDMPolicy to be Chaotic
idm_policy_module.IDMPolicy = ChaoticIDMPolicy

# REGISTER MOTORCYCLE IN METADRIVE REGISTRY
# This is required because BaseVehicle looks up the class in vehicle_class_to_type
from metadrive.component.vehicle import vehicle_type as vehicle_type_module
vehicle_type_module.vehicle_type["motorcycle"] = Motorcycle
vehicle_type_module.vehicle_class_to_type[Motorcycle] = "motorcycle"


# --- 3. MAIN SIMULATION LOOP ---

def interactive_drive():
    # 1. CONFIGURATION
    config = {
        "use_render": True,
        "manual_control": True,
        "traffic_density": 0.5, # Increased density for chaos
        "map": "SCrR",
        "num_scenarios": 1,
        "start_seed": 42,
        "image_observation": True, 
        "sensors": {
            "rgb_camera": (RGBCamera, 256, 256),
            "lidar": (Lidar, ),
        },
        "vehicle_config": {
            "image_source": "rgb_camera",
            "lidar": {
                "num_lasers": 240,
                "distance": 50,
                "num_others": 4
            },
        }
    }
    
    # 2. INITIALIZE ENVIRONMENT
    env = MetaDriveEnv(config)
    
    data_dir = "data_interactive_india"
    img_dir = os.path.join(data_dir, "images")
    lidar_dir = os.path.join(data_dir, "lidar")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)
    
    telemetry_path = os.path.join(data_dir, "telemetry.csv")
    telemetry_file = open(telemetry_path, 'w', newline='')
    telemetry_writer = None
    
    try:
        print(HELP_MESSAGE)
        print(f"Collecting chaotic data to: {data_dir}")
        obs, info = env.reset()
        
        # Get physics world for ray casting
        physics_world = env.engine.physics_world.dynamic_world
        
        for i in range(500): 
            action = [0.0, 0.5] 
            step_result = env.step(action)
            
            if len(step_result) == 5:
                obs, r, terminated, truncated, info = step_result
                d = terminated or truncated
            else:
                obs, r, d, info = step_result
            
            vehicle = env.agent
            
            # --- DEBUG INFO (Step 0) ---
            if i == 0:
                print("\n=== SENSOR DEBUG ===")
                if hasattr(vehicle, 'lidar') and vehicle.lidar is not None:
                    print(f"✓ Lidar hardware found")
                print("====================\n")

            # --- A. TELEMETRY ---
            telemetry = {
                "step": i,
                "speed": vehicle.speed,
                "heading": vehicle.heading_theta,
                "position_x": vehicle.position[0],
                "position_y": vehicle.position[1],
                "steering": vehicle.steering,
                "throttle": vehicle.throttle_brake if vehicle.throttle_brake > 0 else 0,
                "brake": -vehicle.throttle_brake if vehicle.throttle_brake < 0 else 0,
                "lane_index": vehicle.lane_index,
            }
            
            if telemetry_writer is None:
                telemetry_writer = csv.DictWriter(telemetry_file, fieldnames=telemetry.keys())
                telemetry_writer.writeheader()
            
            telemetry_writer.writerow(telemetry)
            telemetry_file.flush()
            
            # --- B. CAMERA ---
            if isinstance(obs, dict) and "image" in obs:
                img = obs["image"]
                if len(img.shape) == 4:
                    img = img[:, :, :, 0]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(img_dir, f"img_{i:04d}.png"), img)
            
            # --- C. LIDAR (FIXED) ---
            lidar_data = None
            try:
                if hasattr(vehicle, "lidar") and vehicle.lidar is not None:
                    lidar_config = vehicle.config["lidar"]
                    
                    # perceive() returns a tuple: (cloud_points, detected_objects)
                    perceive_result = vehicle.lidar.perceive(
                        vehicle,
                        physics_world,
                        lidar_config["num_lasers"],
                        lidar_config["distance"]
                    )
                    
                    # FIX: Unpack the tuple
                    if isinstance(perceive_result, tuple) or isinstance(perceive_result, list):
                        lidar_data = perceive_result[0] # Index 0 is the Point Cloud
                    else:
                        lidar_data = perceive_result

            except Exception as e:
                if i == 0: print(f"Lidar error: {e}")

            if lidar_data is not None:
                # Ensure it is a numpy array before saving
                lidar_data = np.array(lidar_data)
                np.save(os.path.join(lidar_dir, f"lidar_{i:04d}.npy"), lidar_data)
            
            if d:
                print(f"Episode ended at step {i}")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        telemetry_file.close()
        env.close()
        print(f"\nData collection complete. Saved to {data_dir}")

if __name__ == "__main__":
    interactive_drive()
