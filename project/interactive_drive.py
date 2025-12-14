# import argparse
# import os
# import cv2
# import csv
# import numpy as np
# import pandas as pd
# from metadrive.envs.metadrive_env import MetaDriveEnv
# from metadrive.component.sensors.rgb_camera import RGBCamera
# from metadrive.component.sensors.lidar import Lidar
# from metadrive.constants import HELP_MESSAGE

# def interactive_drive():
#     # 1. CONFIGURATION
#     config = {
#         "use_render": True,
#         "manual_control": True,
#         "traffic_density": 0.3,
#         "map": "SCrR",
#         "num_scenarios": 1,
#         "start_seed": 42,
#         "image_observation": True, 
#         # We attach the Lidar, but we will access it manually
#         "sensors": {
#             "rgb_camera": (RGBCamera, 256, 256),
#             "lidar": (Lidar, ),
#         },
#         "vehicle_config": {
#             "image_source": "rgb_camera",
#             "lidar": {
#                 "num_lasers": 240,
#                 "distance": 50,
#                 "num_others": 4
#             },
#         }
#     }
    
#     # 2. INITIALIZE ENVIRONMENT
#     env = MetaDriveEnv(config)
    
#     # Create data directories
#     data_dir = "data_interactive"
#     img_dir = os.path.join(data_dir, "images")
#     lidar_dir = os.path.join(data_dir, "lidar")
#     os.makedirs(img_dir, exist_ok=True)
#     os.makedirs(lidar_dir, exist_ok=True)
    
#     # Open telemetry CSV file
#     telemetry_path = os.path.join(data_dir, "telemetry.csv")
#     telemetry_file = open(telemetry_path, 'w', newline='')
#     telemetry_writer = None
    
#     try:
#         print(HELP_MESSAGE)
#         print(f"Collecting data to: {data_dir}")
#         obs, info = env.reset()
        
#         # Get the physics world (needed for manual Lidar perception)
#         # This is the "God view" object that the Lidar calculates intersections against
#         physics_world = env.engine.physics_world.dynamic_world
        
#         for i in range(300):
#             action = [0.0, 0.5] # Straight, moderate speed
            
#             # Step the environment
#             step_result = env.step(action)
            
#             if len(step_result) == 5:
#                 obs, r, terminated, truncated, info = step_result
#                 d = terminated or truncated
#             else:
#                 obs, r, d, info = step_result
            
#             vehicle = env.agent
            
#             # --- DEBUG INFO (First Step Only) ---
#             if i == 0:
#                 print("\n=== SENSOR DEBUG ===")
#                 if hasattr(vehicle, 'lidar') and vehicle.lidar is not None:
#                     print(f"✓ Lidar hardware found on vehicle")
#                 else:
#                     print("✗ No Lidar hardware found!")
#                 print("====================\n")

#             # --- A. COLLECT TELEMETRY ---
#             telemetry = {
#                 "step": i,
#                 "speed": vehicle.speed,
#                 "heading": vehicle.heading_theta,
#                 "position_x": vehicle.position[0],
#                 "position_y": vehicle.position[1],
#                 "steering": vehicle.steering,
#                 "throttle": vehicle.throttle_brake if vehicle.throttle_brake > 0 else 0,
#                 "brake": -vehicle.throttle_brake if vehicle.throttle_brake < 0 else 0,
#                 "lane_index": vehicle.lane_index,
#             }
            
#             if telemetry_writer is None:
#                 telemetry_writer = csv.DictWriter(telemetry_file, fieldnames=telemetry.keys())
#                 telemetry_writer.writeheader()
            
#             telemetry_writer.writerow(telemetry)
#             telemetry_file.flush()
            
#             # --- B. COLLECT CAMERA IMAGE ---
#             if isinstance(obs, dict) and "image" in obs:
#                 img = obs["image"]
#                 if len(img.shape) == 4:
#                     img = img[:, :, :, 0]
#                 if img.max() <= 1.0:
#                     img = (img * 255).astype(np.uint8)
#                 else:
#                     img = img.astype(np.uint8)
#                 if img.shape[-1] == 3:
#                     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
#                 cv2.imwrite(os.path.join(img_dir, f"img_{i:04d}.png"), img)
            
#             # --- C. COLLECT LIDAR DATA (Manual Perceive Method) ---
#             # This forces the sensor to calculate rays RIGHT NOW
#             lidar_data = None
#             try:
#                 if hasattr(vehicle, "lidar") and vehicle.lidar is not None:
#                     lidar_config = vehicle.config["lidar"]
                    
#                     # Manual perception call
#                     lidar_data = vehicle.lidar.perceive(
#                         vehicle,
#                         physics_world,
#                         lidar_config["num_lasers"],
#                         lidar_config["distance"]
#                     )
#             except Exception as e:
#                 if i == 0: print(f"Lidar error: {e}")

#             if lidar_data is not None:
#                 np.save(os.path.join(lidar_dir, f"lidar_{i:04d}.npy"), lidar_data)
#             else:
#                 if i == 0:
#                     print("Warning: Lidar perceive() returned None.")

#             if d:
#                 print(f"Episode ended at step {i}")
#                 obs, info = env.reset()
                
#     except KeyboardInterrupt:
#         print("\nInterrupted by user")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         telemetry_file.close()
#         env.close()
#         print(f"\nData collection complete. Saved to {data_dir}")

# if __name__ == "__main__":
#     interactive_drive()
import argparse
import os
import cv2
import csv
import numpy as np
import pandas as pd
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.lidar import Lidar
from metadrive.constants import HELP_MESSAGE

def interactive_drive():
    # 1. CONFIGURATION
    config = {
        "use_render": True,
        "manual_control": True,
        "traffic_density": 0.3,
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
    
    data_dir = "data_interactive"
    img_dir = os.path.join(data_dir, "images")
    lidar_dir = os.path.join(data_dir, "lidar")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)
    
    telemetry_path = os.path.join(data_dir, "telemetry.csv")
    telemetry_file = open(telemetry_path, 'w', newline='')
    telemetry_writer = None
    
    try:
        print(HELP_MESSAGE)
        print(f"Collecting data to: {data_dir}")
        obs, info = env.reset()
        
        # Get physics world for ray casting
        physics_world = env.engine.physics_world.dynamic_world
        
        for i in range(300):
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