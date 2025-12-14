import os
import cv2
import numpy as np
import pandas as pd
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE

# Configuration for "Indian Roads" simulation
# High density to simulate congestion
# Complex map string to simulate varied road structures
INDIAN_ROADS_CONFIG = {
    "use_render": True,  # Set to True if you want to see the window, False for headless/faster data collection
    "traffic_density": 0.3,  # High traffic density
    "map": "SCrR",  # S: Straight, C: Circular, r: Ramp, R: Roundabout - varied topology
    "num_scenarios": 1,
    "start_seed": 42,
    "image_observation": False, # Disable image in observation due to window issue
    "sensors": {
        "rgb_camera": (RGBCamera, 256, 256), # Register RGB Camera, 256x256 resolution
    },
    "vehicle_config": {
        # "image_source": "rgb_camera",
        "lidar": {
            "num_lasers": 240,
            "distance": 50,
            "num_others": 0
        },  # Enable Lidar
    },
}

def collect_data(num_steps=1000):
    env = MetaDriveEnv(INDIAN_ROADS_CONFIG)
    
    # Create data directories
    data_dir = "data"
    img_dir = os.path.join(data_dir, "images")
    lidar_dir = os.path.join(data_dir, "lidar")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)
    
    telemetry_data = []
    
    try:
        # MetaDrive 0.2.6.0 likely uses old Gym API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        
        for i in range(num_steps):
            # Random action for now, or simple lane keeping if available
            # Using [0, 1] for throttle to move forward
            action = [0.0, 0.5] 
            
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            
            # 1. Collect Telemetry
            vehicle = env.agent # Use env.agent for newer MetaDrive
            telemetry = {
                "step": i,
                "speed": vehicle.speed,
                "heading": vehicle.heading_theta,
                "position_x": vehicle.position[0],
                "position_y": vehicle.position[1],
                "steering": action[0],
                "throttle": action[1],
                "brake": 0.0, # Assuming no brake in this simple action
                "lane_index": vehicle.lane_index,
            }
            telemetry_data.append(telemetry)
            
            # 2. Collect Camera Data
            # Camera disabled due to window issues
            # if isinstance(obs, dict) and "image" in obs:
            #     ...
            
            # 3. Collect Lidar/Observation Data
            # Lidar ranges are in the observation
            # vehicle.lidar.get_cloud_points() might be deprecated or changed
            np.save(os.path.join(lidar_dir, f"obs_{i:04d}.npy"), obs)
            
            if (i + 1) % 100 == 0:
                print(f"Step {i+1}/{num_steps} completed.")
                
            if terminated or truncated:
                env.reset()
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()
        
        # Save Telemetry
        df = pd.DataFrame(telemetry_data)
        df.to_csv(os.path.join(data_dir, "telemetry.csv"), index=False)
        print(f"Data collection complete. Saved to {data_dir}")

if __name__ == "__main__":
    collect_data()
