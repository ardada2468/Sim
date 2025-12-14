from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

def check_shape():
    config = {
        "use_render": False, # Changed to False
        "manual_control": False, # Changed to False
        "traffic_density": 0.3,
        "map": "SCrR", 
        "num_scenarios": 1,
        "start_seed": 42,
        "image_observation": False,
        "sensors": {
            "rgb_camera": (RGBCamera, 256, 256),
        },
        "vehicle_config": {
            "image_source": "main_camera",
            "lidar": {
                "num_lasers": 240,
                "distance": 50,
                "num_others": 4 # Added radar
            },
        }
    }
    
    env = MetaDriveEnv(config)
    print(f"Observation space shape: {env.observation_space.shape}")
    env.close()

if __name__ == "__main__":
    check_shape()
