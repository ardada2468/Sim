import os
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

def explore_lidar_api():
    config = {
        "use_render": False,  # Faster without rendering
        "manual_control": False,
        "traffic_density": 0.3,
        "map": "S",
        "num_scenarios": 1,
        "start_seed": 42,
        "image_observation": True,
        "sensors": {
            "rgb_camera": (RGBCamera, 256, 256),
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
    
    env = MetaDriveEnv(config)
    
    try:
        obs, info = env.reset()
        vehicle = env.agent
        
        print("="*60)
        print("METADRIVE LIDAR API EXPLORATION")
        print("="*60)
        
        # Check if vehicle has lidar
        print(f"\n1. Vehicle has 'lidar' attribute: {hasattr(vehicle, 'lidar')}")
        
        if hasattr(vehicle, 'lidar'):
            lidar = vehicle.lidar
            print(f"   Lidar object: {lidar}")
            print(f"   Lidar class: {lidar.__class__.__name__}")
            print(f"   Lidar module: {lidar.__class__.__module__}")
            
            # List all public methods and attributes
            print("\n2. Public attributes and methods:")
            attrs = [a for a in dir(lidar) if not a.startswith('_')]
            for attr in attrs:
                attr_obj = getattr(lidar, attr)
                if callable(attr_obj):
                    print(f"   [METHOD] {attr}")
                else:
                    print(f"   [ATTR]   {attr} = {type(attr_obj).__name__}")
            
            # Try common data access patterns
            print("\n3. Trying to access lidar data:")
            
            # Pattern 1: perceive()
            if hasattr(lidar, 'perceive'):
                try:
                    result = lidar.perceive(vehicle)
                    print(f"   ✓ lidar.perceive(vehicle): {type(result)}")
                    if hasattr(result, 'shape'):
                        print(f"     Shape: {result.shape}, dtype: {result.dtype}")
                        print(f"     Min/Max: {result.min():.2f} / {result.max():.2f}")
                        print(f"     First 10 values: {result.flatten()[:10]}")
                except Exception as e:
                    print(f"   ✗ lidar.perceive(vehicle) failed: {e}")
            
            # Pattern 2: get_surrounding_objects()
            if hasattr(lidar, 'get_surrounding_objects'):
                try:
                    result = lidar.get_surrounding_objects(vehicle)
                    print(f"   ✓ lidar.get_surrounding_objects(vehicle): {type(result)}")
                    if hasattr(result, 'shape'):
                        print(f"     Shape: {result.shape}")
                except Exception as e:
                    print(f"   ✗ lidar.get_surrounding_objects(vehicle) failed: {e}")
            
            # Pattern 3: Check for data attribute
            if hasattr(lidar, 'data'):
                print(f"   ✓ lidar.data exists: {type(lidar.data)}")
                if hasattr(lidar.data, 'shape'):
                    print(f"     Shape: {lidar.data.shape}")
            
            # Pattern 4: Check perception_result
            if hasattr(lidar, 'perception_result'):
                print(f"   ✓ lidar.perception_result: {type(lidar.perception_result)}")
                if hasattr(lidar.perception_result, 'shape'):
                    print(f"     Shape: {lidar.perception_result.shape}")
            
            # Pattern 5: Check lasers
            if hasattr(lidar, 'num_lasers'):
                print(f"   ✓ lidar.num_lasers: {lidar.num_lasers}")
            
            # Pattern 6: Try step/update
            if hasattr(lidar, 'track'):
                try:
                    vehicle.before_step([0, 0])
                    env.step([0, 0.5])
                    result = lidar.track(vehicle)
                    print(f"   ✓ lidar.track(vehicle): {type(result)}")
                    if hasattr(result, 'shape'):
                        print(f"     Shape: {result.shape}")
                except Exception as e:
                    print(f"   ✗ lidar.track(vehicle) failed: {e}")
        
        # Check observation dictionary
        print("\n4. Checking observation dictionary:")
        print(f"   Observation type: {type(obs)}")
        if isinstance(obs, dict):
            print(f"   Keys: {list(obs.keys())}")
            for key, value in obs.items():
                if hasattr(value, 'shape'):
                    print(f"     {key}: shape={value.shape}, dtype={value.dtype}")
                    if 'lidar' in key.lower() or 'cloud' in key.lower():
                        print(f"       → POSSIBLE LIDAR DATA!")
                        print(f"       Min/Max: {value.min():.2f} / {value.max():.2f}")
        
        print("\n" + "="*60)
        print("EXPLORATION COMPLETE")
        print("="*60)
        
    finally:
        env.close()

if __name__ == "__main__":
    explore_lidar_api()