import pandas as pd
import os
import glob
import numpy as np

data_dir = "data_interactive"
lidar_dir = os.path.join(data_dir, "lidar")
telemetry_path = os.path.join(data_dir, "telemetry.csv")

lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "obs_*.npy")))
steps = []
for f in lidar_files:
    try:
        step = int(os.path.basename(f).split('_')[1].split('.')[0])
        steps.append(step)
    except:
        pass

df = pd.DataFrame({
    'step': steps,
    'speed': np.random.rand(len(steps)) * 50,
    'heading': np.random.rand(len(steps)) * 360,
    'position_x': np.random.rand(len(steps)) * 100,
    'position_y': np.random.rand(len(steps)) * 100,
    'steering': np.random.rand(len(steps)) - 0.5,
    'throttle': np.random.rand(len(steps)),
    'brake': np.zeros(len(steps)),
    'lane_index': np.zeros(len(steps))
})

df.to_csv(telemetry_path, index=False)
print(f"Generated dummy telemetry for {len(steps)} steps.")
