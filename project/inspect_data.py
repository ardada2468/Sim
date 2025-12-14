import numpy as np
import os

lidar_dir = "data_interactive/lidar"
files = sorted(os.listdir(lidar_dir))
if not files:
    print("No files found")
    exit()

first_file = files[0]
path = os.path.join(lidar_dir, first_file)
data = np.load(path, allow_pickle=True)

print(f"File: {first_file}")
print(f"Type: {type(data)}")
print(f"Shape: {data.shape}")
print(f"Dtype: {data.dtype}")

if isinstance(data, np.ndarray):
    print("First 20 elements:", data.flatten()[:20])
    print("Min:", np.min(data))
    print("Max:", np.max(data))
