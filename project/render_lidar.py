import cv2
import numpy as np
import pandas as pd
import os
import glob

def render_lidar_video():
    data_dir = "data_interactive"
    lidar_dir = os.path.join(data_dir, "lidar")
    img_dir = os.path.join(data_dir, "images")
    telemetry_path = os.path.join(data_dir, "telemetry.csv")
    output_video = "lidar_telemetry.mp4"
    
    # Check if data exists
    if not os.path.exists(telemetry_path):
        print(f"Telemetry file not found at {telemetry_path}")
        return
    if not os.path.exists(lidar_dir):
        print(f"Lidar directory not found at {lidar_dir}")
        return

    # Load telemetry
    df = pd.read_csv(telemetry_path)
    
    # Get list of lidar files
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "obs_*.npy")))
    if not lidar_files:
        print("No lidar files found.")
        return

    print(f"Found {len(lidar_files)} lidar frames.")

    # Video settings
    lidar_width, lidar_height = 800, 800
    img_width, img_height = 800, 800
    total_width = lidar_width + img_width
    height = lidar_height
    
    fps = 20 # Assuming approx 20Hz
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video, fourcc, fps, (total_width, height))
    
    # Lidar settings
    num_lasers = 240
    num_others = 4
    max_distance = 50
    # Create angles: 0 to 360 degrees.
    angles = np.linspace(0, 2 * np.pi, num_lasers, endpoint=False)
    
    for i, lidar_file in enumerate(lidar_files):
        # Extract step number from filename
        basename = os.path.basename(lidar_file)
        # obs_0000.npy -> 0
        try:
            step_str = basename.split('_')[1].split('.')[0]
            step = int(step_str)
        except:
            print(f"Skipping file with unexpected name format: {basename}")
            continue
            
        # Get telemetry for this step
        row = df[df['step'] == step]
        if row.empty:
            telemetry = {}
        else:
            telemetry = row.iloc[0].to_dict()
        
        # Load Lidar
        try:
            obs = np.load(lidar_file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {lidar_file}: {e}")
            continue

        # Check observation size to determine if radar data is present
        has_radar = False
        
        if obs.shape[0] >= num_lasers + num_others * 4:
             # Assume radar is present before lidar
             radar_len = num_others * 4
             radar_start_idx = -num_lasers - radar_len
             radar_data = obs[radar_start_idx:-num_lasers]
             lidar_data = obs[-num_lasers:]
             has_radar = True
        elif obs.shape[0] >= num_lasers:
             lidar_data = obs[-num_lasers:]
        else:
            print(f"Observation shape {obs.shape} smaller than num_lasers {num_lasers}")
            continue
        
        # Create Lidar frame (black background)
        lidar_frame = np.zeros((lidar_height, lidar_width, 3), dtype=np.uint8)
        
        # Draw Lidar points
        cx, cy = lidar_width // 2, lidar_height // 2
        scale = 6 # pixels per meter. 50m * 6 = 300px radius.
        
        # Draw range circles
        cv2.circle(lidar_frame, (cx, cy), int(10 * scale), (50, 50, 50), 1)
        cv2.circle(lidar_frame, (cx, cy), int(25 * scale), (50, 50, 50), 1)
        cv2.circle(lidar_frame, (cx, cy), int(50 * scale), (50, 50, 50), 1)
        
        # Draw axes
        cv2.line(lidar_frame, (cx, cy), (cx, cy-50), (100, 100, 100), 1) # Forward
        cv2.line(lidar_frame, (cx, cy), (cx+50, cy), (100, 100, 100), 1) # Right
        
        for j, dist_norm in enumerate(lidar_data):
            dist_m = dist_norm * max_distance
            angle = angles[j] # 0 is Forward, increases ccw (towards Left)
            
            # x is Forward, y is Left
            x_rel = dist_m * np.cos(angle)
            y_rel = dist_m * np.sin(angle)
            
            # Map to screen: Forward -> Up (y-), Left -> Left (x-)
            # Wait, usually Right is x+.
            # So Left is x-.
            # px = cx - y_rel * scale
            # py = cy - x_rel * scale
            
            px = cx - int(y_rel * scale)
            py = cy - int(x_rel * scale)
            
            b = 0
            g = int(255 * dist_norm)
            r = int(255 * (1 - dist_norm))
            cv2.circle(lidar_frame, (px, py), 2, (b, g, r), -1)

        # Draw Radar
        if has_radar:
            for k in range(num_others):
                idx = k * 4
                rel_x_norm = radar_data[idx]
                rel_y_norm = radar_data[idx+1]
                
                if np.all(radar_data[idx:idx+4] == 0.0):
                    continue
                
                rel_x = (rel_x_norm * 2 - 1) * max_distance # Forward
                rel_y = (rel_y_norm * 2 - 1) * max_distance # Left
                
                px = cx - int(rel_y * scale)
                py = cy - int(rel_x * scale)
                
                # Draw Box for vehicle
                # Blue box
                cv2.rectangle(lidar_frame, (px-5, py-5), (px+5, py+5), (255, 0, 0), 2)
                cv2.putText(lidar_frame, f"V{k}", (px+8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

        # Draw Car (Triangle pointing Up)
        pts = np.array([[cx, cy-10], [cx-5, cy+5], [cx+5, cy+5]], np.int32)
        cv2.fillPoly(lidar_frame, [pts], (0, 255, 255))
        
        # Overlay Telemetry on Lidar Frame
        y_offset = 30
        x_offset = 10
        
        # Title
        cv2.putText(lidar_frame, f"Step: {step}", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25
        
        for key, value in telemetry.items():
            if key == "step": continue
            if isinstance(value, float):
                text = f"{key}: {value:.4f}"
            else:
                text = f"{key}: {value}"
            cv2.putText(lidar_frame, text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
            
        # Load and Process Image Frame
        img_path = os.path.join(img_dir, f"img_{step:04d}.png")
        if os.path.exists(img_path):
            img_frame = cv2.imread(img_path)
            if img_frame is None:
                print(f"Failed to load image: {img_path}")
                img_frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            else:
                img_frame = cv2.resize(img_frame, (img_width, img_height))
        else:
            # print(f"Image not found: {img_path}")
            img_frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            cv2.putText(img_frame, "No Image", (img_width//2 - 50, img_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine Frames
        combined_frame = np.hstack((lidar_frame, img_frame))
        
        out.write(combined_frame)
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(lidar_files)} frames...")
            
    out.release()
    print(f"Video saved to {output_video}")

if __name__ == "__main__":
    render_lidar_video()
