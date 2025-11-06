# 132.py
import cv2
import time
import numpy as np
import cam_client
import os
import datetime

SCALE = 0.75  # 显示压缩比例

cap = cam_client.VideoCapture()

# Create folder with timestamp under 'records'
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
folder = os.path.join("records", timestamp)
os.makedirs(folder, exist_ok=True)

frame_count = 0
total_size = 0
fps = 0.0
start_time = time.time()

while True:
    img = cap.read()
    if img is None:
        continue  # Skip if no frame
    
    img = (img * 65535).astype(np.uint16)
    
    # Save as 16-bit PNG
    save_path = os.path.join(folder, f"{1000000 + frame_count}.png")
    cv2.imwrite(save_path, img)
    total_size += os.path.getsize(save_path)
    
    # Calculate FPS and sizes
    current_time = time.time()
    duration = current_time - start_time
    fps = frame_count / duration if duration > 0 else 0.0
    size_gb = total_size / (1024 * 1024 * 1024)
    size_per_second = (total_size / (1024 * 1024)) / duration if duration > 0 else 0.0
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    
    # Prepare display image (resize and normalize to uint8 for proper display)
    display_img = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)
    if len(display_img.shape) == 2:  # Grayscale
        display_img = (display_img / 256).astype(np.uint8)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    else:  # Assume BGR
        display_img = (display_img / 256).astype(np.uint8)
    
    # Add FPS, size, and duration text
    cv2.putText(display_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_img, f"Size: {size_gb:.2f} GB", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_img, f"Duration: {minutes}m {seconds}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Received Image", display_img)
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# After loop, calculate and print final stats
final_duration = time.time() - start_time
final_size_mb = total_size / (1024 * 1024)
final_size_per_second = final_size_mb / final_duration if final_duration > 0 else 0.0

print(f"Folder: {folder}")
print(f"Total frames: {frame_count}")
print(f"Duration: {final_duration:.2f} seconds")
print(f"Folder size: {final_size_mb:.2f} MB")
print(f"Size per second: {final_size_per_second:.2f} MB/s")