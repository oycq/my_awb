# 132.py
import cv2
import time
import numpy as np
import cam_client
import isp

SCALE = 0.75  # 显示压缩比例

cap = cam_client.VideoCapture()

frame_count = 0
fps = 0.0
start_time = time.time()

while True:
    img = cap.read()
    processed_img = isp.isp_process(img)
    
    frame_count += 1
    curr_time = time.time()
    elapsed_time = curr_time - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = curr_time
    
    text = f"FPS: {fps:.2f}"
    cv2.putText(processed_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 压缩显示尺寸
    h, w = processed_img.shape[:2]
    resized_img = cv2.resize(processed_img, (int(w * SCALE), int(h * SCALE)))
    
    cv2.imshow("Received Image", resized_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()