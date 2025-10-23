# 132.py
import cv2
import time
import numpy as np
import cam_client  # 导入 cam_client 模块

# 初始化 VideoCapture
cap = cam_client.VideoCapture()

# 统计FPS
frame_count = 0
fps = 0.0
start_time = time.time()

while True:
    img = cap.read()  # 阻塞读取图像

    # 统计帧率
    frame_count += 1
    curr_time = time.time()
    elapsed_time = curr_time - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = curr_time

    # 显示帧率
    text = f"FPS: {fps:.2f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (1.0, 1.0, 1.0), 2, cv2.LINE_AA)

    # 显示图像 (注意: img 是 float32 0-1，需要转换为 uint8 以显示)
    cv2.imshow("Received Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()