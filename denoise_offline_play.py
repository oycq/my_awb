# offline_132.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import cv2
import time
import numpy as np
import isp
import os
import glob

SCALE = 0.75  # 显示压缩比例

# 配置参数（手动修改这些值）
denoise_folder = "data_denoise"  # 去噪图像路径
original_folder = "records/20251106_1516_cloud"  # 原图路径

# 获取所有PNG文件，按数字排序（基于denoise_folder）
png_files = glob.glob(os.path.join(denoise_folder, "*.png"))
# 提取文件名中的数字并排序
sorted_files = sorted(png_files, key=lambda x: int(os.path.basename(x).split('.')[0]))

if not sorted_files:
    print("No PNG files found in the denoise folder.")
    exit()

# 窗口名称
window_name = "Processed Images (Original | Denoised)"

# 创建窗口
cv2.namedWindow(window_name)

# 当前帧索引
current_frame = 0

# 播放状态
is_playing = False

# 播放速度 (ms per frame, ~30 FPS)
frame_delay = 33

# 缓存
last_frame = -1
cached_original_img = None
cached_denoised_img = None

# 轨道条回调函数
def on_trackbar(pos):
    pass  # 可以在这里添加刷新逻辑，但为了简单，保持pass

# 创建轨道条
cv2.createTrackbar("Frame", window_name, 0, len(sorted_files) - 1, on_trackbar)

frame_count = 0
fps = 0.0
start_time = time.time()

while True:
    # 获取当前轨道条位置
    current_frame = cv2.getTrackbarPos("Frame", window_name)
    
    # 确保索引在范围内
    current_frame = max(0, min(current_frame, len(sorted_files) - 1))
    
    # 如果帧改变了或正在播放，处理新帧
    if current_frame != last_frame:
        filename = os.path.basename(sorted_files[current_frame])
        
        # 读取去噪图像
        denoise_file = os.path.join(denoise_folder, filename)
        denoised_raw = cv2.imread(denoise_file, cv2.IMREAD_UNCHANGED)
        if denoised_raw is None:
            continue
        denoised_raw = denoised_raw.astype(np.float32) / 65535.0
        cached_denoised_img = isp.isp_process(denoised_raw)
        
        # 读取原图
        original_file = os.path.join(original_folder, filename)
        original_raw = cv2.imread(original_file, cv2.IMREAD_UNCHANGED)
        if original_raw is None:
            continue
        original_raw = original_raw.astype(np.float32) / 65535.0
        cached_original_img = isp.isp_process(original_raw)
        
        last_frame = current_frame
    
    if cached_original_img is None or cached_denoised_img is None:
        continue
    
    # 合并图像：原图 | 去噪图（一行两列）
    merged_img = np.hstack((cached_original_img, cached_denoised_img))
    
    # 计算FPS（仅当帧改变或播放时更新）
    if current_frame != last_frame or is_playing:
        frame_count += 1
        curr_time = time.time()
        elapsed_time = curr_time - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = curr_time
    
    # 添加文本
    status = "Playing" if is_playing else "Paused"
    text = f"FPS: {fps:.2f} | Frame: {current_frame + 1}/{len(sorted_files)} | {status}"
    cv2.putText(merged_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 压缩显示尺寸
    h, w = merged_img.shape[:2]
    resized_img = cv2.resize(merged_img, (int(w * SCALE), int(h * SCALE)))
    
    cv2.imshow(window_name, resized_img)
    
    # 等待按键，播放时使用frame_delay，暂停时使用1ms（以响应轨道条）
    wait_time = frame_delay if is_playing else 1
    key = cv2.waitKey(wait_time) & 0xFF
    
    if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
        break
    elif key == ord(' '):  # 空格切换播放/暂停
        is_playing = not is_playing
    elif key == ord('n'):  # 按'n'下一帧
        current_frame = min(current_frame + 1, len(sorted_files) - 1)
        cv2.setTrackbarPos("Frame", window_name, current_frame)
    elif key == ord('p'):  # 按'p'上一帧
        current_frame = max(current_frame - 1, 0)
        cv2.setTrackbarPos("Frame", window_name, current_frame)
    
    # 如果在播放，自动前进一帧
    if is_playing:
        current_frame = min(current_frame + 1, len(sorted_files) - 1)
        cv2.setTrackbarPos("Frame", window_name, current_frame)
        if current_frame == len(sorted_files) - 1:
            is_playing = False  # 到达末尾停止

cv2.destroyAllWindows()