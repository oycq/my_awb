# offline_132_ffmpeg_with_preview.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import cv2
import numpy as np
import isp
import glob
import subprocess
import shutil
from tqdm import tqdm
import tempfile

# 配置参数（手动修改这些值）
folder = "records/20251106_1516_cloud"  # 输入存储路径
output_video = "output.mp4"  # 输出MP4文件名
bitrate = "4M"  # 比特率
fps = 10  # 假设帧率

# 获取所有PNG文件，按数字排序
png_files = glob.glob(os.path.join(folder, "*.png"))
sorted_files = sorted(png_files, key=lambda x: int(os.path.basename(x).split('.')[0]))

if not sorted_files:
    print("No PNG files found in the folder.")
    exit()

# 创建临时目录保存处理后的图像
temp_dir = tempfile.mkdtemp()
print(f"Temporary directory created: {temp_dir}")

total_frames = len(sorted_files)

# 处理并保存图像
with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
    for idx, file in enumerate(sorted_files):
        # 读取16位PNG
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        
        # 转换为float32 (0-1)
        img = img.astype(np.float32) / 65535.0
        
        # 处理图像
        processed_img = isp.isp_process(img)
        
        # 假设processed_img是float类型，需要转换为uint8用于显示和保存
        if processed_img.dtype == np.float32 or processed_img.dtype == np.float64:
            uint8_img = (processed_img * 255).astype(np.uint8)
        else:
            uint8_img = processed_img
        
        # 显示当前帧
        cv2.imshow("Current Frame", uint8_img)
        cv2.waitKey(1)
        
        # 保存到临时目录，命名如0001.png, 0002.png等
        temp_file = os.path.join(temp_dir, f"{idx+1:04d}.png")
        cv2.imwrite(temp_file, uint8_img)
        
        pbar.update(1)

# 销毁窗口
cv2.destroyAllWindows()

# 使用ffmpeg将临时PNG序列转换为MP4
input_pattern = os.path.join(temp_dir, "%04d.png")
ffmpeg_cmd = [
    "ffmpeg",
    "-y",  # 覆盖输出文件
    "-framerate", str(fps),
    "-i", input_pattern,
    "-c:v", "libx265",
    "-b:v", bitrate,
    "-pix_fmt", "yuv420p",
    output_video
]

print("Running ffmpeg...")
subprocess.run(ffmpeg_cmd, check=True)

# 清理临时目录
shutil.rmtree(temp_dir)
print(f"Temporary directory removed: {temp_dir}")
print(f"Video saved as {output_video}")