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
original_folder = "records/20251106_1516_cloud"           # 原始图像路径
denoise_folder = "data_denoise"    # 降噪图像路径
output_video = "output.mp4"        # 输出MP4文件名
bitrate = "4M"                     # 比特率
fps = 10                           # 假设帧率

# 获取降噪文件夹中的PNG文件，按数字排序
png_files = glob.glob(os.path.join(denoise_folder, "*.png"))
sorted_files = sorted(png_files, key=lambda x: int(os.path.basename(x).split('.')[0]))

if not sorted_files:
    print("No PNG files found in the denoise folder.")
    exit()

# 创建临时目录保存合并后的图像
temp_dir = tempfile.mkdtemp()
print(f"Temporary directory created: {temp_dir}")

total_frames = len(sorted_files)

# 处理并保存图像（左：原图，右：降噪图）
with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
    for idx, denoise_file in enumerate(sorted_files):
        filename = os.path.basename(denoise_file)
        
        # 读取降噪图像
        denoised_raw = cv2.imread(denoise_file, cv2.IMREAD_UNCHANGED)
        if denoised_raw is None:
            print(f"Failed to read denoised image: {denoise_file}")
            continue
        denoised_raw = denoised_raw.astype(np.float32) / 65535.0
        denoised_processed = isp.isp_process(denoised_raw)
        
        # 读取原图（同名）
        original_file = os.path.join(original_folder, filename)
        original_raw = cv2.imread(original_file, cv2.IMREAD_UNCHANGED)
        if original_raw is None:
            print(f"Failed to read original image: {original_file}")
            continue
        original_raw = original_raw.astype(np.float32) / 65535.0
        original_processed = isp.isp_process(original_raw)
        
        # 转换为uint8用于显示和保存
        def to_uint8(img):
            if img.dtype == np.float32 or img.dtype == np.float64:
                return (img * 255).astype(np.uint8)
            return img
        
        original_uint8 = to_uint8(original_processed)
        denoised_uint8 = to_uint8(denoised_processed)
        
        # 合并图像：左原图，右降噪图（一行两列）
        merged_img = np.hstack((original_uint8, denoised_uint8))
        
        # 添加标题
        cv2.putText(merged_img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        h, w = original_uint8.shape[:2]
        cv2.putText(merged_img, "Denoised", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 显示当前帧
        cv2.imshow("Current Frame (Original | Denoised)", merged_img)
        cv2.waitKey(1)
        
        # 保存到临时目录，命名如0001.png, 0002.png等
        temp_file = os.path.join(temp_dir, f"{idx+1:04d}.png")
        cv2.imwrite(temp_file, merged_img)
        
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