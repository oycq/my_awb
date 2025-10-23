# isp.py
import numpy as np
import cv2

# 预计算 sRGB gamma LUT (8-bit)
LUT = np.zeros(256, dtype=np.uint8)
for i in range(256):
    lin = i / 255.0
    if lin < 0.0031308:
        srgb = 12.92 * lin
    else:
        srgb = 1.055 * (lin ** (1 / 2.4)) - 0.055
    LUT[i] = np.clip(round(srgb * 255.0), 0, 255)

def isp_process(img_float: np.ndarray) -> np.ndarray:
    """
    ISP 处理流程：
    - 输入: float32 (h, w), 范围 0-1 (RAW Bayer 图像)
    - 输出: uint8 (h, w, 3), sRGB RGB 图像
    """
    # 步骤1: 乘 255 转换为 0-255 范围
    img = img_float * 255.0
    
    # 步骤2: 减去黑电平 (9.125)
    img -= 9.125
    img = np.clip(img, 0.0, 255.0)
    
    # 转换为 uint8 以进行 demosaicing (假设 Bayer BG 图案，常见于许多传感器；如果不同，可调整)
    img_uint8 = img.astype(np.uint8)
    
    # 步骤3: 去马赛克 (demosaicing)，转换为线性 BGR
    linear_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_BayerBGGR2BGR)
    
    # 转换为 float32 (0-1) 以进行白平衡和 gamma 校正
    linear_rgb_float = linear_bgr.astype(np.float32) / 255.0  # 注意: OpenCV 是 BGR，所以这里是 linear BGR，但通道处理相同
    
    # 步骤4: 白平衡 (以绿色通道为参考，使 r 和 b 的均值等于 g 的均值，g 不变)
    avgs = np.mean(linear_rgb_float, axis=(0,1))  # [avg_b, avg_g, avg_r]
    ref_g = avgs[1]
    scales = np.array([ref_g / avgs[0] if avgs[0] > 0 else 1.0, 1.0, ref_g / avgs[2] if avgs[2] > 0 else 1.0])
    wb_rgb_float = np.clip(linear_rgb_float * scales[None, None, :], 0.0, 1.0)
    
    # 步骤5: 线性转 sRGB (使用预计算 LUT)
    wb_uint8 = np.clip(wb_rgb_float * 255.0, 0, 255).astype(np.uint8)
    srgb_uint8 = LUT[wb_uint8]

    return srgb_uint8  # 返回 BGR uint8，便于 cv2.imshow 和 putText