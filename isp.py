# isp.py
import numpy as np
import cv2
from awb import awb_analysis
import lsc

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

    # 步骤2.1: LSC
    img = lsc.apply_lsc(img)
    img = np.clip(img, 0.0, 255.0)
    
    # 转换为 uint8 以进行 demosaicing (假设 Bayer BG 图案)
    img_uint8 = img.astype(np.uint8)
    
    # 步骤3: 去马赛克 (demosaicing)，转换为线性 BGR
    linear_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_BayerBGGR2BGR_VNG)
    
    # 转换为 float32 (0-1) 以进行白平衡和 gamma 校正
    linear_bgr_float = linear_bgr.astype(np.float32) / 255.0
    
    # 步骤4: 白平衡和CCM计算
    k_b, k_r, ccm = awb_analysis(linear_bgr_float)
    linear_bgr_float[:,:,0] *= k_b
    linear_bgr_float[:,:,2] *= k_r
    wb_bgr_float = np.clip(linear_bgr_float, 0.0, 1.0)
    
    # 步骤4.1: 应用CCM (先转换为RGB，应用矩阵，再转回BGR)
    wb_rgb_float = wb_bgr_float[..., ::-1]  # BGR to RGB
    h, w = wb_rgb_float.shape[:2]
    corrected_rgb = (wb_rgb_float.reshape(-1, 3) @ ccm.T).reshape(h, w, 3)
    corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
    corrected_bgr = corrected_rgb[..., ::-1]  # RGB to BGR
    
    # 步骤5: 线性转 sRGB (使用预计算 LUT)
    corrected_uint8 = np.clip(corrected_bgr * 255.0, 0, 255).astype(np.uint8)
    srgb_uint8 = LUT[corrected_uint8]

    return srgb_uint8  # 返回 BGR uint8，便于 cv2.imshow 和 putText