# isp.py
import numpy as np
import cv2
from awb import awb_analysis
import lsc

BLACK_LEVEL = 9

# 预计算 sRGB gamma LUT (8-bit)
LUT = np.zeros(256, dtype=np.uint8)
for i in range(256):
    lin = i / 255.0
    if lin < 0.0031308:
        srgb = 12.92 * lin
    else:
        srgb = 1.055 * (lin ** (1 / 2.4)) - 0.055
    LUT[i] = np.clip(round(srgb * 255.0), 0, 255)

def reinhard_tonemap(bgr_float: np.ndarray) -> np.ndarray:
    Y_ori = 0.0722 * bgr_float[:, :, 0] + 0.7152 * bgr_float[:, :, 1] + 0.2126 * bgr_float[:, :, 2]
    Y = Y_ori.copy()
    # double AE 18°灰
    Y[Y < 0.003] = 0.003
    mean = np.exp(np.mean(np.log(Y)))
    k = 0.18 / mean
    Y = Y * k
    x = Y
    # --- Naka-Rushton TMO (基于视觉生理响应曲线) ---
    n = 1.5  # 响应斜率，可调 0.8–1.2
    sigma = 0.18  # 半饱和亮度，可随场景调节
    Y = Y ** n
    mapped_Y = Y / (Y + sigma ** n)
    mapped_Y = np.clip(mapped_Y, 0, 1)
    # --- 保持色相与比例映射 ---
    epsilon = 1e-4
    ratio = mapped_Y / (Y_ori + epsilon)
    # --- 限制最大放大比例
    ratio = np.clip(ratio, 0.5, 4)
    tonemapped_bgr = bgr_float * ratio[:, :, np.newaxis]
    return np.clip(tonemapped_bgr, 0.0, 1.0)


def isp_process(img_float: np.ndarray) -> np.ndarray:
    """
    ISP 处理流程：
    - 输入: float32 (h, w), 范围 0-1 (RAW Bayer 图像)
    - 输出: uint8 (h, w, 3), sRGB RGB 图像
    """
    # 步骤1: 转换为 uint16 (0-65535)
    img_uint16 = (img_float * 65535.0).astype(np.uint16)
    
    # 步骤2: 减去黑电平 (9 * 256 = 2304)
    black_level = BLACK_LEVEL * 256
    img_corrected = np.clip(img_uint16.astype(np.int32) - black_level, 0, 65535).astype(np.uint16)
    
    # 步骤3: 去马赛克 (demosaicing)，转换为线性 BGR (uint16)
    linear_bgr_uint16 = cv2.cvtColor(img_corrected, cv2.COLOR_BayerBGGR2BGR)
    
    # 步骤4: 转换为 float32 (0-1)
    linear_bgr_float = linear_bgr_uint16.astype(np.float32) / 65535.0    
    
    # 步骤4.1: LSC
    linear_bgr_float = lsc.apply_lsc(linear_bgr_float)
    linear_bgr_float = np.clip(linear_bgr_float, 0.0, 1.0)

    # 步骤5.1: 白平衡和CCM计算
    k_b, k_r, ccm = awb_analysis(linear_bgr_float)
    linear_bgr_float[:,:,0] *= k_b
    linear_bgr_float[:,:,2] *= k_r
    wb_bgr_float = np.clip(linear_bgr_float, 0.0, 1.0)
    
    # 步骤5.2: 应用CCM (先转换为RGB，应用矩阵，再转回BGR)
    wb_rgb_float = wb_bgr_float[..., ::-1]  # BGR to RGB
    h, w = wb_rgb_float.shape[:2]
    corrected_rgb = (wb_rgb_float.reshape(-1, 3) @ ccm.T).reshape(h, w, 3)
    corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
    corrected_bgr = corrected_rgb[..., ::-1]  # RGB to BGR

    # 步骤6: 高斯滤波
    #corrected_bgr = cv2.GaussianBlur(corrected_bgr, (5, 5), sigmaX=0)
    
    # 步骤7: TMO
    tonemapped_bgr = reinhard_tonemap(corrected_bgr)
    
    # 步骤8: 线性转 sRGB (使用预计算 LUT)
    tonemapped_uint8 = np.clip(tonemapped_bgr * 255.0, 0, 255).astype(np.uint8)
    srgb_uint8 = LUT[tonemapped_uint8]

    return srgb_uint8  # 返回 BGR uint8，便于 cv2.imshow 和 putText