#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json

MIN_THRESHOLD = 10 / 255.0
MAX_THRESHOLD = 200 / 255.0

cv2.namedWindow('AWB Scatter')
cv2.setWindowProperty('AWB Scatter', cv2.WND_PROP_TOPMOST, 1)

# 加载 JSON 数据（全局加载一次）
try:
    with open('results.json', 'r') as f:
        data = json.load(f)
    calibration_results = data['calibration_results']
    white_point_regions = data['white_point_regions']
except FileNotFoundError:
    print("Warning: results.json not found. Reference points and white point regions will not be plotted.")
    calibration_results = {}
    white_point_regions = []

# 提取参考点
reference_rg_bg = []
reference_labels = []
for key, value in calibration_results.items():
    awb = value['awb']
    r_gain, _, b_gain = awb
    if r_gain != 0 and b_gain != 0:
        rg = 1 / r_gain  # R/G
        bg = 1 / b_gain  # B/G
        reference_rg_bg.append((rg, bg))
        reference_labels.append(key)

def awb_analysis(img):
    original_img = cv2.resize(img, (32, 32)).astype(np.float32)
    b_mean = original_img[:, :, 0].mean()
    g_mean = original_img[:, :, 1].mean()
    r_mean = original_img[:, :, 2].mean()
    k_b = g_mean / b_mean if b_mean != 0 else 1.0
    k_r = g_mean / r_mean if r_mean != 0 else 1.0
    
    balanced_img = np.copy(original_img)
    balanced_img[:, :, 0] *= k_b
    balanced_img[:, :, 2] *= k_r
    balanced_img = np.clip(balanced_img, 0, 1)
    
    analysis_scatter(original_img, balanced_img)
    return k_b, k_r

def analysis_scatter(original_img, balanced_img):
    IMG_SHAPE = 450
    scale = IMG_SHAPE / 1.5
    
    # 初始化背景为 (20/255, 20/255, 20/255)
    scatter_img = np.full((IMG_SHAPE, IMG_SHAPE, 3), (20 / 255.0, 20 / 255.0, 20 / 255.0), np.float32)
    
    # 先绘制白点区域矩形（最底层）
    for region in white_point_regions:
        x0, x1, y0, y1 = region
        px_min = int(x0 * scale)
        px_max = int(x1 * scale)
        py_for_y_max = IMG_SHAPE - 1 - int(y1 * scale)  # 上部 (小 py)
        py_for_y_min = IMG_SHAPE - 1 - int(y0 * scale)  # 下部 (大 py)
        if px_min < px_max and py_for_y_max < py_for_y_min:
            cv2.rectangle(scatter_img, (px_min, py_for_y_max), (px_max, py_for_y_min), (0.2, 0.2, 0.2), -1)  # 填充矩形
    
    # 绘制像素散点
    for i in range(32):
        for j in range(32):
            b, g, r = original_img[i, j]
            if g == 0:
                continue
            if not (MIN_THRESHOLD < b < MAX_THRESHOLD and MIN_THRESHOLD < g < MAX_THRESHOLD and MIN_THRESHOLD < r < MAX_THRESHOLD):
                continue
            x = r / g
            y = b / g
            px = int(x * scale)
            py = IMG_SHAPE - 1 - int(y * scale)  # invert y for image coordinates (0 at top)
            if 0 <= px < IMG_SHAPE and 0 <= py < IMG_SHAPE:
                scatter_img[py, px] = balanced_img[i, j] * 2
    
    # 绘制参考点和标签
    for (rg, bg), label in zip(reference_rg_bg, reference_labels):
        px = int(rg * scale)
        py = IMG_SHAPE - 1 - int(bg * scale)
        if 0 <= px < IMG_SHAPE and 0 <= py < IMG_SHAPE:
            cv2.circle(scatter_img, (px, py), 5, (1, 0, 0), 2)
    for (rg, bg), label in zip(reference_rg_bg, reference_labels):
        px = int(rg * scale)
        py = IMG_SHAPE - 1 - int(bg * scale)
        if 0 <= px < IMG_SHAPE and 0 <= py < IMG_SHAPE:
            cv2.putText(scatter_img, label, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)
    
    # Show the scatter
    cv2.imshow('AWB Scatter', scatter_img)