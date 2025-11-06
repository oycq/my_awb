#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
import sys

LUM_MIN_THROAT = 10 / 255.0
LUM_MAX_THROAT = 220 / 255.0
update_ratio = 0.999

cv2.namedWindow('AWB Scatter')
cv2.setWindowProperty('AWB Scatter', cv2.WND_PROP_TOPMOST, 1)

# 加载 JSON 数据（全局加载一次）
with open('results.json', 'r') as f:
    data = json.load(f)
calibration_results = data['calibration_results']
white_point_regions = data['white_point_regions']
ccm_fit_params = data['ccm_fit_params']

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

# 当前rg, bg (初始D65)
avg_rg = 0.54
avg_bg = 0.62

def awb_analysis(img):
    original_img = cv2.resize(img, (32, 32)).astype(np.float32)
    
    analysis_scatter(original_img)
    
    # 计算拟合CCM
    ccm_flat = []
    for params in ccm_fit_params:
        k1, k2, bias = params
        y = k1 * avg_rg + k2 * avg_bg + bias
        ccm_flat.append(y)
    ccm = np.reshape(ccm_flat, (3, 3))
    
    # 行归一化，确保每一行和为1
    for i in range(3):
        row_sum = np.sum(ccm[i])
        if row_sum != 0:
            ccm[i] /= row_sum
    # ccm = ccm * 0
    # ccm[0,0] = 1
    # ccm[1,1] = 1
    # ccm[2,2] = 1
    #return k_b, k_r, ccm
    return 1 / avg_bg, 1 / avg_rg, ccm

def analysis_scatter(original_img):
    global avg_rg, avg_bg
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
    white_point_count = 0
    for i in range(32):
        for j in range(32):
            # 过滤掉过曝像素点与欠曝像素点
            b, g, r = original_img[i, j]
            max_val = max(b, g, r)
            avg_val = (b + g + r) / 3
            if not (max_val < LUM_MAX_THROAT and avg_val > LUM_MIN_THROAT):
                continue

            #过滤掉white_point_regions外的像素点
            if g == 0:
                continue
            rg = r / g
            bg = b / g
            in_region = False
            for region in white_point_regions:
                x0, x1, y0, y1 = region
                if x0 <= rg <= x1 and y0 <= bg <= y1:
                    in_region = True
                    break
            if not in_region:
                continue

            # 更新awb比值
            avg_rg = avg_rg * update_ratio + rg * (1 - update_ratio)
            avg_bg = avg_bg * update_ratio + bg * (1 - update_ratio)
            white_point_count += 1

            px = int(rg * scale)
            py = IMG_SHAPE - 1 - int(bg * scale)
            if 0 <= px < IMG_SHAPE and 0 <= py < IMG_SHAPE:
                scatter_img[py, px] = 1#balanced_img[i, j] * 10

    # 绘制白点数量
    cv2.putText(scatter_img, str(white_point_count), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

    # 绘制(avg_rg, avg_bg)
    px = int(avg_rg * scale)
    py = IMG_SHAPE - 1 - int(avg_bg * scale)
    cv2.circle(scatter_img, (px, py), 5, (0, 0, 255), 2)

    # 绘制参考点和标签
    for (rg, bg), label in zip(reference_rg_bg, reference_labels):
        px = int(rg * scale)
        py = IMG_SHAPE - 1 - int(bg * scale)
        if 0 <= px < IMG_SHAPE and 0 <= py < IMG_SHAPE:
            cv2.circle(scatter_img, (px, py), 5, (1, 0, 0), 2)
            cv2.putText(scatter_img, label, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

    # 计算CCM并在右上角显示（保留两位小数）
    ccm_flat = []
    for params in ccm_fit_params:
        k1, k2, bias = params
        y = k1 * avg_rg + k2 * avg_bg + bias
        ccm_flat.append(y)
    ccm = np.reshape(ccm_flat, (3, 3))
    
    # 行归一化，确保每一行和为1
    for i in range(3):
        row_sum = np.sum(ccm[i])
        if row_sum != 0:
            ccm[i] /= row_sum
    
    # 显示CCM在右上角
    x_pos = IMG_SHAPE - 150
    y_pos = 20
    for i in range(3):
        row_str = f"{ccm[i,0]:.2f} {ccm[i,1]:.2f} {ccm[i,2]:.2f}"
        cv2.putText(scatter_img, row_str, (x_pos, y_pos + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 1, 1), 1)
    
    # Show the scatter
    cv2.imshow('AWB Scatter', scatter_img)