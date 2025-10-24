import cv2
import numpy as np

MIN_THRESHOLD = 10 / 255.0
MAX_THRESHOLD = 200 / 255.0

cv2.namedWindow('AWB Scatter')
cv2.setWindowProperty('AWB Scatter', cv2.WND_PROP_TOPMOST, 1)

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
    scatter_img = np.zeros((IMG_SHAPE, IMG_SHAPE, 3), np.float32)
    
    # Scale: IMG_SHAPE pixels for 1.5 units -> IMG_SHAPE pixels per unit
    scale = IMG_SHAPE / 1.5
    
    # Loop over each pixel
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
    
    # Show the scatter
    cv2.imshow('AWB Scatter', scatter_img)