import cv2
import numpy as np

def awb_analysis(img):
    b_mean = img[:,:,0].mean()
    g_mean = img[:,:,1].mean()
    r_mean = img[:,:,2].mean()
    k_b = g_mean / b_mean
    k_r = g_mean / r_mean
    return k_b, k_r