import numpy as np
import cv2

def abs_sobel_thresh(img, orient='x', thresh=(20, 100)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0.7, 1.3)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255
    return binary_output.astype(np.uint8)

def get_combined_gradients(img, thresh_x, thresh_y, thresh_mag, thresh_dir):
    R_channel = img[:, :, 2]
    sobelx = abs_sobel_thresh(R_channel, 'x', thresh_x)
    sobely = abs_sobel_thresh(R_channel, 'y', thresh_y)
    mag_binary = mag_thresh(R_channel, 3, thresh_mag)
    dir_binary = dir_thresh(R_channel, 15, thresh_dir)
    gradient_combined = np.zeros_like(dir_binary).astype(np.uint8)
    gradient_combined[((sobelx > 1) & (mag_binary > 1) & (dir_binary > 1)) | ((sobelx > 1) & (sobely > 1))] = 255
    return gradient_combined

def channel_thresh(channel, thresh=(80, 255)):
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 255
    return binary

def get_combined_hls(img, th_h, th_l, th_s):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    h_channel = channel_thresh(H, th_h)
    l_channel = channel_thresh(L, th_l)
    s_channel = channel_thresh(S, th_s)
    yellow_lower = np.array([15, 38, 115], dtype=np.uint8)
    yellow_upper = np.array([35, 204, 255], dtype=np.uint8)
    white_lower = np.array([0, 200, 0], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, white_lower, white_upper)
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    hls_comb = np.zeros_like(s_channel).astype(np.uint8)
    hls_comb[((s_channel > 1) & (l_channel == 0)) | 
             ((s_channel == 0) & (h_channel > 1) & (l_channel > 1)) | 
             (combined_mask > 1)] = 255
    return hls_comb

def combine_grad_hls(grad, hls):
    result = np.zeros_like(hls).astype(np.uint8)
    result[(grad > 1)] = 100
    result[(hls > 1)] = 255
    return result


