import numpy as np
import cv2
import matplotlib.image as mpimg
from PIL import Image


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Set the width of the windows +/- margin
        self.window_margin = 56
        # x values of the fitted line over the last n iterations
        self.prevx = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_info = None
        self.curvature = None
        self.deviation = None


def get_perspective_transform(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    return warp_img, M, Minv


def measure_curvature(left_lane, right_lane):
    ploty = left_lane.ally
    leftx, rightx = left_lane.allx, right_lane.allx
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    y_eval = np.max(ploty)

    lane_width = abs(right_lane.startx - left_lane.startx)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 * (720 / 1280) / lane_width  # meters per pixel in x dimension

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    left_lane.radius_of_curvature = left_curverad
    right_lane.radius_of_curvature = right_curverad


def smoothing(lines, prev_n_lines=3):
    lines = np.squeeze(lines)
    avg_line = np.zeros((720))
    for i, line in enumerate(reversed(lines)):
        if i == prev_n_lines:
            break
        avg_line += line
    avg_line = avg_line / prev_n_lines
    return avg_line


def detect_left_lane(binary_img, left_lane):
    histogram = np.sum(binary_img[int(binary_img.shape[0] / 2):, :], axis=0)
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    midpoint = int(histogram.shape[0] / 2)
    leftX_base = np.argmax(histogram[:midpoint])

    num_windows = 9
    window_height = int(binary_img.shape[0] / num_windows)
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    current_leftX = leftX_base
    min_num_pixel = 50
    win_left_lane = []
    window_margin = left_lane.window_margin

    for window in range(num_windows):
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        win_leftx_min = current_leftX - window_margin
        win_leftx_max = current_leftX + window_margin

        cv2.rectangle(out_img, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)

        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
            nonzerox <= win_leftx_max)).nonzero()[0]
        win_left_lane.append(left_window_inds)

        if len(left_window_inds) > min_num_pixel:
            current_leftX = int(np.mean(nonzerox[left_window_inds]))

    win_left_lane = np.concatenate(win_left_lane)
    leftx = nonzerox[win_left_lane]
    lefty = nonzeroy[win_left_lane]
    out_img[lefty, leftx] = [255, 0, 0]

    left_fit = np.polyfit(lefty, leftx, 2)
    left_lane.current_fit = left_fit

    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

    left_lane.prevx.append(left_plotx)
    if len(left_lane.prevx) > 10:
        left_avg_line = smoothing(left_lane.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_lane.current_fit = left_avg_fit
        left_lane.allx, left_lane.ally = left_fit_plotx, ploty
    else:
        left_lane.current_fit = left_fit
        left_lane.allx, left_lane.ally = left_plotx, ploty

    left_lane.startx, left_lane.endx = left_lane.allx[len(left_lane.allx)-1], left_lane.allx[0]
    left_lane.detected = True

    return out_img

def detect_right_lane(binary_img, right_lane):
    histogram = np.sum(binary_img[int(binary_img.shape[0] / 2):, :], axis=0)
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    midpoint = int(histogram.shape[0] / 2)
    rightX_base = np.argmax(histogram[midpoint:]) + midpoint

    num_windows = 9
    window_height = int(binary_img.shape[0] / num_windows)
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    current_rightX = rightX_base
    min_num_pixel = 50
    win_right_lane = []
    window_margin = right_lane.window_margin

    for window in range(num_windows):
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        win_rightx_min = current_rightX - window_margin
        win_rightx_max = current_rightX + window_margin

        cv2.rectangle(out_img, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
            nonzerox <= win_rightx_max)).nonzero()[0]
        win_right_lane.append(right_window_inds)

        if len(right_window_inds) > min_num_pixel:
            current_rightX = int(np.mean(nonzerox[right_window_inds]))

    win_right_lane = np.concatenate(win_right_lane)
    rightx = nonzerox[win_right_lane]
    righty = nonzeroy[win_right_lane]
    out_img[righty, rightx] = [0, 0, 255]

    right_fit = np.polyfit(righty, rightx, 2)
    right_lane.current_fit = right_fit

    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    right_lane.prevx.append(right_plotx)
    if len(right_lane.prevx) > 10:
        right_avg_line = smoothing(right_lane.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_lane.current_fit = right_avg_fit
        right_lane.allx, right_lane.ally = right_fit_plotx, ploty
    else:
        right_lane.current_fit = right_fit
        right_lane.allx, right_lane.ally = right_plotx, ploty

    right_lane.startx, right_lane.endx = right_lane.allx[len(right_lane.allx)-1], right_lane.allx[0]
    right_lane.detected = True

    return out_img
def line_search_tracking_left(binary_img, left_line):
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = left_line.window_margin

    left_lane_inds = ((nonzerox > (left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy +
                    left_line.current_fit[2] - margin)) & (nonzerox < (left_line.current_fit[0] * (nonzeroy ** 2) +
                    left_line.current_fit[1] * nonzeroy + left_line.current_fit[2] + margin))).nonzero()[0]

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = leftx, lefty
        left_line.startx, left_line.endx = left_fit[0] * binary_img.shape[0] ** 2 + left_fit[1] * binary_img.shape[0] + left_fit[2], left_fit[0] * 0 ** 2 + left_fit[1] * 0 + left_fit[2]

    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]

    return out_img

def line_search_tracking_right(binary_img, right_line):
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = right_line.window_margin

    right_lane_inds = ((nonzerox > (right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy +
                     right_line.current_fit[2] - margin)) & (nonzerox < (right_line.current_fit[0] * (nonzeroy ** 2) +
                     right_line.current_fit[1] * nonzeroy + right_line.current_fit[2] + margin))).nonzero()[0]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = rightx, righty
        right_line.startx, right_line.endx = right_fit[0] * binary_img.shape[0] ** 2 + right_fit[1] * binary_img.shape[0] + right_fit[2], right_fit[0] * 0 ** 2 + right_fit[1] * 0 + right_fit[2]

    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img

def line_search_tracking_both(binary_img, left_lane, right_lane):
    # Sol ve sağ şeritlerin geçmiş pozisyonlarından hareketle, her iki şeride ait piksellerin x ve y konumlarını belirleyin
    left_nonzero = binary_img[left_lane.ally, left_lane.allx]
    right_nonzero = binary_img[right_lane.ally, right_lane.allx]

    # Sol ve sağ şeritler için polinomlar uyumlayın
    left_fit = np.polyfit(left_lane.ally, left_lane.allx, 2)
    right_fit = np.polyfit(right_lane.ally, right_lane.allx, 2)

    # Polinomlar aracılığıyla her iki şeridin tahmini konumlarını hesaplayın
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Tahmini konumları görselleştirin ve çıktı görüntüsünü oluşturun
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255
    out_img[left_lane.ally, left_lane.allx] = [255, 0, 0]
    out_img[right_lane.ally, right_lane.allx] = [0, 0, 255]
    out_img = cv2.polylines(out_img, np.int32([np.column_stack((left_fitx, ploty))]), isClosed=False, color=(255, 255, 0), thickness=2)
    out_img = cv2.polylines(out_img, np.int32([np.column_stack((right_fitx, ploty))]), isClosed=False, color=(255, 255, 0), thickness=2)

    # Her iki şeridin de detaylarını güncelleyin
    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit
    left_lane.allx, left_lane.ally = left_lane.allx, left_lane.ally
    right_lane.allx, right_lane.ally = right_lane.allx, right_lane.ally

    # Ölçülen eğriliği hesaplayın
    measure_curvature(left_lane, right_lane)

    return out_img


def process_image(binary_img, left_lane, right_lane):
    if left_lane.detected:
        out_img = line_search_tracking_left(binary_img, left_lane)
    else:
        out_img = detect_left_lane(binary_img, left_lane)
        if not left_lane.detected:
            if right_lane.detected:
                out_img = line_search_tracking_right(binary_img, right_lane)
            else:
                out_img = detect_right_lane(binary_img, right_lane)
                if right_lane.detected:
                    # Sağ şerit üzerinden sol şeridi tahmin etme
                    left_lane.current_fit = right_lane.current_fit
                    out_img = line_search_tracking_left(binary_img, left_lane)

    return out_img
