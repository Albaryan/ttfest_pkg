#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2

import numpy as np
from ttfest_pkg.camera_calibration import calib, undistort
from ttfest_pkg.threshold import get_combined_gradients, get_combined_hls, combine_grad_hls
from ttfest_pkg.line import Line, get_perspective_transform, get_lane_lines_img

import time
from ttfest_pkg.pid_controller import PIDController



class LaneFollowingNode(Node):
    def __init__(self):
        super().__init__("lane_following_node")
        self.subscriber_ = self.create_subscription(Image, "/camera/image_raw", self.callback_camera, 10)
        
        self.bridge = CvBridge()
        
        self.left_line = Line()
        self.right_line = Line()
        
        self.th_sobelx, self.th_sobely, self.th_mag, self.th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
        self.th_h, self.th_l, self.th_s = (10, 100), (0, 60), (85, 255)
                
        self.ptime = 0
        self.ctime = 0
        
        self.Kp = 0.3    
        self.Ki = 0.2
        self.Kd = 0.1
        
        self.center_car = 320
        self.setpoint = self.center_car
        
        self.pid_controller = PIDController(self.Kp, self.Ki, self.Kd, self.setpoint)
                
        self.get_logger().info("Camera has started.")

    def callback_camera(self, msg):
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except CvBridgeError as e:
            print(e)
            
        
        cv_image = cv2.resize(cv_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        rows, cols = cv_image.shape[:2]
        
        combined_gradient = get_combined_gradients(cv_image, self.th_sobelx, self.th_sobely, self.th_mag, self.th_dir)
        
        combined_hls = get_combined_hls(cv_image, self.th_h, self.th_l, self.th_s)
        
        combined_result = combine_grad_hls(combined_gradient, combined_hls)
        
        c_rows, c_cols = combined_result.shape[:2]
        s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
        s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]
        
        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])
        
        warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (720, 720))

        searching_img = get_lane_lines_img(warp_img, self.left_line, self.right_line)
        
        lane_color = np.zeros_like(cv_image)
        
        result = cv2.addWeighted(cv_image, 1, lane_color, 0.3, 0)
        
        current_position = (self.left_line.startx + self.right_line.startx) / 2
        
        error = self.setpoint - current_position
        
        steering_angle = self.pid_controller.update(error)
        print(error,steering_angle)
        
        cv2.imshow("img",cv_image)
        
        key=cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowingNode()
    rclpy.spin(node)
    rclpy.shutdown

if __name__=="__main__":
    main()
