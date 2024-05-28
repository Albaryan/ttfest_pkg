#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ttfest_pkg.camera_calibration import calib, undistort
from ttfest_pkg.threshold import get_combined_gradients, get_combined_hls, combine_grad_hls
from ttfest_pkg.line import Line, get_perspective_transform, line_search_reset, process_image
from ttfest_pkg.pid_controller import PIDController
from geometry_msgs.msg import Twist


class LaneFollowingNode(Node):
    def __init__(self):
        super().__init__("lane_following_node")
        self.subscriber_ = self.create_subscription(Image, "/camera/image_raw", self.callback_camera, 10)
        self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)
        
        self.bridge = CvBridge()
        self.left_line = Line()
        self.right_line = Line()

        # Threshold initial values
        self.th_sobelx_min = 35
        self.th_sobelx_max = 100
        self.th_sobely_min = 30
        self.th_sobely_max = 255
        self.th_mag_min = 30
        self.th_mag_max = 255
        self.th_dir_min = 0.7
        self.th_dir_max = 1.3
        self.th_h_min = 10
        self.th_h_max = 100
        self.th_l_min = 0
        self.th_l_max = 60
        self.th_s_min = 85
        self.th_s_max = 255

        # PID Controller parameters
        self.Kp = 0.020
        self.Ki = 0.0
        self.Kd = 0.05
        self.center_car = 414 / 2
        self.setpoint = self.center_car
        self.pid_controller = PIDController(self.Kp, self.Ki, self.Kd, self.setpoint)

        self.get_logger().info("Camera has started.")

        # Create trackbars for threshold adjustments
        cv2.namedWindow('Threshold Adjustments')
        self.create_trackbar('SobelX Min', self.th_sobelx_min)
        self.create_trackbar('SobelX Max', self.th_sobelx_max)
        self.create_trackbar('SobelY Min', self.th_sobely_min)
        self.create_trackbar('SobelY Max', self.th_sobely_max)
        self.create_trackbar('Mag Min', self.th_mag_min)
        self.create_trackbar('Mag Max', self.th_mag_max)
        self.create_trackbar('Dir Min', int(self.th_dir_min * 100))
        self.create_trackbar('Dir Max', int(self.th_dir_max * 100))
        self.create_trackbar('H Min', self.th_h_min)
        self.create_trackbar('H Max', self.th_h_max)
        self.create_trackbar('L Min', self.th_l_min)
        self.create_trackbar('L Max', self.th_l_max)
        self.create_trackbar('S Min', self.th_s_min)
        self.create_trackbar('S Max', self.th_s_max)

    def create_trackbar(self, name, value):
        cv2.createTrackbar(name, 'Threshold Adjustments', value, 255, self.nothing)

    def nothing(self, x):
        pass

    def update_thresholds(self):
        self.th_sobelx_min = cv2.getTrackbarPos('SobelX Min', 'Threshold Adjustments')
        self.th_sobelx_max = cv2.getTrackbarPos('SobelX Max', 'Threshold Adjustments')
        self.th_sobely_min = cv2.getTrackbarPos('SobelY Min', 'Threshold Adjustments')
        self.th_sobely_max = cv2.getTrackbarPos('SobelY Max', 'Threshold Adjustments')
        self.th_mag_min = cv2.getTrackbarPos('Mag Min', 'Threshold Adjustments')
        self.th_mag_max = cv2.getTrackbarPos('Mag Max', 'Threshold Adjustments')
        self.th_dir_min = cv2.getTrackbarPos('Dir Min', 'Threshold Adjustments') / 100.0
        self.th_dir_max = cv2.getTrackbarPos('Dir Max', 'Threshold Adjustments') / 100.0
        self.th_h_min = cv2.getTrackbarPos('H Min', 'Threshold Adjustments')
        self.th_h_max = cv2.getTrackbarPos('H Max', 'Threshold Adjustments')
        self.th_l_min = cv2.getTrackbarPos('L Min', 'Threshold Adjustments')
        self.th_l_max = cv2.getTrackbarPos('L Max', 'Threshold Adjustments')
        self.th_s_min = cv2.getTrackbarPos('S Min', 'Threshold Adjustments')
        self.th_s_max = cv2.getTrackbarPos('S Max', 'Threshold Adjustments')

    def process_image(self, image, left_line, right_line):
        lane_img = process_image(image, left_line, right_line)
        return lane_img

    def callback_camera(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: %s" % e)
            return

        controlMsg = Twist()

        # Resize image for processing
        cv_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Update thresholds from trackbars
        self.update_thresholds()

        # Apply combined gradient and HLS thresholds
        combined_gradient = get_combined_gradients(cv_image, 
                                                   (self.th_sobelx_min, self.th_sobelx_max),
                                                   (self.th_sobely_min, self.th_sobely_max),
                                                   (self.th_mag_min, self.th_mag_max),
                                                   (self.th_dir_min, self.th_dir_max))
        cv2.imshow("Combined Gradient", combined_gradient)

        combined_hls = get_combined_hls(cv_image, 
                                        (self.th_h_min, self.th_h_max), 
                                        (self.th_l_min, self.th_l_max), 
                                        (self.th_s_min, self.th_s_max))
        cv2.imshow("Combined HLS", combined_hls)

        combined_result = combine_grad_hls(combined_gradient, combined_hls)
        cv2.imshow("Combined Result", combined_result)

        # Perspective transform points
        src = np.float32([
            [0, 260],
            [110, 200],
            [285, 200],
            [414, 260]
        ])
        dst = np.float32([
            [0, 720],
            [0, 0],
            [410, 0],
            [414, 720]
        ])

        # Perspective transform
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warp_img = cv2.warpPerspective(combined_result, M, (414, 720))
        cv2.imshow("Warped Image", warp_img)

        # Process image to find and track lane lines
        
        searching_img = process_image(warp_img, self.left_line, self.right_line)
        cv2.imshow("b", searching_img)
        # Add lane lines to the original image
        lane_color = np.zeros_like(cv_image)
        result = cv2.addWeighted(cv_image, 1, lane_color, 0.3, 0)
        # Calculate current position and steering error
        if self.left_line.detected and self.right_line.detected:
            current_position = (self.left_line.startx + self.right_line.startx) / 2
        elif self.left_line.detected and not self.right_line.detected:
            current_position = self.left_line.startx + 100
        elif self.right_line.detected and not self.left_line.detected:
            current_position = self.right_line.startx - 100
        else:
            current_position = self.setpoint  




        error = self.setpoint - current_position
        # Update PID controller and publish control message
        steering_angle = self.pid_controller.update(error)
        controlMsg.angular.z = -steering_angle
        controlMsg.linear.x = 0.4
        self.publisher_.publish(controlMsg)
        # Draw visual aids
        cv2.line(result, (int(self.setpoint), 3 * result.shape[0] // 4), (int(current_position), 3 * result.shape[0] // 4), (0, 255, 0), 5)
        cv2.line(result, (int(self.setpoint), 3 * result.shape[0] // 4), (int(self.setpoint - steering_angle), 3 * result.shape[0] // 4), (0, 0, 255), 5)

        cv2.imshow("Result", result)

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            rclpy.shutdown()




def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
