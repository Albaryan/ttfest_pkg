#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ttfest_pkg.camera_calibration import calib, undistort
from ttfest_pkg.threshold import get_combined_gradients, get_combined_hls, combine_grad_hls
from ttfest_pkg.line import Line, get_perspective_transform, get_lane_lines_img, line_search_tracking, line_search_reset
from ttfest_pkg.pid_controller import PIDController

from geometry_msgs.msg import Twist

class LaneFollowingNode(Node):
    def __init__(self):
        super().__init__("lane_following_node")
        self.subscriber_ = self.create_subscription(Image, "/camera/image_raw", self.callback_camera, 10)

        self.publisher_ = self.create_publisher(Twist, "/cmd_vel",10)
        msg=Twist()

        self.bridge = CvBridge()
        self.left_line = Line()
        self.right_line = Line()

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

        self.ptime = 0
        self.ctime = 0
        self.Kp = 0.015
        self.Ki = 0.0
        self.Kd = -0.0005
        self.center_car = 414/2
        self.setpoint = self.center_car
        self.pid_controller = PIDController(self.Kp, self.Ki, self.Kd, self.setpoint)
        self.get_logger().info("Camera has started.")

        # Create trackbars for threshold adjustments
        cv2.namedWindow('Threshold Adjustments')
        cv2.createTrackbar('SobelX Min', 'Threshold Adjustments', self.th_sobelx_min, 255, self.nothing)
        cv2.createTrackbar('SobelX Max', 'Threshold Adjustments', self.th_sobelx_max, 255, self.nothing)
        cv2.createTrackbar('SobelY Min', 'Threshold Adjustments', self.th_sobely_min, 255, self.nothing)
        cv2.createTrackbar('SobelY Max', 'Threshold Adjustments', self.th_sobely_max, 255, self.nothing)
        cv2.createTrackbar('Mag Min', 'Threshold Adjustments', self.th_mag_min, 255, self.nothing)
        cv2.createTrackbar('Mag Max', 'Threshold Adjustments', self.th_mag_max, 255, self.nothing)
        cv2.createTrackbar('Dir Min', 'Threshold Adjustments', int(self.th_dir_min * 100), 300, self.nothing)
        cv2.createTrackbar('Dir Max', 'Threshold Adjustments', int(self.th_dir_max * 100), 300, self.nothing)
        cv2.createTrackbar('H Min', 'Threshold Adjustments', self.th_h_min, 255, self.nothing)
        cv2.createTrackbar('H Max', 'Threshold Adjustments', self.th_h_max, 255, self.nothing)
        cv2.createTrackbar('L Min', 'Threshold Adjustments', self.th_l_min, 255, self.nothing)
        cv2.createTrackbar('L Max', 'Threshold Adjustments', self.th_l_max, 255, self.nothing)
        cv2.createTrackbar('S Min', 'Threshold Adjustments', self.th_s_min, 255, self.nothing)
        cv2.createTrackbar('S Max', 'Threshold Adjustments', self.th_s_max, 255, self.nothing)

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

    def callback_camera(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: %s" % e)
            return
        
        controlMsg=Twist()

        cv_image = cv2.resize(cv_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        rows, cols = cv_image.shape[:2]

        #print(rows,cols)

        self.update_thresholds()

        combined_gradient = get_combined_gradients(cv_image, 
                                                   (self.th_sobelx_min, self.th_sobelx_max),
                                                   (self.th_sobely_min, self.th_sobely_max),
        (self.th_mag_min, self.th_mag_max),
        (self.th_dir_min, self.th_dir_max))
        #cv2.imshow("Combined Gradient", combined_gradient)
        combined_hls = get_combined_hls(cv_image, 
                                    (self.th_h_min, self.th_h_max), 
                                    (self.th_l_min, self.th_l_max), 
                                    (self.th_s_min, self.th_s_max))
        #cv2.imshow("Combined HLS", combined_hls)

        combined_result = combine_grad_hls(combined_gradient, combined_hls)
        #cv2.imshow("Combined Result", combined_result)

        c_rows, c_cols = combined_result.shape[:2]
    
    # Belirtilen src noktaları
        src = np.float32([
            [0, 260 ],
            [110, 200],
            [285, 200 ],
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

        searching_img = get_lane_lines_img(warp_img, self.left_line, self.right_line)

        lane_color = np.zeros_like(cv_image)

        result = cv2.addWeighted(cv_image, 1, lane_color, 0.3, 0)

        current_position = (self.left_line.startx + self.right_line.startx) / 2
    # Draw src points on the original image
        for point in src:
            cv2.circle(cv_image, tuple(point.astype(int)), 5, (0, 255, 0), -1)

        # Draw dst points on the warped image for verification
        for point in dst:
            cv2.circle(warp_img, tuple(point.astype(int)), 5, (0, 0, 255), -1)

        cv2.imshow("warp_image", warp_img)

        #cv2.imshow("searching_img", searching_img)

        error = self.setpoint - current_position

        steering_angle = self.pid_controller.update(error)
        print(error, self.setpoint, steering_angle)

        controlMsg.angular.z=-steering_angle
        controlMsg.linear.x=0.1

        self.publisher_.publish(controlMsg)

        cv2.line(result,(int(self.setpoint),3*result.shape[0]//4),(int(self.setpoint-error),3*result.shape[0]//4),(0,255,0),5)
        cv2.line(result,(int(self.setpoint),3*result.shape[0]//4),(int(self.setpoint-steering_angle),3*result.shape[0]//4),(0,0,255),5)
        

        cv2.imshow("result" ,result)



            # print(undist_img.shape)



    # Check if lines are detected and not None
        #if self.left_line.startx is not None and self.right_line.startx is not None:
        #    current_position = (self.left_line.startx + self.right_line.startx) / 2
        #    error = self.setpoint - current_position
        #    steering_angle = self.pid_controller.update(error)
        #    self.get_logger().info("Error: %s, Steering Angle: %s" % (error, steering_angle))
        #else:
        #    self.get_logger().warn("Lines not detected, cannot compute current_position and steering_angle.")
    
        cv2.imshow("img", cv_image)
        key = cv2.waitKey(1)
def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()