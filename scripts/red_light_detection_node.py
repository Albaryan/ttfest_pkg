#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2
import numpy as np



class RedLightDetectionNode(Node):
    def __init__(self):
        super().__init__("red_light_detection_node")
        self.subscriber_ = self.create_subscription(Image, "/camera/image_raw", self.callback_camera, 10)
        
        self.bridge = CvBridge()
                
        self.get_logger().info("Red light detection has started.")

    def callback_camera(self, msg):
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except CvBridgeError as e:
            print(e)

        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cimg = cv_image.copy()

        # Adaptif histogram eşitleme
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Kontrastı artırmak için clipLimit değerini artırdık
        img_yuv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # Parlaklığı azaltma
        alpha = 0.8  # Kontrastı ayarlamak için kullanılan çarpan
        beta = -60  # Parlaklığı azaltmak için kullanılan sabit, daha negatif bir değer parlaklığı daha fazla azaltır
        img_output = cv2.convertScaleAbs(img_output, alpha=alpha, beta=beta)

        hsv = cv2.cvtColor(img_output, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([140, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        maskr = cv2.add(mask1, mask2)

        size = img_output.shape

        # Konturları bulma
        contours, _ = cv2.findContours(maskr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Kontur alanını hesapla
            area = cv2.contourArea(cnt)
            if area > 150:  # Küçük konturları filtrele
                # Momentleri hesapla
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Konturları ve merkezi çiz
                    cv2.drawContours(cimg, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(cimg, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(cimg, 'RED LIGHT', (cx, cy), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
          
        cv2.imshow("img",cimg)
        
        key=cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RedLightDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown

if __name__=="__main__":
    main()
