#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2




class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.subscriber_ = self.create_subscription(Image, "/camera/image_raw", self.callback_camera, 10)
        
        self.bridge = CvBridge()
                
        self.get_logger().info("Camera has started.")

    def callback_camera(self, msg):
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except CvBridgeError as e:
            print(e)
            
        cv2.imshow("img",cv_image)
        
        key=cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown

if __name__=="__main__":
    main()