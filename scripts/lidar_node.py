#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan

class LidarNode(Node):
    def __init__(self):
        super().__init__("lidar_node")
        self.subscriber_ = self.create_subscription(LaserScan, "/lidar/data", self.callback_lidar, 10)
                
        self.get_logger().info("Lidar node has started.")

    def callback_lidar(self, msg):
        
        print(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    rclpy.spin(node)
    rclpy.shutdown

if __name__=="__main__":
    main()
