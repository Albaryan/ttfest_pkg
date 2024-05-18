#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan

import pygame
from math import cos, sin, pi, floor

W = 640
H = 480

pygame.display.init()
lcd = pygame.display.set_mode((W,H))
pygame.mouse.set_visible(True)
lcd.fill((0,0,0))
pygame.display.update()

class LidarVisualizeNode(Node):
    def __init__(self):
        super().__init__("lidar_visualize_node")
        self.subscriber_ = self.create_subscription(LaserScan, "/lidar/data", self.callback_lidar, 10)
                
        self.get_logger().info("Lidar node has started.")

    def callback_lidar(self, msg):
        lcd.fill((0,0,0))
        b = msg.angle_increment

        pygame.event.get()
        
        for ind,measurement in enumerate(msg.ranges):
            d = b*ind*180/pi
            distance = measurement
            if measurement<6.0:

                max_distance = max([min([1000, distance]), msg.range_max])
                radians = d * pi / 180.0
                x = distance * cos(radians)
                y = distance * sin(radians)
                point = ( int(W / 2) + int(x / max_distance * (W/2)), int(H/2) + int(y / max_distance * (H/2) ))
                pygame.draw.circle(lcd,pygame.Color(250, 250, 0),point,2 )
        pygame.display.update()

def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizeNode()
    rclpy.spin(node)
    rclpy.shutdown

if __name__=="__main__":
    main()
