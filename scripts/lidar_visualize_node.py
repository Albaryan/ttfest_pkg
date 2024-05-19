#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import pygame
from math import cos, sin, pi, floor, tan
import numpy as np

from simple_pid import PID

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
        self.publisher_ = self.create_publisher(Twist, "/cmd_vel",10)
        
              
            
        self.pid = PID(-0.15, 0, 0, setpoint=0) 

        self.get_logger().info("Lidar node has started.")

    def callback_lidar(self, msg):
        lcd.fill((0,0,0))
        b = msg.angle_increment

        msgControl=Twist()

        lidarRanges = np.array(msg.ranges)
        lidarRadian = np.arange(0,len(msg.ranges)).astype('float')
        lidarDegree = (2*pi -lidarRadian)*b*180/pi

        pygame.event.get()
        for degree in lidarDegree:

            diff=abs(-90-degree)
            range=lidarRanges[np.where(lidarDegree==degree)]
            
            msgControl.linear.x=0.2

            radians = degree * pi / 180.0
            if diff>50 and diff < 75: 

                if range<=0.4:

                    if 90+degree <0:
                        msgControl.angular.z=float(+range*5)
                    else:
                        msgControl.angular.z=float(-range*5)

            elif diff < 50: 
                #print(degree)

                if range<=0.5:

                    if 90+degree <0:
                        msgControl.angular.z=float(+range*5)
                    else:
                        msgControl.angular.z=float(-range*5)

            elif diff<105 and diff>75:
                if range<=0.2:

                    if 90+degree <0:
                        msgControl.angular.z=float(+range*5)
                    else:
                        msgControl.angular.z=float(-range*5)

            if range<=6.0 and diff<105:
                max_distance = max([min([1000, range]), 2.0])
                x = range * cos(radians)
                y = range * sin(radians)
                point = ( int(W / 2) + int(x / max_distance * (W/2)), int(H/2) + int(y / max_distance * (H/2) ))
                pygame.draw.circle(lcd,pygame.Color(250, 250, 0),point,2 )
                
                self.publisher_.publish(msgControl)                
        pygame.display.update()
    
def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizeNode()
    rclpy.spin(node)
    rclpy.shutdown

if __name__=="__main__":
    main()
