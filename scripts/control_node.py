#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")
        
        self.vel=Twist()
        
        self.publisher_= self.create_publisher(Twist, "/cmd_vel",10)
        
        self.timer_ = self.create_timer(1, self.publish_car)
        
        self.get_logger().info("Robot speed control started")
        
    def publish_car(self):
        self.vel.linear.x=0.5
        self.vel.angular.z=0.0
    
        self.publisher_.publish(self.vel) 
            


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    rclpy.shutdown

if __name__=="__main__":
    main()
