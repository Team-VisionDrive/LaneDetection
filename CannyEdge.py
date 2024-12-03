#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
'''

'''

class CannyEdgeLaneDetector:
    def __init__(self):
        rospy.init_node("CannyEdgeLaneDetector_node")
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/canny_edge/compressed", CompressedImage, queue_size=10)
        rospy.Subscriber("/camera/rgb/image_raw/compressed", CompressedImage, self.img_CB)
        self.ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.cmd_vel_msg = Twist()

    def detect_edges(self, img):
        """
        이미지에서 Canny Edge Detection을 수행하는 함수
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def region_of_interest(self, edges):
        """
        관심 영역(ROI)을 설정하여 불필요한 부분을 제거
        """
        mask = np.zeros_like(edges)
        height, width = edges.shape
        # Define a triangular region of interest
        polygon = np.array([[
            (width * 0.1, height),
            (width * 0.9, height),
            (width * 0.5, height * 0.6)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges

    def detect_lane_lines(self, edges):
        """
        Hough Line Transform을 사용하여 차선을 탐지
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=150
        )
        return lines

    def draw_lines(self, img, lines):
        """
        탐지된 차선을 이미지에 그리는 함수.
        """
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    def img_CB(self, msg):
        """
        ROS 콜백 함수로 이미지를 처리.
        """
        # Decode the image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Edge detection pipeline
        edges = self.detect_edges(img)
        roi_edges = self.region_of_interest(edges)
        lines = self.detect_lane_lines(roi_edges)

        # Visualize the result
        img_with_lines = img.copy()
        self.draw_lines(img_with_lines, lines)

        # Publish the processed image
        compressed_msg = CompressedImage()
        compressed_msg.header.stamp = rospy.Time.now()
        compressed_msg.format = "jpeg"
        compressed_msg.data = cv2.imencode(".jpg", img_with_lines)[1].tobytes()
        self.pub.publish(compressed_msg)

        # Control logic (basic example, can be extended)
        if lines is not None:
            self.cmd_vel_msg.linear.x = 0.5
            self.cmd_vel_msg.angular.z = 0.0  # Basic straight driving logic
        else:
            self.cmd_vel_msg.linear.x = 0.0
            self.cmd_vel_msg.angular.z = 0.2  # Rotate if no lanes detected
        self.ctrl_pub.publish(self.cmd_vel_msg)

if __name__ == "__main__":
    try:
        detector = CannyEdgeLaneDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
