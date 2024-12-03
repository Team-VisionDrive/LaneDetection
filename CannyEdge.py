#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32
#from dynamic_reconfigure.server import Server
#from limo_application.cfg import image_processingConfig
import cv2
import numpy as np
'''
요약      : Lane Detection
흐름      : Sub → ROI → Bird-eye View → CannyEdge(윤곽선 검출) → HoughLineTransform(직선 검출) → Pub(x, y)
'''

class LaneDetection:
    def __init__(self):
        rospy.init_node("lane_detect")
        self.cvbridge = CvBridge()
        rospy.Subscriber(
            rospy.get_param("~image_topic_name", "/camera/rgb/image_raw/compressed"), 
            CompressedImage, 
            self.image_topic_callback
        )
        self.distance_pub = rospy.Publisher("/limo/lane_x", Int32, queue_size=5)
        self.viz = rospy.get_param("~visualization", True)

    def applyROI(self, _img=np.ndarray(shape=(480, 640))):
        '''
            관심 영역 검출
        '''
        return _img[360:480, 320:640]

    def image_topic_callback(self, img):
        '''
            실제 이미지를 입력 받아서 동작하는 부분
            1. CompressedImage --> OpenCV Type Image 변경 (compressed_imgmsg_to_cv2)
            2. 차선 영역만 ROI 지정 (applyROI)
            3. Bird-eye View 변환 (applyBirdEyeView)
            4. ROI 영역에서 윤곽선 검출 (applyCanny)
            5. 직선 검출 (applyHoughLine)
            6. 검출된 차선을 기반으로 거리 계산 (calcLaneDistance)
            7. 최종 검출된 값을 기반으로 카메라 좌표계 기준 차선 무게중심 점의 x, y 좌표 Publish
        '''
        self.frame = self.cvbridge.compressed_imgmsg_to_cv2(img, "bgr8") #1
        self.roi_image = self.applyROI(self.frame) #2
        self.wrap_image = self.applyBirdEyeView(self.roi_image) #3
        self.edge_image = self.applyCanny(self.wrap_image) #4
        self.line = self.applyHoughLine(self.edge_image) #5
        self.distance = self.calcLaneDistance(self.line) #6
        self.distance_pub.publish(self.distance) #7

        # visualization
        if self.viz:
            self.visResult()

def run():
    new_class = LaneDetection()
    rospy.spin()

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("program down")