#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point
from std_msgs.msg import Int32
import cv2
import numpy as np
'''
요약      : ROI
흐름      : Sub → ROI → 화면에 출력
'''

class ROI:
    def __init__(self):
        rospy.init_node("lane_detect")
        self.cvbridge = CvBridge()
        rospy.Subscriber(
            "/camera/rgb/image_raw/compressed/compressed",
            CompressedImage, 
            self.image_topic_callback
        )
        self.distance_pub = rospy.Publisher("/limo/lane_x", Point, queue_size=5)
        self.viz = rospy.get_param("~visualization", True)

    def applyBirdEyeView(self, img): # Bird-eye view 변환하는 함수
        '''
            Bird-eye view 변환
            관심 영역: 이미지 하단 [360:480, 0:640]
        '''
        src = np.float32(      # 원본 이미지의 관심 영역 설정
            [
                [0, 479],      # 좌측 하단
                [0, 360],      # 좌측 상단
                [639, 360],    # 우측 상단
                [639, 479],    # 우측 하단
            ]
        )
        dst = np.float32(      # 변환 후 출력 이미지의 매핑 좌표 설정
            [
                [0, 479],      # 좌측 하단
                [0, 0],        # 좌측 상단
                [639, 0],      # 우측 상단
                [639, 479],    # 우측 하단
            ]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)                              # 원근 변환 행렬 계산
        warped_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0])) # Bird-eye view 변환 적용
        return warped_img # (640 x 480)
    
    def visResult(self):
        '''
            최종 결과가 추가된 원본 이미지 (lane_original)
            차선 영역만 ROI로 잘라낸 이미지 (lane_cropped)
            ROI 내부 중 특정 색 영역만 검출한 이미지 (lane_threshold)
        '''
        print("-ing")
        cv2.imshow("lane_original", self.frame)
        cv2.imshow("lane_cropped", self.bird_image)
        
        cv2.waitKey(1)

    def image_topic_callback(self, img):
        '''
            실제 이미지를 입력 받아서 동작하는 부분
            1. CompressedImage --> OpenCV Type Image 변경 (compressed_imgmsg_to_cv2)
            2. ROI 지정 & Bird-eye View 변환 (applyBirdEyeView)
        '''
        self.frame = self.cvbridge.compressed_imgmsg_to_cv2(img, "bgr8") #1
        self.bird_image = self.applyBirdEyeView(self.frame) #2

        # visualization
        if self.viz:
            self.visResult()

def run():
    new_class = ROI()
    rospy.spin()

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("program down")
