#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point
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
        self.distance_pub = rospy.Publisher("/limo/lane_x", Point, queue_size=5)
        self.viz = rospy.get_param("~visualization", True)

    def applyROI(self, _img=np.ndarray(shape=(480, 640))):
        '''
            관심 영역 검출
        '''
        return _img[360:480, 0:640]
    
    def applyBirdEyeView(self, _img=np.ndarray(shape=(120, 640))):
        '''
            잘라낸 이미지에서 Bird-eye view 변환
        '''
        height, width = _img.shape[:2]       # (640, 480)
        src_points = np.float32([            # 변환 전 사각형 영역
            [0, height],                     # 왼쪽 아래
            [width, height],                 # 오른쪽 아래
            [width // 2 - 50, height // 2],  # 왼쪽 중간
            [width // 2 + 50, height // 2]   # 오른쪽 중간
        ])
        dst_points = np.float32([            # 변환 후 사각형 영역
            [0, height],                     # 왼쪽 아래
            [width, height],                 # 오른쪽 아래
            [width, 0],                      # 오른쪽 위
            [0, 0]                           # 왼쪽 위
        ])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)    # 원근 변환 행렬 계산
        wrap_image = cv2.warpPerspective(_img, matrix, (width, height)) # 원근 변환 적용
        return wrap_image # np.ndarray 형식의 이미지

    def applyCanny(self, _img):
        '''
            윤곽선 검출
        '''
        gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge_image = cv2.Canny(blur, 50, 150)
        return edge_image
    
    def applyHoughLine(self, edges):
        '''
            Canny edge 검출 후의 이미지에서 직선 검출
        '''
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20) #numpy.ndarray (검출된 직선의 시작점과 끝점을 포함하는 값들)
        return lines
    
    def drawHoughLines(self, img, lines):
        '''
            원본 이미지에 검출된 직선을 표시
            - img: 원본 이미지 (numpy.ndarray)
            - lines: HoughLinesP에서 반환된 직선 데이터 (numpy.ndarray)
        '''
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    # 각 직선을 원본 이미지에 표시 (빨간색, 두께 2)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            print("No lines detected to draw.")
        return img

    def calcLaneDistance(self, lines):
        '''
            직선들을 기반으로 차선의 모멘트 계산
            여러 차선에 대한 무게중심 (x, y)를 계산
        '''
        lane_centers = []
        try:
            for line in lines:
                for x1, y1, x2, y2 in line:   # 각 직선의 중간점을 구함
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    lane_centers.append((center_x, center_y))
            self.publishLaneCenter(lane_centers)
        except Exception as e:
            #rospy.logerr(f"Error in calcLaneDistance: {str(e)}")
            return []

    def publishLaneCenter(self, lane_centers):
        '''
            차선 무게중심을 ROS 메시지로 Publish (차선이 1개일 수도, 2개일 수도 있다.)
        '''
        try:
            if not lane_centers:
                rospy.logwarn("No lane centers to publish")
                return
            for center in lane_centers:          # 각 차선의 무게중심 좌표를 Point 메시지로 변환하여 publish
                point = Point()
                point.x = center[0]
                point.y = center[1]
                #print("차선의 좌표: ", point.x, point.y)
                self.distance_pub.publish(point) # 각 차선의 좌표를 하나씩 publish
        except Exception as e:
            #rospy.logerr(f"Error in publishLaneCenter: {str(e)}")
            return 
    
    def visResult(self):
        '''
            최종 결과가 추가된 원본 이미지 (lane_original)
            차선 영역만 ROI로 잘라낸 이미지 (lane_cropped)
            ROI 내부 중 특정 색 영역만 검출한 이미지 (lane_threshold)
        '''
        #cv2.circle(self.cropped_image, (self.x, self.y), 10, 255, -1)
        cv2.imshow("lane_original", self.frame)
        cv2.imshow("lane_cropped", self.roi_image)
        cv2.imshow("lane_thresholded", self.edge_image)

        cv2.waitKey(1)

    def image_topic_callback(self, img):
        '''
            실제 이미지를 입력 받아서 동작하는 부분
            1. CompressedImage --> OpenCV Type Image 변경 (compressed_imgmsg_to_cv2)
            2. 차선 영역만 ROI 지정 (applyROI)
            3. Bird-eye View 변환 (applyBirdEyeView)
            4. ROI 영역에서 윤곽선 검출 (applyCanny)
            5. 직선 검출 (applyHoughLine)
            6. 검출된 직선을 기반으로 차선의 모멘트 계산 (calcLaneDistance)
            7. 최종 검출된 값을 기반으로 카메라 좌표계 기준 차선들의 중심 x, y 좌표 계산하고 Publish
        '''
        self.frame = self.cvbridge.compressed_imgmsg_to_cv2(img, "bgr8") #1
        self.roi_image = self.applyROI(self.frame) #2
        self.wrap_image = self.applyBirdEyeView(self.roi_image) #3
        self.edge_image = self.applyCanny(self.wrap_image) #4
        self.lines = self.applyHoughLine(self.edge_image) #5
        self.distance = self.calcLaneDistance(self.lines) #6 #7

        self.drawHoughLines(img, self.lines)

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