#!/opt/local/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
#재생할 파일 
#VIDEO_FILE_PATH = '0'
left_ear_cascade = cv2.CascadeClassifier('./Ear_detect/haarcascades/update_haarcascade_leftear.xml')
# 동영상 파일 열기
cap = cv2.VideoCapture(0)
if left_ear_cascade.empty():
  raise IOError('Unable to load the left ear cascade classifier xml file')

print (cap)

#print (cv2.CAP_PROP_FRAME_WIDTH)
#print (cv2.CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
#잘 열렸는지 확인
if cap.isOpened() == False:
    print ('Can not open the video ')
    exit()

titles = ['orig']
#윈도우 생성 및 사이즈 변경
for t in titles:
    cv2.namedWindow(t)

#재생할 파일의 넓이 얻기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#재생할 파일의 높이 얻기
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#재생할 파일의 프레임 레이트 얻기
fps = cap.get(cv2.CAP_PROP_FPS)

print('width {0}, height {1}, fps {2}'.format(width, height, fps))

#XVID가 제일 낫다고 함.
#linux 계열 DIVX, XVID, MJPG, X264, WMV1, WMV2.
#windows 계열 DIVX
#저장할 비디오 코덱
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#저장할 파일 이름
filename = 'sprite_with_face_detect.avi'

#파일 stream 생성
out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
#filename : 파일 이름
#fourcc : 코덱
#fps : 초당 프레임 수
#width : 넓이
#height : 높이


#left,right ear cropped image numbering
l_ear_num = 0
r_ear_num = 0


timestart = time.time()
start=0

while(True):
    #파일로 부터 이미지 얻기
    ret, frame = cap.read()
    #더 이상 이미지가 없으면 종료
    #재생 다 됨
    if frame is None:
        break;

    #얼굴인식 영상 처리
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blur =  cv2.GaussianBlur(grayframe,(5,5), 0)
    left_ear = left_ear_cascade.detectMultiScale(grayframe, 1.3, 5)

    for (x, y, w, h) in left_ear:
        l_ear_path="./Ear_image/Test/Before_preprocess/%04d.jpg"%l_ear_num
        a = x + 32
        #a = x -5
        b = y + 10
        #b = y -5
        c = x + w -30
        d = y + h - 6
        cv2.rectangle(frame, (a, b), (c, d), (0, 255, 0), 1)
        crop_left_ear = frame[(b+2): (d-2), (a+2):(c-2)]
        cv2.imwrite(l_ear_path, crop_left_ear)
        l_ear_num+=1


    # 얼굴 인식된 이미지 화면 표시
    cv2.imshow(titles[0],frame)

    # 인식된 이미지 파일로 저장
    out.write(frame)

    #1ms 동안 키입력 대기

    if len(left_ear)!=0:
        if start == 0:
            timestart = time.time()
            start = 1

    if ((cv2.waitKey(1) & 0xFF == ord('q')) | (time.time() - timestart > 5)& start==1):
        break

#재생 파일 종료
cap.release()
#저장 파일 종료
out.release()
#윈도우 종료
cv2.destroyAllWindows()
