#2018-11-27
#打开摄像头，识别人脸

import os
os.chdir('C:\\Users\\VISSanKCJ\\Desktop\\AI工程师计算机视觉\\CLASSDATA_第三门_计算机视觉库OpenCV（9.3更新）\\10.人脸检测识别')

import cv2
from imutils import *

# 0代表从摄像头获取图像数据，也可输入视频路径，读取视频
cap = cv2.VideoCapture(0)

while(True):
    #有两个返回值
    # ret 读取成功True或失败Falseq
    # frame读取到的图像的内容
    # 读取一帧数据，一帧代表一幅图像
    ret,frame = cap.read()

    # 级联分类器
    detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    rects = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=2, minSize=(10,10), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in rects:
	    # 画矩形框
	    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    
    cv2.imshow('frame',frame)
    #给图片取个名字'frame'
    # waitKey功能是不断刷新图像，单位ms，下面为延时1毫秒，延时后会读取到键盘的一个输入，有个返回值是当前键盘按键值
    # ord返回对应的ASCII数值
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

