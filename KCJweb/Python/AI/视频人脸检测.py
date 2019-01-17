#2018-11-27
#打开摄像头，识别人脸

import os
os.chdir('C:\\Users\\VISSanKCJ\\Desktop\\AI工程师计算机视觉\\CLASSDATA_第三门_计算机视觉库OpenCV（9.3更新）\\10.人脸检测识别')

import cv2
from imutils import *

# 0代表从摄像头获取图像数据，也可输入视频路径，读取视频
cap = cv2.VideoCapture('1.mp4')

# 视频每秒传输帧数
fps = cap.get(cv2.CAP_PROP_FPS)
# 视频图像的宽度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 视频图像的长度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('2.avi',fourcc,fps,(frame_width,frame_height))

while(True):
    #有两个返回值
    # ret 读取成功True或失败Falseq
    # frame读取到的图像的内容
    # 读取一帧数据，一帧代表一幅图像
    ret,frame = cap.read()
    if ret==True:
        # 级联分类器
        detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        rects = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=2, minSize=(10,10), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in rects:
    	    # 画矩形框
    	    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        out.write(frame)
        cv2.imshow('frame',frame)
        # waitKey功能是不断刷新图像，单位ms，下面为延时1毫秒，延时后悔读取到键盘的一个输入，有个返回值是当前键盘按键值
        # ord返回对应的ASCII数值
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()

