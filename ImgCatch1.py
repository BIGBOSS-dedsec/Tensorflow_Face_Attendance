# Coding BIGBOSSyifi
# Datatime:2022/4/25 14:39
# Filename:ImgCatch.py
# Toolby: PyCharm
# 本代码调用opencv模块，通过摄像头可以进行拍照 ：采集 anchor图像 按键'A'  采集positive图像 按键'P' 按q键退出 (waitKey的delay时间单位为ms)

import cv2
import uuid
import os

# 设置文件数据集读取路径
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# 调用opencv库，调用摄像头捕捉
cap = cv2.VideoCapture(0)       #设置摄像头路径：0为内置摄像头    1为USB摄像头
while cap.isOpened():
    ret, frame = cap.read()

    # 将视频窗口裁剪为250px:250px———内部形参(X, Y, opencv色彩通道)
    frame = frame[120:120+250,200:200+250, :]

    cv2.imshow('Image Collection', frame)       # 建立视频窗口

    # 采集 anchor图像 按键'A'
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # 为每一个图片生成唯一的UUID标识符路径——(uuid1类型)
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # 储存 anchor图像
        cv2.imwrite(imgname, frame)
        print("anchor added successfully")

    # 采集positive图像 按键'P'
    if cv2.waitKey(1) & 0XFF == ord('s'):
        # 为每一个图片生成唯一的UUID标识符路径——(uuid1类型)
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # 储存 positive图像
        cv2.imwrite(imgname, frame)
        print("positive added successfully")

    # 按q键退出 (waitKey的delay时间单位为ms)：
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()       # 释放视频窗口
cv2.destroyWindow()     # 销毁视频窗口

