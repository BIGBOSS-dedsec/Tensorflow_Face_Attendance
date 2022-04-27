# Coding BIGBOSSyifi
# Datatime:2022/4/27 19:00
# Filename:cvOS.py
# Toolby: PyCharm
# 本代码调用Training.py的preprocess功能，可对已训练的模型进行预验证

import cv2
import os
import numpy as np
from Training import preprocess, make_embedding, model

# 定义对照图片路径：
os.listdir(os.path.join('application_data', 'verification_images'))
os.path.join('application_data', 'input_image', 'input_image.jpg')      # 定义输出图片路径和路径名

# 循环遍历验证图片：
for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join('application_data', 'verification_images', image)
    print(validation_img)

# 模型验证：
def verify(model, detection_threshold, verification_threshold):
    # 生成结果数组
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        # 对数据进行预测
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # 检测阈值：高于该阈值的预测被认为是正的指标
    detection = np.sum(np.array(results) > detection_threshold)

    # 验证阈值：阳性预测和总阳性样本的比例 (positive)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified


'''
# 训练完后对模型进行测试
'''
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]      # 对摄像捕捉图像裁剪

    cv2.imshow('Verification', frame)

    # 验证触发器
    if cv2.waitKey(10) & 0xFF == ord('v'):      # 按V进行验证
        # 将输入图像保存到input_image文件夹
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # 运行验证
        results, verified = verify(model, 0.9, 0.7)

        print(verified)         # 输出验证结果

    if cv2.waitKey(10) & 0xFF == ord('q'):      # q退出
        break
cap.release()       # 释放缓存资源
cv2.destroyAllWindows()     # 销毁窗口