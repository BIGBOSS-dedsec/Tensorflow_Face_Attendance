# Coding BIGBOSSyifi
# Datatime:2022/4/27 22:07
# Filename:FaceAPP.py
# Toolby: PyCharm
# 本篇代码实现功能：加载模型通过摄像头进行验证 代码51可修改模型路径

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os
import numpy as np

# 向命运妥协法(CPU)：
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    # 相似性计算：
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# 构建APP布局：
class CamApp(App):

    def build(self):
        # 主界面布局：
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Start Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Uninitiated...", size_hint=(1, .1))

        # 添加按键功能
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # 加载tensorflow/keras模型
        self.model = tf.keras.models.load_model('siamesemodel100.h5', custom_objects={'L1Dist': L1Dist})

        # 设置cv2摄像捕捉
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    # 连续获取摄像头图像
    def update(self, *args):
        # 读取cv2的框架：
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]      # 对摄像捕捉图像裁剪

        # 翻转水平并将图像转换为纹理图像
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # 将图像从文件和转换器转换为100x100px
    def preprocess(self, file_path):
        # 读取路径图片
        byte_img = tf.io.read_file(file_path)
        # 加载路径图片
        img = tf.io.decode_jpeg(byte_img)

        # 预处理步骤-将图像大小调整为100x100x3 (3通道)
        img = tf.image.resize(img, (100, 100))
        # 将图像缩放到0到1之间
        img = img / 255.0

        # Return image
        return img

    # 验证人脸图像
    def verify(self, *args):
        # 指定阈值
        detection_threshold = 0.99
        verification_threshold = 0.8         # 近似值设置

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # 生成结果数组
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # 对模型进行预测(验证)
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # 检测阈值：高于该阈值的预测被认为是正的指标
        detection = np.sum(np.array(results) > detection_threshold)

        # 验证阈值：阳性预测/总阳性样本的比例
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # 设置APP文本
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # 输出验证结果
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified


if __name__ == '__main__':
    CamApp().run()