# Coding BIGBOSSyifi
# Datatime:2022/4/25 18:39
# Filename:Model_Engineering3.py
# Toolby: PyCharm
# 本代码将构建模型规格

import os
import numpy
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import matplotlib.pyplot as plt

# 设置文件数据集读取路径
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# 读取目录下的文件 (形参take = 取样数量)
anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(300)


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)  # 读取文件路径

    # 通过TensorFlow的jpeg解码器(字节图像)来加载图像
    img = tf.io.decode_jpeg(byte_img)
    # 图像预处理：将图像全部剪裁到100x100x3
    img = tf.image.resize(img, (100, 100))
    # 图像预处理：图像的像素点规模0-1
    img = img / 255.0
    return img


# (anchor , positive) => 1,1,1,1,1
# (anchor , negative) => 0,0,0,0,0
# 创建标记数据集-双子图像集(正样本和负样本)
positive = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positive.concatenate(negative)

samples = data.as_numpy_iterator()


# print(samples.next())

# 构建训练数据集
def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# 孪生数据通道
data = data.map(preprocess_twin)  # 读取孪生样本数据集
data = data.cache()  # 样本混合
data = data.shuffle(buffer_size=1024)  # 设置样本缓冲区大小

# 训练模块
train_data = data.take(round(len(data) * .7))  # 选取训练集的70% = 420
train_data = train_data.batch(16)  # batch通过值16
train_data = train_data.prefetch(8)

# 数据测试模块
test_data = data.skip(round(len(data) * .7))  # 选取训练集的70% = 420
test_data = test_data.take(round(len(data) * .3))  # 选取训练集最后的30% = 180
test_data = test_data.batch(16)  # batch通过值16
test_data = test_data.prefetch(8)


# 建立卷积神经网络嵌入层模型：
def make_embedding():
    # 模型输入层
    inp = Input(shape=(100, 100, 3), name='input_image')
    # 卷积神经网络第一层
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # 卷积神经网络第二层
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # 卷积神经网络第三层
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # 卷积神经网络最终层
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_embedding()
embedding.summary()


# print(make_embedding().summary())       # 模型输出测试

# 构建卷积神经网络距离层
class L1Dist(Layer):

    # Init 继承方法
    def __init__(self, **kwargs):
        super().__init__()

    # 模型相似度计算
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# print(L1Dist()) # 距离层测试
# l1 = L1Dist()
# L1(anchor_embedding,validation_embedding)

# 建立Siamese模型
def make_siamese_model():
    # 卷积网Anchor正极图像
    input_image = Input(name='input_image', shape=(100, 100, 3))

    # 卷积网Validation负极图像
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # 结合Siamese距离模型
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # 连接层
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetWork')

'''
# 模型输出测试
input_image = Input(name='input_image', shape=(100, 100, 3))
validation_image = Input(name='validation_img', shape=(100, 100, 3))
input_embedding = embedding(input_image)
validation_embedding = embedding(validation_image)
siamese_layer = L1Dist()
distances = siamese_layer(input_embedding, validation_embedding)
classifier = Dense(1, activation='sigmoid')(distances)
#print(classifier)
#print(siamese_layer(input_embedding, validation_embedding))
#print(input_embedding)
#print(validation_embedding)
#print(make_siamese_model())
#print(make_siamese_model().summary())'''

siamese_model = make_siamese_model()
siamese_model.summary()
print(siamese_model.summary())