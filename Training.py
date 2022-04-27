# Coding BIGBOSSyifi
# Datatime:2022/4/27 2:07
# Filename:Training.py
# Toolby: PyCharm

# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf


'''
# 避免爆内存，设置一个GPU内存增长的过度消耗
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
'''

# 向命运妥协法(CPU)：
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

''''''
# 数据测试模块
test_data = data.skip(round(len(data) * .7))  # 选取训练集的70% = 420
test_data = test_data.take(round(len(data) * .3))  # 选取训练集最后的30% = 180
test_data = test_data.batch(128)  # batch通过值16
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


siamese_model = make_siamese_model()
siamese_model.summary()

'''
# 模型训练部分
'''
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

# 建立模型训练检查点(方便数据回滚)
checkpoint_dir = './training_checkpoints'       # 建立检查点路径
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)      # 遍历检查点集

'''
# batch数据返回测试
test_batch = train_data.as_numpy_iterator()
# print(test_batch.next())
# print(len(test_batch.next()))
# print(test_batch.next()[0])
# print(test_batch.next()[2])
# print(len(test_batch.next()[0]))

X = test_batch.next()[:2]
# print(len(X))
i = np.array(X).shape
# print(i)
Y = test_batch.next()[2]
# print(Y)
'''

@tf.function
def train_step(batch):
    # 建立神经网络梯度模型
    with tf.GradientTape() as tape:
        # 获取anchor,positive/negative图像
        X = batch[:2]
        # 获取label参数：
        y = batch[2]

        # 获取siamese_model模型传值
        yhat = siamese_model(X, training=True)
        # 计算损失值
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # 计算gradient值
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # 计算模型权重并应用到Siamese_model中
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss


def train(data, EPOCHS):
    # EPOCHS训练循环循环
    for epoch in range(1, EPOCHS + 1):      # EPOCHS从1开始循环
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))     # 建立一个训练进度条

        # 对每一个each batch循环：
        for idx, batch in enumerate(data):
            # 调用训练方法
            train_step(batch)
            progbar.update(idx + 1)

        # 检查点保存
        if epoch % 10 == 0:         # 检查点每10个保存一次
            checkpoint.save(file_prefix=checkpoint_prefix)


# 训练模型：
EPOCHS = 100     # 训练次数
train(train_data, EPOCHS)       # 执行训练


# 获取测试数据
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
# 模型数据预测
y_hat = siamese_model.predict([test_input, test_val])
# 对结果进行后处理
[1 if prediction > 0.5 else 0 for prediction in y_hat ]
# 创建度量对象
m = Recall()
# 计算召回值
m.update_state(y_true, y_hat)
# 返回召回结果
m.result().numpy()
# 设置绘图大小
plt.figure(figsize=(10,8))
# 设置第一个子地块
plt.subplot(1,2,1)
plt.imshow(test_input[0])
# 设置第二个子地块
plt.subplot(1,2,2)
plt.imshow(test_val[0])
plt.show()


'''......................................加载模型部分.......................................'''
# 保存模型.h文件
siamese_model.save('siamesemodel.h5')

# 重载模型  (compile=False)
model = tf.keras.models.load_model('siamesemodel.h5',
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy}, compile=False)
# 使用重载的模型进行预测
model.predict([test_input, test_val])
# 输出模型摘要
model.summary()


# 定义对照图片路径：
os.listdir(os.path.join('application_data', 'verification_images'))
os.path.join('application_data', 'input_image', 'input_image.jpg')      # 定义输出图片路径和路径名

# 循环遍历验证图片：
for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join('application_data', 'verification_images', image)
    # print(validation_img)

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
