# Coding BIGBOSSyifi
# Datatime:2022/4/25 15:26
# Filename:ImgPreprocess.py
# Toolby: PyCharm
# 本代码预处理data下的img文件，统一剪裁100x100x3规格

import os
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# 设置文件数据集读取路径
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# 读取目录下的文件 (形参take = 取样数量)
anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(300)

'''
# 数据集输出测试
dir_test = anchor.as_numpy_iterator()
print(dir_test.next())'''


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)  # 读取文件路径

    # 通过TensorFlow的jpeg解码器(字节图像)来加载图像
    img = tf.io.decode_jpeg(byte_img)
    # 图像预处理：将图像全部剪裁到100x100x3
    img = tf.image.resize(img, (100, 100))
    # 图像预处理：图像的像素点规模0-1
    img = img / 255.0
    return img


'''
# 图像返回测试：
img = preprocess('data\\anchor\\64098282-c467-11ec-9d0e-8863df8ae9a1.jpg')
print(img.numpy().max)
plt.imshow(img)
plt.show()
'''

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


'''
#预处理图像数据输出测试：
res = preprocess_twin(*samples.next())
print(res[0])
plt.imshow(res[0])
plt.show()'''

# 孪生数据通道
data = data.map(preprocess_twin)  # 读取孪生样本数据集
data = data.cache()  # 样本混合
data = data.shuffle(buffer_size=1024)  # 设置样本缓冲区大小

'''
# 孪生数据通道输出测试：
samples = data.as_numpy_iterator()
# print(len(samples.next()))
#print(samples.next()[0])
plt.imshow(samples.next()[0])
plt.show()'''

'''
# 查看当前数据长度
print(data)
print(len(data))
'''

# 训练模块
train_data = data.take(round(len(data)*.7))     # 选取训练集的70% = 420
train_data = train_data.batch(16)       # batch通过值16
train_data = train_data.prefetch(8)

'''
print(round(len(data)*.7))      #420
print(round(len(data)*.3))      #180 
'''

'''
# 训练模块代码测试
train_samples = train_data.as_numpy_iterator()
train_sample = train_samples.next()
print(train_sample)
print(len(train_sample[0]))
'''

# 数据测试模块
test_data = data.skip(round(len(data)*.7))     # 选取训练集的70% = 420
test_data = test_data.take(round(len(data)*.3))     # 选取训练集最后的30% = 180
test_data = test_data.batch(16)       # batch通过值16
test_data = test_data.prefetch(8)