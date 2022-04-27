# Coding BIGBOSSyifi
# Datatime:2022/4/24 19:45
# Filename:TensorFlowTest.py
# Toolby: PyCharm
# 本代码是对TensorFlow-GPU可用性测试

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D,Dense,MaxPooling2D,Input,Flatten

# 避免爆内存，设置一个GPU内存增长的过度消耗
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
    print('GPU', tf.test.is_gpu_available())


'''
tf.compat.v1.disable_eager_execution()

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/gpu:1'):
    c = a + b

# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(c))'''