## Project implementation effect
![在这里插入图片描述](https://img-blog.csdnimg.cn/48f2b63643bb4ebc83d12ec40380c598.png)

### Model data
#### embedded model
```bash
Model: "embedding"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_image (InputLayer)     [(None, 100, 100, 3)]     0         
_________________________________________________________________
conv2d (Conv2D)              (None, 91, 91, 64)        19264     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 46, 46, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 40, 40, 128)       401536    
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 20, 20, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 17, 128)       262272    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 9, 9, 128)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 6, 256)         524544    
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0         
_________________________________________________________________
dense (Dense)                (None, 4096)              37752832  
=================================================================
Total params: 38,960,448
Trainable params: 38,960,448
Non-trainable params: 0
_________________________________________________________________
```
#### CNN neural network model
```bash
Model: "SiameseNetWork"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_image (InputLayer)        [(None, 100, 100, 3) 0                                            
__________________________________________________________________________________________________
validation_img (InputLayer)     [(None, 100, 100, 3) 0                                            
__________________________________________________________________________________________________
embedding (Functional)          (None, 4096)         38960448    input_image[0][0]                
                                                                 validation_img[0][0]             
__________________________________________________________________________________________________
distance (L1Dist)               (None, 4096)         0           embedding[4][0]                  
                                                                 embedding[5][0]                  
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            4097        distance[0][0]                   
==================================================================================================
Total params: 38,964,545
Trainable params: 38,964,545
Non-trainable params: 0
__________________________________________________________________________________________________
```

# Project Overview
## Project operation process
**1. Collect face data - set the path of the data and preprocess the data set **

**2. Build training models -- Build deep neural networks **

**3. Deep training face data -- CNN Convolutional Neural Network +TensorFlow+Keras**

**4. Build a face recognition APP -- OpenCV+Kivy.APP**

## Core environment configuration
**Python == 3.9.0
labelme == 5.0.1
tensorflow -gpu == 2.7.0 （CUDA11.2）
opencv-python == 4.0.1
Kivy == 2.1.0
albumentations == 0.7.12**
# Project core code details
## Directory
![在这里插入图片描述](https://img-blog.csdnimg.cn/767c43c5c01f4266b9dd9a4287c81637.png)


| Name | Use |
|--|--|
| data | collected face data |
| data-anchor | Face data of the tested |
|data-negative | confuses the data set |
| data-positive | After pretreatment, face data |
| training_checkpoints | Training dataset logs (checkpoints) |
|.h5 | The trained face model (.h5) |
|ImgPath0.py | Sets the directory of the dataset |
|ImgCatch1.py | mobile phone face data |
|ImgPreprocess2.py | Image preprocessing |
|Model_Engineering3 | Build training model |
| training. py | deep Training dataset |
|cvOS.py | facial recognition APP |
|TensorFlowTest.py | CUDA environment test |

**This project used to download the address to the wild face dataset：[Deep learning face training dataset](https://download.csdn.net/download/weixin_50679163/86500793)**

**This project is based on the paper "Siamese Neural Networks for One-shot Image Recognition"：[Siamese Neural Networks for One-shot Image Recognition](https://download.csdn.net/download/weixin_50679163/86500796)**
![在这里插入图片描述](https://img-blog.csdnimg.cn/985203de43e54464a74aa9fa5251b7b0.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/5401d7e389124484ac9f48cf7a40fec8.png)
# Core code
** Introduced core library files: **
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
```

** Added GPU memory growth limit - prevent flash memory **

```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
```
# Set the data set directory

```python
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

# 导入野生数据集
for directory in os.listdir('666'):
    for file in os.listdir(os.path.join('666', directory)):
        EX_PATH = os.path.join('666', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)
```

# Collect face recognition data - UUID format naming

```python
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    # 裁剪图像大小250x250px
    frame = frame[120:120+250,200:200+250, :]
    
    # 收集人脸数据——正面清晰的数据集
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # 对数据进行UUID命名
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # 写入并保存数据
        cv2.imwrite(imgname, frame)
    
    # 收集数据集——侧脸斜脸的数据集（可以较模糊）
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # 对数据进行UUID命名
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # 写入并保存数据
        cv2.imwrite(imgname, frame)
    
    cv2.imshow('Image Collection', frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
```

# Create a labeled dataset

```python
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()
exampple = samples.next()
```

# Build partitions of training and test data

```python
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

res = preprocess_twin(*exampple)


data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)``
```

# Create a model

```python
inp = Input(shape=(100,100,3), name='input_image')
c1 = Conv2D(64, (10,10), activation='relu')(inp)
m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)

mod = Model(inputs=[inp], outputs=[d1], name='embedding')
mod.summary()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/c90310f587954d3886a54a8ea0d5b922.png)

```python
def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    # 第一层卷积
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # 第二层卷积
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # 第三层卷积 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # 最终卷积
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')
    embedding = make_embedding()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/5a1d72b7d8434a16a1e6b93f26e2ae50.png)
# Build the distance layer

```python
# L1距离层
class L1Dist(Layer):
    
    # 初始化方法
    def __init__(self, **kwargs):
        super().__init__()
       
    # 数据相似度计算
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
        l1 = L1Dist()
        l1(anchor_embedding, validation_embedding)
```
# Build a neural network model

```python
input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))

inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)

siamese_layer = L1Dist()
distances = siamese_layer(inp_embedding, val_embedding)
classifier = Dense(1, activation='sigmoid')(distances)
siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/c7bf0911574a430fb7d2deac7eeced62.png)
# Deep training model
## Build loss values and optimizers

```python
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001
```
## Set up training checkpoints

```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
```
# Set up the training batch

```python
test_batch = train_data.as_numpy_iterator()
batch_1 = test_batch.next()

X = batch_1[:2]
y = batch_1[2]

@tf.function
def train_step(batch):
    
    # 日志记录
    with tf.GradientTape() as tape:     
        # 获得人脸数据
        X = batch[:2]
        # 获得标签
        y = batch[2]
        
        # yhat的值向上传递
        yhat = siamese_model(X, training=True)
        # 计算损失值
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # 计算渐变值
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # 计算更新的权重传递给模型
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # 返回损失值
    return loss
```
## Set the training loop

```python
def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch)
            progbar.update(idx+1)
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)
```

### Start training

```python
EPOCHS = 50000
train(train_data, EPOCHS)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/ba23010065944f4c9b0894f1dd1c8964.png)
### Save the model

```python
siamese_model.save('siamesemodel.h5')
```
### Load model
```python
model = tf.keras.models.load_model('siamesemodel.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
```
# Test the model recognition effect
```python
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
   
    if cv2.waitKey(10) & 0xFF == ord('v'):  
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        results, verified = verify(model, 0.9, 0.7)
        print(verified)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
# Face recognition APP - Window UI

```python
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
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
        self.model = tf.keras.models.load_model('siamesemodelPRO.h5', custom_objects={'L1Dist': L1Dist})

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
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/48f2b63643bb4ebc83d12ec40380c598.png)
