## Project implementation effect
! [insert picture description here] (https://img-blog.csdnimg.cn/48f2b63643bb4ebc83d12ec40380c598.png)

### Add
**PS: The project address will be open source at the end **

This project uses **TensorFlow-GPU** for training: A **CUDA environment ** needs to be built in advance. For details, please refer to this article: [TensorFlow GPU - against 2.4.1 and CUDA installation tutorial] (https://blog.csdn.net/weixin_50679163/article/details/124395836?spm=1001.2014.3001.5502)
### Model data
#### embedded model
```bash
Model: "embedding"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
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
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
Total params: 38,960,448
Trainable params: 38,960,448
Non-trainable params: 0
_________________________________________________________________
` ` `
#### CNN neural network model
```bash
Model: "SiameseNetWork"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
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
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
Total params: 38,964,545
Trainable params: 38,964,545
Non-trainable params: 0
__________________________________________________________________________________________________
` ` `

# Project Overview
## Project operation process
**1. Collect face data - set the path of the data and preprocess the data set **

**2. Build training models -- Build deep neural networks **

**3. Deep training face data -- CNN Convolutional Neural Network +TensorFlow+Keras**

**4. Build a face recognition APP -- OpenCV+Kivy.APP**

## Core environment configuration
**Python == 3.9.0
labelme == 5.0.1
tensorflow-gpu == 2.7.0 (CUDA11.2)
opencv-python == 4.0.1
Kivy == 2.1.0
albumentations == 0.7.12**
# Project core code details
## Directory
! [insert picture description here] (https://img-blog.csdnimg.cn/767c43c5c01f4266b9dd9a4287c81637.png)


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

* * this item with the wild face data set download address: [] deep learning face training data set (https://download.csdn.net/download/weixin_50679163/86500793) * *

** This project is based on the paper "Siamese Neural Networks for One-shot Image Recognition" as the theoretical basis: [Siamese Neural Networks for One-shot Image Recognition](https://download.csdn.net/download/weixin_50679163/86500796)**
! [insert picture description here] (https://img-blog.csdnimg.cn/985203de43e54464a74aa9fa5251b7b0.png)
! [insert picture description here] (https://img-blog.csdnimg.cn/5401d7e389124484ac9f48cf7a40fec8.png)
# Core code
** Introduced core library files: **

```python
import cv2
import numpy as np
## Finally in the APP folder, run FaceAPP.py
