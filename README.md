# Tensorflow_Face_Attendance
# Tensorflow_Face_Attendance项目使用说明

## 文件夹说明：

### APP中     ——application_data——input——image（验证结果输出文件夹）——verification_images（验证的模型图片建议从positive中随机选取50张图片）——FaceAPP.py

### 根目录中     ——application_data——input——image（验证结果输出文件夹）——verification_images（验证的模型图片建议从positive中随机选取50张图片）

### data中     ——anchor通过ImgCatch1中自定义要训练的人脸，建议录取300张；——positive中，通过ImgCatch1中自定义要训练的人脸，建议录取300张 	——negative中，内置了野生数据集（无需更改）

### training_checkpoints中————保存训练数据的节点（方便后期数据回滚）

## 在Pycham中导入本项目：

## py文件运行顺序：

## 0.先运行ImgPath0.py，预处理野生数据集（本项目已经处理好，无需运行，若需要训练其他类型数据集，可以按照py文件内部注释导入操作）

## 1.运行ImgCatch1.py，通过该文件调用摄像头，捕捉人脸图像，图像需录入两次：一次anchor文件夹，一次positive文件夹（具体录入按键操作根据py文件内注释提示操作）

## 1.1 运行ImgPreprocess2.py对录取图片的预处理（剪裁100x100x3）

1. ## 测试py文件

## 2.1  Model_Engineerin3.py运行该文件验证模型是否可用

## 2.2 Training.py为训练代码，运行该文件对已录入的图像按照模型进行训练

1. ## cvOS.py对已训练的模型进行预验证（可测试模型是否可用）

## PS : TensorFlowTest.py 该文件验证TensorFlow-GPU是否可用

## 最后在APP文件夹中，运行FaceAPP.py即可
