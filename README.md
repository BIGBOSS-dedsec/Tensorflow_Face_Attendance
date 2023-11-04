# Tensorflow_Face_Attendance

## Folder description:

### APP -- application_data -- input -- image (verification result output folder) -- verification_images (it is recommended to randomly select 50 images from positive for verification model images) -- FaceAPP.py

### In the root directory -- application_data -- input -- image (verification result output folder) -- verification_images (It is recommended to randomly select 50 images from positive for verification model images)

In ### data -- anchor defines the faces to be trained through ImgCatch1, and it is recommended to admit 300 faces; -- In positive, customize the faces to be trained through ImgCatch1, and it is recommended to admit 300 faces -- in negative, wild data set is built in (no need to change)

### training_checkpoints ———— checkpoints (convenient for later data rollback)

## Import this project in Pycham:

## py file run order:

## 0. Run ImgPath0.py first to preprocess wild data sets (this project has already been processed, so there is no need to run it. If you need to train other types of data sets, you can follow the import operation of annotations inside the py file)

Run ImgCatch1.py to call the camera through the file and capture the face image. Input the image twice: once in the anchor folder and once in the positive folder. (For specific input, press the key according to the comments in the py file.)

## 1.1 Run ImgPreprocess2.py to preprocess the image (crop 100x100x3)

1. ## Test the py file

## 2.1 Model_Engineerin3.py Runs this file to verify that the model is available

## 2.2 Training.py is the Training code. Run this file to train the entered images according to the model

1. ## cvOS.py pre-validates the trained model (can test whether the model is available)

## PS: TensorFlowTest.py This file verifies that TensorFlow-GPU is available

## Finally in the APP folder, run FaceAPP.py
