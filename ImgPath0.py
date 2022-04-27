# Coding BIGBOSSyifi
# Datatime:2022/4/24 17:33
# Filename:Facial.py
# Toolby: PyCharm
# 本代码对wild 数据集进行数据处理，具体将数据集下个文件夹的图片统一储存到一个文件夹中
# 数据集文件夹-lfw

import os

# 设置文件数据集读取路径
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# 创建文件夹
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

# 将数据集LFW文件夹下的每一个图片加入NEG_PATH路径地址
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

# 将数据集LFW文件夹下的每一个图片加入NEG_PATH路径地址
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        print(os.path.join('lfw', directory, file))
        print(os.path.join(NEG_PATH, file))

