#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在两个文件夹下，比例改0.7这个值即可
import os
import random
import shutil
from shutil import copy2
trainfiles = os.listdir('/home/geekplusa/ai/datasets/my/fire/fire_110_labelimg/images')
num_train = len(trainfiles)
print( "num_train: " + str(num_train) )
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)
num = 0
trainDir = '/home/geekplusa/ai/datasets/my/fire/fire_110_labelimg/train_val/train'
validDir = '/home/geekplusa/ai/datasets/my/fire/fire_110_labelimg/train_val/val'
for i in index_list:
    fileName = os.path.join('/home/geekplusa/ai/datasets/my/fire/fire_110_labelimg/images', trainfiles[i])
    fileName_xml = os.path.join('/home/geekplusa/ai/datasets/my/fire/fire_110_labelimg/xml', trainfiles[i].split('.')[0] + '.xml')
    if num < num_train*0.8:
        print(str(fileName))
        print(str(fileName_xml))
        copy2(fileName, trainDir)
        copy2(fileName_xml, trainDir)
    else:
        copy2(fileName, validDir)
        copy2(fileName_xml, validDir)
    num += 1