# -*- coding: UTF-8 -*-
import keras
import tensorflow as tf
from keras2onnx import convert_keras

def keras_to_onnx(model, output_filename):
   onnx = convert_keras(model, output_filename)
   with open(output_filename, "wb") as f:
       f.write(onnx.SerializeToString())

semantic_model = keras.models.load_model('D:/ai/模型文件/facenet_keras.h5')
keras_to_onnx(semantic_model, 'semantic_segmentation.onnx')