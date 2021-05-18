# -*- coding: UTF-8 -*-
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
import keras.backend as K
K.set_learning_phase(0)

def keras_to_pb(model, output_filename, output_node_names):

   """
   This is the function to convert the Keras model to pb.

   Args:
      model: The Keras model.
      output_filename: The output .pb file name.
      output_node_names: The output nodes of the network. If None, then
      the function gets the last layer name as the output node.
   """

   # Get the names of the input and output nodes.
   in_name = model.layers[0].get_output_at(0).name.split(':')[0]

   if output_node_names is None:
       output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]

   sess = keras.backend.get_session()

   # The TensorFlow freeze_graph expects a comma-separated string of output node names.
   output_node_names_tf = ','.join(output_node_names)

   frozen_graph_def = tf.graph_util.convert_variables_to_constants(
       sess,
       sess.graph_def,
       output_node_names)

   sess.close()
   wkdir = ''
   tf.train.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)

   return in_name, output_node_names

# load the ResNet-50 model pretrained on imagenet
model = keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

# Convert the Keras ResNet-50 model to a .pb file
in_tensor_name, out_tensor_names = keras_to_pb(model, "models/resnet50.pb", None)