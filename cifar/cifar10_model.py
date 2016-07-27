# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10_input

import cifar10_input
from nets import *

NUM_CLASSES=10

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tigress/knv/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
        
def conv_layer(x, kernel, b, name):
    return compile_net(Net([
         Apply(tf.nn.conv2d, {"filter":kernel, "strides":[1, 1, 1, 1], "padding":'SAME'}),
         Apply(tf.nn.bias_add,{"bias":b}),
         Apply(tf.nn.relu,{"name":name}),
         Apply(activation_summary)]))(x)

lrn_args = {"depth_radius":4, "bias":1.0, "alpha":0.001 / 9.0, "beta":0.75}
pool_params = {"ksize":[1, 3, 3, 1], "strides":[1, 2, 2, 1], "padding":'SAME'}
dim = 6*6*64
#each max-pooling halves the height and width. There are 64 channels.
#w, h = get_conv2d_dims(input_dim, filter_dim, strides)

#initialize variables
var_list = join(
    [map(lambda sc,sh: ("kernel", sh, tf.truncated_normal_initializer(stddev=1e-4), sc),
         ["conv1", "conv2"],
         [[5,5,3,64],[5,5,64,64]]),
     map(lambda sc, d, c: ("b", [d], tf.constant_initializer(c), sc),
         ["conv1", "conv2", "local3", "local4", "softmax_linear"],
         [64, 64, 384, 192, NUM_CLASSES],
         [0.0,0.1,0.1,0.1,0]),
     map(lambda sc, sh, std: ("W", sh, tf.truncated_normal_initializer(stddev=std), sc),
         ["local3","local4", "softmax_linear"],
         [[dim, 384], [384, 192], [192, NUM_CLASSES]],
         [0.04, 0.04, 1/192.0])])

def get_dim(x):
    print(x.get_shape())
    return x

#tf.get_variable_scope().reuse_variables()
inference = compile_net(Net(
        [InitVars(var_list),
         Scope("conv1", [
            Apply(conv_layer, {"name":"conv1"}),
            Apply(tf.nn.max_pool, merge_two_dicts(pool_params, {"name":'pool1'})),
            #local response normalization: http://stackoverflow.com/questions/37376861/what-does-the-tf-nn-lrn-method-do
             Apply(tf.nn.lrn, merge_two_dicts(lrn_args, {"name":"norm1"}))]),
         Apply(get_dim),
         Scope("conv2", [
            Apply(conv_layer, {"name":"conv2"}),
            Apply(tf.nn.lrn, merge_two_dicts(lrn_args, {"name":"norm2"})),
            Apply(tf.nn.max_pool, merge_two_dicts(pool_params, {"name":'pool2'}))]),
         Apply(get_dim),
         Apply(tf.reshape, {"shape": [FLAGS.batch_size, -1]}),
         Apply(get_dim),
         Scope("local3", [
            Apply(relu_layer),
            Apply(activation_summary)]),
         Scope("local4", [
            Apply(relu_layer),
            Apply(activation_summary)]),
         Scope("softmax_linear", [
            Apply(linear_layer, {}),
            Apply(activation_summary)])]))

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  weight_decays = mapkw(weight_decay,
                        [(tf.get_default_graph().get_tensor_by_name("local3/W:0"), 0.004),
                         (tf.get_default_graph().get_tensor_by_name("local4/W:0"), 0.004)])
  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  weight_decays.append(cross_entropy_mean)
  return tf.add_n(weight_decays, name='total_loss')

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)

def cifar_fs():
    images, labels = distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = inference(images)

    # Calculate loss.
    total_loss = loss(logits, labels)
    return {"inference": logits, "loss": total_loss, "losses": tf.get_collection("losses")}
