import tensorflow as tf
from nets import *

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

placeholder_dict = {}
#var_dict = {}

conv_layer = Net(
    [Apply(conv2d),
     Apply(add_bias),
     Apply(tf.nn.relu), 
     Apply(tf.nn.max_pool, {"ksize":[1,2,2,1], "strides":[1,2,2,1], "padding":'SAME'})])

model = compile_net(Net(
    [Apply(tf.reshape, {"shape":[-1,28,28,1]}),
     Scope("conv1", [
       InitVar("W", [5,5,1,32]),
       InitVar("b", [32]),
       conv_layer]),
     Scope("conv2", [
       InitVar("W", [5,5,32,64]),
       InitVar("b", [64]),
       conv_layer]),
     Apply(tf.reshape, {"shape":[-1,7*7*64]}),
     Scope("fc", [
       InitVar("W",[7*7*64,1024]),
       InitVar("b",[1024]),
       Placeholder("keep_prob",dtype=tf.float32),
       Apply(relu_layer),
       Apply(tf.nn.dropout)]),
     Scope("readout", [
       InitVar("W", [1024,10]),
       InitVar("b", [10]),
       Apply(softmax_layer)])]))
       
def _mnist_fs(x, y_):
    """
    y_: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
    """
#    global var_dict
    logits = model(x)
    labels = tf.to_int64(y_)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    #Return the number of true entries.
    accuracy =  tf.reduce_sum(tf.cast(correct, tf.int32))
    return {"inference":logits, "loss":loss, "accuracy":accuracy, "losses":[]}

def mnist_fs(batch_size):
    global placeholder_dict
    x = variable_on_cpu("x", shape=(batch_size,IMAGE_PIXELS), var_type="placeholder")
    y_ = variable_on_cpu("y_", shape=(batch_size), var_type="placeholder")
    #tf.placeholder(tf.float32, shape=(batch_size,IMAGE_PIXELS), name="x")
    #y_ = tf.placeholder(tf.int32, shape=(batch_size), name="y_")
    placeholder_dict["x"] = x
    placeholder_dict["y_"] = y_
    return _mnist_fs(x,y_)
