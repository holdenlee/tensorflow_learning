import tensorflow as tf
from tensorflow.models.rnn import rnn

#xs is a list of length l, y is the next in the sequence. There are n labels overall. 
#OR
#xs is a list of lists of length l (matrix), y is a list.
def lstm(xs, l, size, num_layers, initial_state=None):
    batch_size = tf.size(xs)[0]
    n = tf.size(xs)[-1]
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    #add dropout
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
    if initial_state == None:
        initial_state = cell.zero_state(batch_size, data_type=tf.float32)
    inputs = tf.one_hot(xs, n)
#    inputs = [tf.squeeze(input_, [1])
#              for input_ in tf.split(1, num_steps, inputs)]
    outputs, _ = rnn.rnn(cell, inputs, initial_state=initial_state)
    #state
    return outputs

def loss(outputs, y):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      outputs, y, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy') #reduce all dimensions
  return cross_entropy_mean

#(xs, l, size, num_layers, initial_state=None)

def inference(xs, y, l, num_layers=1, initial_state=None):
    #??
    #variable_on_cpu(name, shape, initializer=initializer)
    output = lstm(xs, l, size, num_layers, initial_state)
