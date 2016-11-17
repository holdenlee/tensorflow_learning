from lstm_input import *
from lstm_model import *
from nets import *
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from nltk.book import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

"""
  # Decay the learning rate exponentially based on the number of steps.
def cifar_lr(global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.scalar_summary('learning_rate', lr)
    return lr
"""

def main(argv=None):  # pylint: disable=unused-argument
    string = " ".join(text1)
    bid = make_bidict(sorted(uniques(string)))
    l = len(string)
    m = 50
    n = len(bid)
    train_string = string[0:9*l/10]
    test_string = string[9*l/10:l]
    seq_length = 10
    [(xtrain, ytrain), (xtest, ytest)]= [one_hot_data(n,make_data(bid, seq_length, s, mode="all")) for s in [train_string, test_string]]
    train_data = make_batch_feeder(xtrain,ytrain)
    test_data = make_batch_feeder(xtest,ytest)
    summary_f = lambda global_step: tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    lr = 1e-3
    step = lambda fs, global_step: (
        train_step(fs["loss"], [], global_step, lambda gs: tf.train.GradientDescentOptimizer(lr)))
    train(lambda: lstm_fs(32, seq_length,m,n), step, 
          max_steps=FLAGS.max_steps, 
          eval_steps=1000,
          train_dir=FLAGS.train_dir,
          batch_size=10,
          train_data=train_data,
          test_data=test_data,
          validation_data=None,
          log_device_placement=FLAGS.log_device_placement)

if __name__ == '__main__':
  #tf.get_variable_scope().reuse_variables()
  tf.app.run()

"""
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
  step = lambda fs, global_step: (
      train_step(fs["loss"], fs["losses"], global_step, 
                 lambda gs: tf.train.AdamOptimizer(1e-4)))
  train(lambda: mnist_fs(FLAGS.batch_size), 
        step, 
        max_steps=FLAGS.max_steps, 
        eval_steps=FLAGS.eval_steps,
        train_dir=FLAGS.train_dir,
        batch_size=FLAGS.batch_size,
        train_data=data_sets.train,
        validation_data=data_sets.validation,
        test_data=data_sets.test,
        train_feed={"fc/keep_prob:0": 0.5},
        eval_feed={"fc/keep_prob:0": 1.0},
        x_pl = "x:0",
        y_pl = "y_:0",
        batch_feeder_args={FLAGS.fake_data}) 
"""
