import tensorflow as tf
from mnist_model import *
from tensorflow.examples.tutorials.mnist import input_data
#import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
tf.app.flags.DEFINE_integer('eval_steps', 1000, 'Number of steps to run trainer.')
tf.app.flags.DEFINE_string('train_dir', 'train', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('fake_data', False, 'Use fake data.  ')

class BatchFeederD:
  def __init__(self, data):
    self.data = data
    self.num_examples = data.num_examples
  def next_batch(self, batch_size, *args):
    (x, y) = self.data.next_batch(batch_size, *args)
    return {"x": x, "y": y}
    

def main(argv=None):
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
  step = lambda fs, global_step: (
      train_step(fs["loss"], fs["losses"], global_step, 
                 lambda gs: tf.train.AdamOptimizer(1e-4)))
  with tf.Graph().as_default():
    train(lambda: mnist_fs(FLAGS.batch_size), 
          step, 
          max_steps=FLAGS.max_steps, 
          eval_steps=FLAGS.eval_steps,
          train_dir=FLAGS.train_dir,
          batch_size=FLAGS.batch_size,
          train_data=BatchFeederD(data_sets.train),
          validation_data=BatchFeederD(data_sets.validation),
          test_data=BatchFeederD(data_sets.test),
          train_feed={"fc/keep_prob:0": 0.5},
          eval_feed={"fc/keep_prob:0": 1.0},
          args_pl = {"x" : "x:0", "y" : "y_:0"},
          batch_feeder_args={FLAGS.fake_data}) 

if __name__ == '__main__':
  tf.app.run()
