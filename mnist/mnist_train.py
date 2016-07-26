import tensorflow as tf
from mnist_model import *
from tensorflow.examples.tutorials.mnist import input_data
#import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
tf.app.flags.DEFINE_integer('eval_steps', 1000, 'Number of steps to run trainer.')
tf.app.flags.DEFINE_string('train_dir', '/tigress/holdenl/tmp/mnist_train', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('fake_data', False, 'Use fake data.  ')

def main(argv=None):
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
  #losses not created yet??
  losses = tf.get_collection('losses')
  step = lambda loss, global_step: (
      train_step(loss, losses, global_step, 
                 lambda gs: tf.train.AdamOptimizer(1e-4)))
#  def feed_dict_fun():
    #tf.get_variable_scope().reuse_variables()
#    return merge_two_dicts(
#      fill_feed_dict(data_sets.train, FLAGS.batch_size, 
#                     placeholder_dict["x"],
#                     placeholder_dict["y_"]),
#      {"fc/keep_prob:0": 0.5})
#tf.Graph.get_tensor_by_name
                          #variable_on_cpu("x", shape=(batch_size,IMAGE_PIXELS), var_type="placeholder"), 
                          #variable_on_cpu("y_", shape=(batch_size), var_type="placeholder"))
#tf.get_variable("x"), tf.get_variable("y_"))
  train(lambda: mnist_fs(FLAGS.batch_size), 
        step, 
        max_steps=FLAGS.max_steps, 
        eval_steps=FLAGS.eval_steps,
        train_dir=FLAGS.train_dir,
        batch_size=FLAGS.batch_size,
        train_data=data_sets.train,
        validation_data=data_sets.validation,
        test_data=data_sets.test,
#        feed_dict_fun=feed_dict_fun,
        train_feed={"fc/keep_prob:0": 0.5},
        eval_feed={"fc/keep_prob:0": 1.0},
        x_pl = "x:0",
        y_pl = "y_:0",
        batch_feeder_args={FLAGS.fake_data}) 
#placeholder_dict["x"], placeholder_dict["y"]))
#tf.get_variable("x", reuse=True), tf.get_variable("y_",reuse=True)))
#ValueError: Variable W already exists, disallowed. Did you mean to set reuse=True in VarScope? ?????
#def train(fs, step_f, output_steps=10, summary_steps=100, save_steps=1000, eval_steps = 1000, max_steps=1000000, train_dir="/", log_device_placement=False, batch_size=128,train_data=None,validation_data=None, test_data=None, feed_dict_fun = None):

if __name__ == '__main__':
  #tf.get_variable_scope().reuse_variables()
  tf.app.run()
