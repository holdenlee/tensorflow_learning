from lstm_input import *
from lstm_model import *
from nets_old import *
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from nets import *
from nltk.book import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'train_moby_dick',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
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
    #string = " ".join(text2)
    string = " ".join(text1)
    #string = "".join([lowercases for i in range(100)])
    bid = make_bidict(sorted(uniques(string)))
    l = len(string)
    #m = 50
    #n = len(bid)
    #batch_size = 32
    m = 200
    n = len(bid)
    batch_size = 32
    train_string = string[0:9*l/10]
    test_string = string[9*l/10:l]
    seq_length = 1 #20 #10
    #make_data(bid, l, doc, mode="last"):
    [(xtrain, ytrain), (xtest, ytest)]= [np.asarray(make_data(bid, seq_length, s, mode="all")) for s in [train_string, test_string]]
#[np.asarray(one_hot_data(n,make_data(bid, seq_length, s, mode="all"))) for s in [train_string, test_string]]
    [train_data, test_data] = [BatchFeeder(args, None, 
                                           (lambda bf, batch_size: map_vals(lambda v: np.transpose(v),(batch_feeder_f(bf, batch_size))))) # , axes=[1,0,2]
        for args in [{"xs": xtrain, "ys": ytrain}, {"xs": xtest, "ys": ytest}]]#np.transpose, axes=[1,0,2]
    #print(np.shape(train_data.next_batch(batch_size)["xs"]))
    #print(np.shape(train_data.next_batch(batch_size)["ys"]))
    summary_f = lambda global_step: tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    lr = 1e-1 #e-1
    step = lambda fs, global_step: (
        train_step(fs["loss"], [], global_step, lambda gs: tf.train.GradientDescentOptimizer(lr)))
    train(lambda: indexed_lstm_fs(batch_size, seq_length,m,n), step, #lstm_fs
          max_steps=FLAGS.max_steps, 
          eval_steps=1000,
          train_dir=FLAGS.train_dir,
          batch_size=batch_size,
          train_data=train_data,
          test_data=test_data,
          validation_data=None,
          args_pl = {"xs" : "xs:0", "ys" : "ys:0"},
          log_device_placement=FLAGS.log_device_placement)

if __name__ == '__main__':
  #tf.get_variable_scope().reuse_variables()
  tf.app.run()


def test():
    np.set_printoptions(threshold=np.inf,precision=2)
    string = " ".join(text2)
    #string = " ".join(text1)
    #string = "".join([lowercases for i in range(100)])
    bid = make_bidict(sorted(uniques(string)))
    l = len(string)
    #m = 50
    #n = len(bid)
    #batch_size = 32
    m = 200
    n = len(bid)
    batch_size = 32
    train_string = string[0:9*l/10]
    test_string = string[9*l/10:l]
    seq_length = 20 #10
    #make_data(bid, l, doc, mode="last"):
    [(xtrain, ytrain), (xtest, ytest)]= [np.asarray(make_data(bid, seq_length, s, mode="all")) for s in [train_string, test_string]]
#[np.asarray(one_hot_data(n,make_data(bid, seq_length, s, mode="all"))) for s in [train_string, test_string]]
    [train_data, test_data] = [BatchFeeder(args, batch_size, 
                                           (lambda bf, batch_size: map_vals(lambda v: np.transpose(v),(batch_feeder_f(bf, batch_size))))) # , axes=[1,0,2]
        for args in [{"xs": xtrain, "ys": ytrain}, {"xs": xtest, "ys": ytest}]]
    with tf.Session() as sess:
        d = indexed_lstm_fs(1, seq_length, m, n)
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver() 
        ckpt = tf.train.get_checkpoint_state("train_sense/")
        C = np.zeros(m)
        #print(np.shape(C))
        h = np.zeros(m)
        saver.restore(sess, ckpt.model_checkpoint_path)
        xs = train_data.next_batch(1)["xs"]
        ys = train_data.next_batch(1)["ys"]
        d1 = sess.run(d, feed_dict = map_feed_dict({"xs:0" : xs, "ys:0" : ys}))
        inference = d1["inference"]
        loss = d1["loss"]
        print(xs)
        print(ys)
        print(deepmap(lambda i: bid[i], xs.tolist()))
        print(deepmap(lambda i: bid[i], ys.tolist()))
        print(inference)
        print(loss)
