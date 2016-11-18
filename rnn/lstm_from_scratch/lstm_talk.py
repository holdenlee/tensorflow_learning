from lstm_input import *
from lstm_model import *
from nets import *
import tensorflow as tf
import numpy as np
from nlp_process import *
from utils import *
from nltk.book import *

if __name__ == "__main__":
    # just abcd...
    #string = "".join([lowercases for i in range(100)])
    #string = " ".join(text2)
    string = " ".join(text1)
    bid = make_bidict(sorted(uniques(string)))
    m = 200 #100
    n = len(bid)
    #print(n)

    output = 'a'
    x = e_(bid.inv[output], n)
    
    with tf.Session() as sess:
        d = eval_step_lstm_fs(m, n)
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver() 
        ckpt = tf.train.get_checkpoint_state("train_moby_dick/")

        C = np.zeros(m)
        #print(np.shape(C))
        h = np.zeros(m)
        saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(1000):
            d1 = sess.run(d, feed_dict = map_feed_dict({"x:0" : x, "C:0" : C, "h:0" : h}))
            C = d1["C"]
            h = d1["h"]
            a = d1["prediction"]
            inference = d1["inference"]
            selected = select(inference)
            x = e_(selected, n)
            #x = e_(d1["prediction"], n)
            output = output + bid[selected]
    print(output)
