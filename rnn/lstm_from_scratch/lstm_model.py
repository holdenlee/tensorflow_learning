import tensorflow as tf
from nets import *

#C, h
def step_lstm1(x, mem, Wf, bf, Wi, bi, WC, bC, Wo, bo, Wo1, bo1):
    C = mem[0]
    h = mem[1]
    #print(h)
    #print(x)
    #print(tf.pack(batches*[h]))
    #print(x)
    hx = tf.concat(1, [h,x]) #dimension m+n
    f = tf.sigmoid(tf.matmul(hx, Wf) + bf) #dimension m
    i = tf.sigmoid(tf.matmul(hx, Wi) + bi) #dimension m
    C_add = tf.tanh(tf.matmul(hx, WC) + bC) #dimension m
    C1 = f * C + i * C_add #dimension m
    o = tf.sigmoid(tf.matmul(hx, Wo) + bo) #dimension m
    h1 = o * tf.tanh(C1) #dimension m
    #print(h1)
    #print(Wo1)
    #print(bo1)
    out = tf.nn.softmax(tf.matmul(h1, Wo1) + bo1) #! do softmax
    return (out, (C1, h1)) #dimension m+m

#def step_rnn(x, m, W, b, 

#def step_lstm(x, h, m, n):
#    Wf = variable_on_cpu("Wf", [m+n,m], initializer=initializer)

#scanl :: (c -> a -> (b,c)) -> c -> [a] -> Int -> [b]
#just do repeat first
#l is length (can I access length of tensor?)
def scan(f, start, li, l, repeat=True, scope=""):
    #print(start)
    #print(li)
    cur = start
    outs = []
    for i in range(l):
        (out, cur) = f(cur, li[i])
        outs.append(out)
    return (outs, cur)

def indexed_lstm_fs(batches, l, m, n):
    xs = variable_on_cpu("xs", shape = [l,batches], var_type="placeholder", dtype=tf.uint8)
    ys = variable_on_cpu("ys", shape = [l,batches], var_type="placeholder", dtype=tf.uint8)
    xs = tf.one_hot(xs, n)
    ys = tf.one_hot(ys, n)
    return lstm_fs_(xs, ys, batches, l, m, n)

def lstm_fs_(xs, ys, batches, l, m, n):
    #(name, shape=None, initializer=None,dtype=tf.float32, var_type="variable")
    [Wf, Wi, WC, Wo] = map(lambda name: variable_on_cpu(name, shape=[m+n,m], initializer=tf.truncated_normal_initializer(stddev=1e-2)), ["Wf", "Wi", "WC", "Wo"])
    Wo1 = variable_on_cpu( "Wo1", shape=[m, n], initializer=tf.truncated_normal_initializer(stddev=1e-2))
    [bf, bi, bC, bo] = map(lambda name: variable_on_cpu(name, shape=[m], initializer=tf.truncated_normal_initializer(stddev=1e-2)), ["bf", "bi", "bC", "bo"])
    bo1 = variable_on_cpu( "bo1", shape=[n], initializer=tf.truncated_normal_initializer(stddev=1e-2))
    # C = variable_on_cpu("C", shape=[m], var_type="variable")
    # h = variable_on_cpu("h", shape=[m], var_type="variable")
    #C = tf.ones([batches,m])
    C = tf.zeros([batches,m])
    #h = tf.zeros([m])
    #h = tf.ones([batches,m])
    h = tf.zeros([batches,m])
    (outs, end) = scan(lambda mem, x: step_lstm1(x, mem, Wf, bf, Wi, bi, WC, bC, Wo, bo, Wo1, bo1), 
                       (C,h), xs, l)
    yhats = tf.pack(outs)
    #print(ys)
    #print(yhats)
    loss = cross_entropy(ys, yhats,t=1e-6)
    #tf.nn.sparse_softmax_cross_entropy_with_logits(outs, yhats, name='xentropy')
    #loss = cross_entropy(outs, yhats)
    #is not actually accuracy
    accuracy = cross_entropy(ys[-1], yhats[-1])
    #tf.nn.sparse_softmax_cross_entropy_with_logits(outs[-1], yhats[-1])
    return {"loss": loss, "inference": yhats, "accuracy": accuracy}

def lstm_fs(batches, l, m, n):
    xs = variable_on_cpu("xs", shape=[l,batches,n], var_type="placeholder")
    ys = variable_on_cpu("ys", shape=[l,batches,n], var_type="placeholder")
    return lstm_fs_(xs, ys, batches, l, m, n)

def eval_step_lstm_fs(m, n):
    C = variable_on_cpu("C", shape=[m], var_type="placeholder")
    C = tf.expand_dims(C, 0)
    h = variable_on_cpu("h", shape=[m], var_type="placeholder")
    h = tf.expand_dims(h, 0)
    x = variable_on_cpu("x", shape=[n], var_type="placeholder")
    xs = tf.expand_dims(x, 0)
    # copied from above
    [Wf, Wi, WC, Wo] = map(lambda name: variable_on_cpu(name, shape=[m+n,m]), ["Wf", "Wi", "WC", "Wo"])
    #, initializer=tf.truncated_normal_initializer()
    Wo1 = variable_on_cpu( "Wo1", shape=[m, n])
    [bf, bi, bC, bo] = map(lambda name: variable_on_cpu(name, shape=[m]), ["bf", "bi", "bC", "bo"])
    # , tf.truncated_normal_initializer()
    bo1 = variable_on_cpu( "bo1", shape=[n])
    (outs, (C1, h1)) = step_lstm1(xs, (C, h), Wf, bf, Wi, bi, WC, bC, Wo, bo, Wo1, bo1)
    out = outs[0]
    C1 = C1[0]
    h1 = h1[0]
    return {"inference": out, "prediction" : tf.argmax(out, 0), "C": C1, "h": h1}

def e_(i,n):
    return [1 if j==i else 0 for j in range(n)]

if __name__=="__main__":
    l = 20
    m = 4
    n = 5
    fs = lstm_fs(2, l, m, n)
    loss = fs["loss"]
    li = [[e_(i % n, n), e_(i % n, n)] for i in range(l+1)]
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    ans = sess.run(loss, {"xs:0": li[0:l], "ys:0": li[1:]})
    print(ans)
