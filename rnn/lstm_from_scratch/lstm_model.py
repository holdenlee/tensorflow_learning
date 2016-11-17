import tensorflow as tf
from nets import *

#C, h
def step_lstm1(x, mem, Wf, bf, Wi, bi, WC, bC, Wo, bo):
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
    out = tf.nn.softmax(h1) #! do softmax
    return (out, (C1, h1)) #dimension m+m

#def step_lstm(x, h, m, n):
#    Wf = variable_on_cpu("Wf", [m+n,m], initializer=initializer)

#scanl :: (c -> a -> (b,c)) -> c -> [a] -> Int -> [b]
#just do repeat first
#l is length (can I access length of tensor?)
def scan(f, start, li, l, repeat=True, scope=""):
    print(start)
    print(li)
    cur = start
    outs = []
    for i in range(l):
        (out, cur) = f(cur, li[i])
        outs.append(out)
    return (outs, cur)
    
def lstm_fs(batches, l, m, n):
    #(name, shape=None, initializer=None,dtype=tf.float32, var_type="variable")
    [Wf, Wi, WC, Wo] = map(lambda name: variable_on_cpu(name, shape=[m+n,m], initializer=tf.truncated_normal_initializer(stddev=1e-4)), ["Wf", "Wi", "WC", "Wo"])
    [bf, bi, bC, bo] = map(lambda name: variable_on_cpu(name, shape=[m], initializer=tf.truncated_normal_initializer(stddev=1e-4)), ["bf", "bi", "bC", "bo"])
    # C = variable_on_cpu("C", shape=[m], var_type="variable")
    # h = variable_on_cpu("h", shape=[m], var_type="variable")
    C = tf.zeros([batches,m])
    #h = tf.zeros([m])
    h = tf.zeros([batches,m])
    xs = variable_on_cpu("xs", shape=[l,batches,n], var_type="placeholder")
    ys = variable_on_cpu("ys", shape=[l,batches,n], var_type="placeholder")
    (outs, end) = scan(lambda mem, x: step_lstm1(x, mem, Wf, bf, Wi, bi, WC, bC, Wo, bo), 
                       (C,h), xs, l)
    yhats = tf.pack(outs)
    loss = cross_entropy(outs, yhats)
    return {"loss": loss, "inference": yhats}

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
