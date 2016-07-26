import tensorflow as tf
from nets import *

#x = tf.placeholder(tf.float32, shape=[1], name="x")
#y = tf.placeholder(tf.float32, shape=[1], name="x")

"""
with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[1], name="x")
    y = tf.get_variable("y", [1], tf.float32)
    z =  tf.placeholder(tf.float32, shape=[1], name="x")
    print(x==z)
    print(x)
    print(z)
    li = tf.all_variables()
    #doesn't include x!
    print(li)
    w = tf.get_variable("x", [1], tf.float32)
    li = tf.all_variables()
    print(li)
"""

f = compile_net(Net(
    [Scope("fc"),
       Placeholder("a",dtype=tf.float32),
       Placeholder("b",dtype=tf.float32),
       InitVar("W",[7*7*64,1024]),
       InitVar("b",[1024]),
       Placeholder("x",dtype=tf.float32),
       Placeholder("y",dtype=tf.float32),
       Placeholder("keep_prob",dtype=tf.float32)]))

li = [Scope("fc"),
       Placeholder("a",dtype=tf.float32),
       Placeholder("b",dtype=tf.float32)]

with tf.Graph().as_default() as g:
    """
    with tf.variable_scope("fc"):
        x = tf.placeholder(tf.float32, [1], "keep_prob")
        print(x)
    """
    """
    x = tf.placeholder(tf.float32, shape=[1], name="x")
    #x = tf.placeholder(tf.float32, shape=[1], name="x")
    #print(g.get_tensor_by_name("x"))
    y = f(x)
    """
    #print(g.get_tensor_by_name("x:0"))
    #print(g.get_tensor_by_name("fc/keep_prob:0"))
    #print(g.get_tensor_by_name("fc/keep_prob/cpu:0"))
    """
    with tf.variable_scope("fc"):
        with tf.device('/cpu:0'):
            x1 = tf.get_variable("x1", shape=[1])
        print(x1)
        with tf.device('/cpu:0'):
            x2 = tf.placeholder(tf.float32, [1], "x2")
        print(x2)
        with tf.device('/cpu:0'):
            x3 = tf.placeholder(tf.float32, [1], "x3")
        print(x3)
#        with tf.device('/cpu:0'):
#        y = variable_on_cpu("blah", shape=[1], var_type="placeholder")
#        print(y)
    """

    for name in ["a","b"]:
        with tf.variable_scope("fc"):
            x = tf.placeholder(tf.float32, shape=[1], name=name)
            print(x)

    sc = ""
    x = tf.placeholder(tf.float32, shape=[1], name="x")
    d={}

    for entry in li:
        if isinstance(entry, Scope):
            sc = entry.scope
        elif isinstance(entry, Var):
            (name, shape, initializer, scope1, dtype, var_type) = entry.get()
            scope2 = concat_scopes(sc, scope1)
            with tf.variable_scope(scope2):
                n=concat_scopes(scope2, name)
                print(scope2)
                print(name)
#                d[n] = 
                x= tf.placeholder(dtype, shape=shape, name=name)
                print(x)
                #                    d[n] = variable_on_cpu(name, shape, initializer=initializer, dtype=dtype, var_type=var_type)
#                print(d[n])
                    #print(tf.get_default_graph().get_tensor_by_name(n+":0"))
