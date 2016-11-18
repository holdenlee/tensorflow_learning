import tensorflow as tf
from functools import *
import itertools
import inspect
from datetime import datetime
import re
import os.path
import time
import numpy as np
from six.moves import xrange
from utils import *
from nets import *

def linear_layer_with_dims(m,n,scope=None,name=None):
    li = [InitVar("W", [m,n]),
          InitVar("b", [n]),
          Apply(linear_layer, {"name":name})]
    if scope==None:
        return Net(li)
        #return li
    else:
        return Net([Scope(scope, li)])
        #return Scope(scope,li)

#TODO: abstract this
def relu_layer_with_dims(m,n,scope=None,name=None):
    li = [InitVar("W", [m,n]),
          InitVar("b", [n]),
          Apply(relu_layer, {"name":name})]
    if scope==None:
        return Net(li)
        # return li
    else:
        return Net([Scope(scope, li)])
        #return Scope(scope, li)

class Net:
    def __init__(self, li):
        self.inputs = [1]
        self.outputs = [1]
        self.d = {1:(li, [])}

class Scope:
    def __init__(self, s, li):
        self.scope = s
        self.li = li

class Apply:
    def __init__(self, f, args={}):
        self.f = f
        self.args = args

class Var:
    def __init__(self, name, shape=None, initializer=None, scope="", dtype=tf.float32, var_type="variable"):
        self.name=name
        self.shape=shape
        self.initializer=initializer
        self.scope=scope
        self.dtype=tf.float32
        self.var_type=var_type
    def get(self):
        return (self.name,self.shape,self.initializer,self.scope, self.dtype, self.var_type)
"""
class InitVar(Var):
    def __init__(self, name, shape=None, initializer=None, scope="", dtype=tf.float32):
        super(InitVar, self).__init__(name, shape, initializer, scope, dtype)
        self.var_type="var"

class Placeholder(Var):
    def __init__(self, name, shape=None, initializer=None, scope="", dtype=tf.float32):
        super(Placeholder, self).__init__(name, shape, initializer, scope, dtype)
        self.var_type="placeholder"
"""
def InitVar(name, shape=None, initializer=None, scope="", dtype=tf.float32):
    return Var(name, shape, initializer, scope, dtype, "variable")

def Placeholder(name, shape=None, scope="", dtype=tf.float32):
    return Var(name, shape, None, scope, dtype, "placeholder")

#don't use this
class InitVars:
    def __init__(self, li):
        self.li = li

def concat_scopes(s1, s2):
    if s2!="":
        if s1!="":
            return s1 + "/" + s2
        else: 
            return s2
    else:
        return s1

#d is actually unnecessary because you can get tensors by name.
def _compile_list(li, d, scope):
    def f(x,d):
        sc = scope
        for entry in li:
            if isinstance(entry, Scope):
                with tf.variable_scope(entry.scope):
                    g = _compile_list(entry.li, d, concat_scopes(sc, entry.scope))
                    x, d = g(x)
            elif isinstance(entry, Var):
                (name, shape, initializer, scope1, dtype, var_type) = entry.get()
                scope2 = concat_scopes(sc, scope1)
                print(sc)
                print(scope1)
                if scope1 == "":
                    n = concat_scopes(scope2, name)
                    #print(n, shape)
                    if var_type=="variable":
                        d[n] = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
                    else:
                        d[n] = tf.placeholder(dtype, shape=shape, name=name)
                else:
                    with tf.variable_scope(scope1): #scope1 or scope2
                        n = concat_scopes(scope2, name)
                        #print(n, shape)
                        if var_type=="variable":
                            d[n] = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
                        else:
                            d[n] = tf.placeholder(dtype, shape=shape, name=name)
#                    d[n] = variable_on_cpu(name, shape, initializer=initializer, dtype=dtype, var_type=var_type)
                    #print(d[n])
                    #print(tf.get_default_graph().get_tensor_by_name(n+":0"))
            elif isinstance(entry, InitVars):
                for (name, shape, initializer, scope1) in entry.li:
                    scope2 = concat_scopes(sc, scope1)
                    #same as above.
                    with tf.variable_scope(scope1):
                        n = concat_scopes(scope2, name)
                        d[n] = variable_on_cpu(name, shape, initializer=initializer)
            elif isinstance(entry, Apply):
                f = entry.f
                args = entry.args
                #print(scope)
                #print(type(scope))
                x = eval_with_missing_args(f, x, args, d)
                #tf.get_variable_scope().reuse_variables()
                #f(x,**args)
            elif isinstance(entry, Net):
                g = compile_net_with_vars(entry, d, sc)
                x, d = g(x)
        return x,d
    return (lambda x : f(x,d))

def compile_net_with_vars(net, d={}, scope=""):
    #fix this
    return _compile_list(net.d[1][0], d, scope)

def compile_net(net, d={},scope=""):
    return (lambda x: compile_net_with_vars(net, d, scope)(x)[0])
