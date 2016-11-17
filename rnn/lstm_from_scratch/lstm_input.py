from bidict import bidict
from nlp_process import *
from utils import *
from nets import *

"""
Convert from text to something trainable!
"""

#https://bidict.readthedocs.io/en/master/basic-usage.html
def make_bidict(chars):
    return bidict(enumerate(chars))

def make_data(bid, l, doc, mode="last"):
    xs = []
    ys = []
    for i in range(len(doc)-l):
        xs.append([bid.inv[doc[j]] for j in range(i, i+l)])
        if mode=="last":
            #print(doc[i+l])
            ys.append(bid.inv[doc[i+l]])
        else:
            ys.append([bid.inv[doc[j]] for j in range(i+1, i+l+1)])
    return (xs, ys)

def one_hot_data(l,t):
    f = lambda i: e_(i,l)
    if isinstance(t, tuple):
        (x,y) = t
        return (deepmap(f,x),deepmap(f,y))
    else:
        return deepmap(f,t)

if __name__ == "__main__":
    d = make_bidict(good_chars)
    print(d)
    l = len(d)
    (xs, ys) = make_data(d, 5, "hello world")
    print(xs,ys)
    xs = deepmap(lambda i: e_(i, l), xs)
    ys = deepmap(lambda i: e_(i, l), ys)
    print(xs,ys)
    bf = make_batch_feeder(xs, ys)
    print(bf.next_batch(2))
    print(bf.next_batch(2))

#make the feed
