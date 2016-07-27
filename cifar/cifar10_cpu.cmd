#!/bin/bash

module load python
module load cudatoolkit/7.5
module load cudann
THEANO_FLAGS=mode=FAST_RUN,floatX=float32 python cifar10_train_f.py --train_dir='/tigress/holdenl/tmp/cifar10_f_train1'
