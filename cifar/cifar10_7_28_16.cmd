#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin  
#SBATCH --mail-type=end  
#SBATCH --mail-user=holdenl@princeton.edu  

module load python
module load cudatoolkit/7.5
module load cudann
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_train.py --train_dir='/tigress/holdenl/tmp/cifar10_7_28_16'
