#!/bin/bash

# Use GPU:
# May need flags
#export THEANO_FLAGS="mode=FAST_RUN,device=cuda0,floatX=float32,gpuarray.preallocate=0.9"
#export CPATH=$CPATH:/home/2136420/theanoenv/include
#export LIBRARY_PATH=$LIBRARY_PATH:/home/2136420/theanoenv/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/2136420/theanoenv/lib

export CUDA_VISIBLE_DEVICES=0

#####################
# Running the network
#####################

# To run network call this file:
# ./runGAN.sh
# eg:
# ./runCNN.sh 8 metricmass metricmass

# train or generate?
mode=train

# directory for .txt waveform files are stored
#data_file=/home/hunter.gabbard/Burst/rricard_gan/data/data_1000_samples.pkl

# For long term, training data should be in the form of Gaussian noise.
data_file=/home/hunter.gabbard/Burst/GenNet/train_on_wvf_version/data/sineGuass_set/sineGauss100b10tau0d1/data.pkl

# directory for output to be stored
outdir=/home/hunter.gabbard/public_html/Burst/sine-gaussian_runs/single_waveform_training

# number of noise and GW waveform samples to train on
n_samples=1
n_epochs=100
g_lr=(0.5e-1)
d_lr=(1e-6)

# run script
./nn.py --datafile=$data_file --n-samples=$n_samples --outdir=$outdir \
--n_epochs=$n_epochs --g_lr=$g_lr --d_lr=$d_lr
