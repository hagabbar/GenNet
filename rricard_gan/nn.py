#!/usr/bin/env python
"""
This code will generate sinusoid curves using a generative adversarial
neural network.

Author: Hunter Gabbard <h.gabbard.1@research.gla.ac.uk>

"""

import os,sys,glob
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
from keras.models import Model
from keras.layers import Input, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
import argparse
import pickle

#Configure tensorflow to use gpu memory as needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def get_args():
    parser = argparse.ArgumentParser(prog='nn.py', description='Generative Adversarial Neural Network in keras with tensorflow')
    
    # arguments
    parser.add_argument('--datafile', type=str,
                        help='data file. Please provide FULL path.')
    parser.add_argument('--n-samples', type=int,
                        help='number of waveforms and noise samples to train over (e.g. 10000 would mean 10000 noise and 10000 waveforms signals, 20000 in total.')
    parser.add_argument('--outdir', type=str,
                        help='Location for output to be stored. Please provide FULL path.')


    # network arguments
    parser.add_argument('-NE', '--n_epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--g_lr', type=float, default=0.425e-1,
                        help='generator learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-6,
                        help='discriminator learning rate')

    return parser.parse_args()

def sample_data(n_samples=10000, x_vals=np.arange(0, 5, .1), max_offset=100, mul_range=[1, 2]):
    """
    Function to make simple sinusoid waveforms that are 50 elements long.
    Is not essential, but can be used for toy problems in future.
    """
    vectors = []
    for i in range(n_samples):
        offset = np.random.random() * max_offset
        mul = mul_range[0] + np.random.random() * (mul_range[1] - mul_range[0])
        vectors.append(
            np.sin(offset + x_vals * mul) / 2 + .5
        )
    return np.array(vectors)

def get_generative(G_in, dense_dim=300, out_dim=8192, lr=0.425e-1):
    x = Dense(dense_dim)(G_in)
    x = Activation('relu')(x)
    x = Dense(150)(x)
    x = Activation('relu')(x)
    G_out = Dense(out_dim, activation='tanh')(x)
    G = Model(G_in, G_out)
    opt = SGD(lr=lr)
    G.compile(loss='binary_crossentropy', optimizer=opt)
    return G, G_out

def get_discriminative(D_in, lr=1e-6, drate=.25, n_channels=25, conv_sz=5, leak=.2):
    x = Reshape((-1, 1))(D_in)
    x = Conv1D(n_channels, conv_sz, activation='relu')(x)
    x = Dropout(drate)(x)
    x = Flatten()(x)
    x = Dense(n_channels)(x)
    D_out = Dense(2, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D, D_out

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def make_gan(GAN_in, G, D):
    set_trainability(D, False)
    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    return GAN, GAN_out

def sample_data_and_gen(G, x_train, noise_dim=10, n_samples=10000):
    #XT = sample_data(n_samples=n_samples)
    XT = x_train[:n_samples]
    XN_noise = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    XN = G.predict(XN_noise)
    X = np.concatenate((XT, XN))
    y = np.zeros((2*n_samples, 2))
    y[:n_samples, 1] = 1
    y[n_samples:, 0] = 1
    return X, y

def pretrain(G, D, x_train, noise_dim=10, n_samples=10000, batch_size=32):
    X, y = sample_data_and_gen(G, x_train, n_samples=n_samples, noise_dim=noise_dim)
    set_trainability(D, True)
    D.fit(X, y, epochs=1, batch_size=batch_size)

def sample_noise(G, noise_dim=10, n_samples=10000):
    X = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    y = np.zeros((n_samples, 2))
    y[:, 1] = 1
    return X, y

def train(GAN, G, D, x_train, epochs=1000, n_samples=10000, noise_dim=10, batch_size=32, verbose=False, v_freq=1):
    d_loss = []
    g_loss = []
    e_range = range(epochs)
    if verbose:
        e_range = tqdm(e_range)
    for epoch in e_range:
        X, y = sample_data_and_gen(G, x_train, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, True)
        d_loss.append(D.train_on_batch(X, y))
        
        X, y = sample_noise(G, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch(X, y))
        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
    return d_loss, g_loss

def load_data(path,n_samples):
    """
    Truncates data file into a numpy array called data.
    """
    print('Using data for: {0}'.format(path))

    #load in dataset 0
    with open(path, 'rb') as rfp:
        data = pickle.load(rfp)

    data = data[:n_samples]

    return data 

def main():
    args = get_args()

    x_train = load_data(args.datafile,args.n_samples)

    G_in = Input(shape=[10])
    G, G_out = get_generative(G_in, lr=args.g_lr, out_dim=x_train.shape[1])
    G.summary()

    D_in = Input(shape=[x_train.shape[1]])
    D, D_out = get_discriminative(D_in, lr=args.d_lr)
    D.summary()

    GAN_in = Input([10])
    GAN, GAN_out = make_gan(GAN_in, G, D)
    GAN.summary()

    pretrain(G, D, x_train, n_samples=args.n_samples)

    d_loss, g_loss = train(GAN, G, D, x_train, epochs=args.n_epochs, n_samples=args.n_samples, verbose=True)

    # create directory if it does not exist
    if not os.path.exists('{0}'.format(args.outdir)):
        os.makedirs('{0}'.format(args.outdir))

    Nrun = 0
    while os.path.exists('{0}/run{1}'.format(args.outdir,Nrun)):
        Nrun += 1
    os.makedirs('{0}/run{1}'.format(args.outdir, Nrun))

    # plot an example of the first waveform in the trianing
    # set.
    plt.plot(x_train[0])
    plt.savefig('{0}/run{1}/original_wvf_ex.png'.format(args.outdir,Nrun))
    plt.close()

    # plot loss curves
    ax = pd.DataFrame(
        {
            'Generative Loss': g_loss,
            'Discriminative Loss': d_loss,
        }
    ).plot(title='Training loss', logy=True)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.savefig('{0}/run{1}/loss.png'.format(args.outdir,Nrun))
    plt.close()


    # plot results
    N_VIEWED_SAMPLES = 1
    data_and_gen, _ = sample_data_and_gen(G, x_train, n_samples=N_VIEWED_SAMPLES)
    plt.plot(pd.DataFrame(np.transpose(data_and_gen[N_VIEWED_SAMPLES:])))
    plt.savefig('{0}/run{1}/raw_gen_waveforms.png'.format(args.outdir,Nrun))
    plt.close()

    N_VIEWED_SAMPLES = 1
    data_and_gen, _ = sample_data_and_gen(G, x_train, n_samples=N_VIEWED_SAMPLES)
    plt.plot(pd.DataFrame(np.transpose(data_and_gen[N_VIEWED_SAMPLES:])).rolling(5).mean()[5:])
    plt.savefig('{0}/run{1}/smoothed_gen_waveforms.png'.format(args.outdir,Nrun))
    plt.close()

if __name__ == "__main__":
    main()
