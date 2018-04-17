
"""
This is a script to generate sinusoid waveforms using a DCGAN

Author: Hunter Gabbard <h.gabbard.1@research.gla.ac.uk>
"""

import os,sys
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from keras.models import Model
from keras.layers import Input, Reshape, Conv2DTranspose, GaussianDropout
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from scipy import signal

class hyperparams():
    def __init__(self,n_total,n_samples,noise_samples,noise_dim,
                 batch_size,epochs,g_lr,d_lr,loss,outdim,noise_level):
        # set hyperparamters
        self.n_total = n_total
        self.n_samples = n_samples
        self.noise_samples = noise_samples
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.loss = loss
        self.outdim = outdim
        self.noise_level = noise_level

# set hyperparamters
hyperparams.n_total = 500
hyperparams.n_samples = int(hyperparams.n_total*0.5)
hyperparams.noise_samples = int(hyperparams.n_total*0.5)
hyperparams.noise_dim = 1
hyperparams.batch_size = 16
hyperparams.epochs = 20000
hyperparams.g_lr = 1.5e-3 #4e-3
hyperparams.d_lr = 1.5e-6 #4e-6
hyperparams.loss = 'binary_crossentropy'
hyperparams.outdim = 50
hyperparams.noise_level = 0.25


def sample_data(n_samples=10000, x_vals=np.arange(0, 5, .1), max_offset=2*np.pi, mul_range=[1, 2]):
    vectors = []
    for i in range(n_samples):
        offset = np.random.random() * max_offset
        #mul = mul_range[0] + np.random.random() * (mul_range[1] - mul_range[0])
        mul = (2 * np.pi) / 5
        vectors.append(
            np.sin(offset + x_vals * mul)
        )
    return np.array(vectors)


def get_generative(G_in, dense_dim=128, drate=0.1, out_dim=50, lr=1e-3):

    # transpose convolutional network

    x = Reshape((-1,1,1))(G_in)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(1024,(1,4),strides=(1,1),padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = GaussianDropout(drate)(x)
    x = Conv2DTranspose(512,(1,8),strides=(1,1),padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = GaussianDropout(drate)(x)
    x = Conv2DTranspose(256,(1,16),strides=(1,1),padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = GaussianDropout(drate)(x)
    x = Conv2DTranspose(128,(1,32),strides=(1,1),padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = GaussianDropout(drate)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    #x = Dense(out_dim, activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(drate)(x)
    G_out = Dense(out_dim, activation='tanh')(x)
    #G_out = BatchNormalization()(G_out)
    #G_out = Conv2DTranspose(1,(1,out_dim))(x)
    G = Model(G_in, G_out)
    opt = SGD(lr=lr)
    G.compile(loss=hyperparams.loss, optimizer=opt)


    return G, G_out


def get_discriminative(D_in, lr=1e-3, drate=.25, n_channels=50, conv_sz=5, leak=.2):
    # old network

    """
    x = Reshape((-1, 1))(D_in)
    x = Conv1D(n_channels, conv_sz)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(drate)(x)
    x = Flatten()(x)
    x = Dense(n_channels)(x)
    """


    x = Reshape((-1, 1))(D_in)
    x = Conv1D(64, 16, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    #x = GaussianDropout(drate)(x)
    x = Conv1D(128, 8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = GaussianDropout(drate)(x)
    #x = Conv1D(256, 4)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    #x = BatchNormalization()(x)
    #x = Conv1D(512, 4)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    #x = Dense(n_channels)(x)
    #x = BatchNormalization()(x)
    D_out = Dense(2, activation='sigmoid')(x)
    D_out = BatchNormalization()(D_out) # may have to add this in later.
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr, beta_1=0.5)
    D.compile(loss=hyperparams.loss, optimizer=dopt)
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
    GAN.compile(loss=hyperparams.loss, optimizer=G.optimizer)
    return GAN, GAN_out


def sample_data_and_gen(G, xt_train, noise_dim=10, n_samples=10000, noise_samples=100):
    XT = np.random.normal(-hyperparams.noise_level, hyperparams.noise_level, size=[n_samples, hyperparams.outdim])
    XN_noise = np.random.normal(-hyperparams.noise_level, hyperparams.noise_level, size=[noise_samples, 1, noise_dim])

    XN = G.predict(XN_noise)
    #xt_train = np.resize(xt_train, (xt_train.shape[0], xt_train.shape[2]))
    # subtract out generated waveform from signal for each fake gen waveform
    for s in range(noise_samples):
        XN[s] =  xt_train - XN[s]

    X = np.vstack((XT, XN))
    y = np.zeros((n_samples+len(XN_noise), 2))
    y[:n_samples, 1] = 1
    y[n_samples:, 0] = 1
    return X, y

def pretrain(G, D, xt_train, noise_dim=10, n_samples=10000, noise_samples=10000, batch_size=32):
    X, y = sample_data_and_gen(G, xt_train, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)
    set_trainability(D, True)


    D.fit(X, y, epochs=1, batch_size=batch_size)


def sample_noise(G, noise_dim=10, n_samples=10000):
    X = np.random.normal(-hyperparams.noise_level, hyperparams.noise_level, size=[n_samples, 1, noise_dim])
    y = np.zeros((n_samples, 2))
    y[:, 1] = 1
    return X, y

def train(GAN, G, D, xt_train, epochs=500, n_samples=10000, noise_samples=hyperparams.noise_samples, noise_dim=10, batch_size=32, verbose=False, v_freq=1):
    d_loss = []
    g_loss = []
    e_range = range(epochs)
    if verbose:
        e_range = tqdm(e_range)
    for epoch in e_range:

        # use experience replay
        """
        if epoch == 0:
            X, y = sample_data_and_gen(G, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)
            X_past, y_past = X, y
        elif epoch%5 == 0 and epoch > 0:
            X, y = sample_data_and_gen(G, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)
            X_past, y_past = X, y
            X = np.vstack((X[:int(len(X)/2),:],X_past[int(len(X_past)*(3/4)):,:],X[int(len(X)*(3/4)):,:]))
            y = np.vstack((y[:int(len(y)/2),:],y_past[int(len(y_past)*(3/4)):,:],y[int(len(y)*(3/4)):,:]))
        else:
            X, y = sample_data_and_gen(G, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)
        """

        X, y = sample_data_and_gen(G, xt_train, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)

        # train networks
        set_trainability(D, True)
        d_loss.append(D.train_on_batch(X, y))

        X, y = sample_noise(G, n_samples=noise_samples, noise_dim=noise_dim)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch(X, y))
        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
    return d_loss, g_loss

def main():
    outdir = 'output/'

    ht_train = sample_data(1)
    xt_train = ht_train + np.random.normal(-hyperparams.noise_level, hyperparams.noise_level, size=[1, ht_train.shape[1]])

    ax = pd.DataFrame(np.transpose(ht_train)).plot(legend=False)
    ax = ax.get_figure()
    ax.savefig('%sinput_waveforms.png' % outdir, dpi=1200)
    plt.close(ax)

    G_in = Input(shape=(1,hyperparams.noise_dim))
    G, G_out = get_generative(G_in, lr=hyperparams.g_lr, out_dim=hyperparams.outdim)
    G.summary()

    D_in = Input(shape=(hyperparams.outdim,))
    D, D_out = get_discriminative(D_in, lr=hyperparams.d_lr)
    D.summary()

    GAN_in = Input((1,hyperparams.noise_dim))
    GAN, GAN_out = make_gan(GAN_in, G, D)
    GAN.summary()

    pretrain(G, D, xt_train, n_samples=hyperparams.n_samples, noise_samples=hyperparams.noise_samples, noise_dim=hyperparams.noise_dim, batch_size=hyperparams.batch_size)


    d_loss, g_loss = train(GAN, G, D, xt_train, epochs=hyperparams.epochs, n_samples=hyperparams.n_samples,
                           noise_samples=hyperparams.noise_samples, noise_dim=hyperparams.noise_dim, batch_size=hyperparams.batch_size, verbose=True)


    ax = pd.DataFrame(
        {
            'Generative Loss': g_loss,
            'Discriminative Loss': d_loss,
        }
    ).plot(title='Training loss', logy=True)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax = ax.get_figure()
    ax.savefig('%sloss.png' % outdir, dpi=1200)
    plt.close(ax)


    N_VIEWED_SAMPLES = 25
    data_and_gen, _ = sample_data_and_gen(G, xt_train, noise_samples=N_VIEWED_SAMPLES, n_samples=N_VIEWED_SAMPLES, noise_dim=hyperparams.noise_dim)
    ax = pd.DataFrame(np.transpose(data_and_gen[N_VIEWED_SAMPLES:]))[5:].plot(legend=False)
    ax = ax.get_figure()
    ax.savefig('%sgen_waveform.png' % outdir, dpi=1200)
    plt.close(ax)

    """
    # Check whether output distribution is similar to input training set
    # get two distributions
    ai_dist = data_and_gen[N_VIEWED_SAMPLES:]
    sample_orig_dist = sample_data(hyperparams.n_samples)

    samp_angle = []
    ai_angle = []
    # calculate phi for both distributions
    for idx in range(ai_dist.shape[0]):
        ai_angle.append(np.arcsin(ai_dist[idx][0]))
        samp_angle.append(np.arcsin(sample_orig_dist[idx][0]))

    # make histogram of two distributions
    plt.hist(ai_angle, bins=25)
    plt.title("Generative network phi histogram")
    plt.xlabel("Value")
    plt.savefig('/home/hunter.gabbard/public_html/Burst/Gauss_pulse_testing/sineGauss_subtract/ai_phi_hist.png')
    plt.close()

    plt.hist(samp_angle, bins=25)
    plt.title("Orig training set phi histogram")
    plt.xlabel("Value")
    plt.savefig('/home/hunter.gabbard/public_html/Burst/Gauss_pulse_testing/sineGauss_subtract/orig_phi_hist.png')
    plt.close()
    """

if __name__ == '__main__':
    main()
