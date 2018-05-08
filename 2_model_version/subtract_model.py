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
import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Reshape, Conv2DTranspose, GaussianDropout, Activation
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam, SGD, RMSprop
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
hyperparams.noise_dim = 1
hyperparams.noise_samples = int(hyperparams.n_total*0.5)
hyperparams.batch_size = 16
hyperparams.epochs = 500
hyperparams.g_lr = 1e-4 #4e-3
hyperparams.d_lr = 1e-4#1e-2
hyperparams.loss = 'binary_crossentropy'
hyperparams.snr = 5
hyperparams.outdim = 50


def sample_data(n_samples=10000, x_vals=np.arange(0, 5, .1), max_offset=2*np.pi, mul_range=[1, 2], snr=hyperparams.snr):
    vectors = []
    for i in range(n_samples):
        offset = np.random.random() * max_offset
        #mul = mul_range[0] + np.random.random() * (mul_range[1] - mul_range[0])
        mul = (2 * np.pi) / 5
        vectors.append(
            np.sin(offset + x_vals * mul) * snr
        )
    return np.array(vectors)


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

def sample_data_and_gen(G, xt_train, encoder, noise_dim=10, n_samples=10000, noise_samples=100):
    # produce latent variables
    #latent_noise = np.random.normal(0, 1, size=[noise_samples, 1, 50])
    #XN_noise = encoder.predict(latent_noise)

    XT = np.random.normal(0, 1, size=[n_samples, hyperparams.outdim])
    XN_noise = np.random.normal(0, 1, size=[noise_samples, 1, noise_dim])
    XN = G.predict(XN_noise)
    for s in range(noise_samples):
        #XN[s] = XN[s] / np.max(XN[s])
        #XN[s] =  np.subtract(XN[s], xt_train[0])
        XN[s] = np.subtract(xt_train[0], XN[s])

    X = np.vstack((XT, XN))
    y = np.ones((n_samples+len(XN_noise), 2))
    y[:n_samples, 1] = 1
    y[n_samples:, 0] = 1
    return X, y

def pretrain(G, D, xt_train, encoder, noise_dim=10, n_samples=10000, noise_samples=10000, batch_size=32):
    X, y = sample_data_and_gen(G, xt_train, encoder, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)
    set_trainability(D, True)


    D.fit(X, y, epochs=1, batch_size=batch_size)


def sample_noise(G, xt_train, encoder, noise_dim=10, n_samples=10000):
    # produce latent variables
    #latent_noise = np.random.normal(0, 1, size=[n_samples, 1, 50])
    #X = encoder.predict(latent_noise)

    X = np.random.normal(0, 1, size=[n_samples, 1, noise_dim])
    y = np.ones((n_samples, 2))
    y[:, 1] = 1
    return X, y

def train(GAN, G, D, xt_train, encoder, epochs=500, n_samples=10000, noise_samples=hyperparams.noise_samples, noise_dim=10, batch_size=32, verbose=False, v_freq=1):
    d_loss = []
    g_loss = []
    e_range = range(epochs)
    if verbose:
        e_range = tqdm(e_range)
    for epoch in e_range:

        # use experience replay
        """
        if epoch == 0:
            X, y = sample_data_and_gen(G, xt_train, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)
            X_past, y_past = X, y
        elif epoch%50 == 0 and epoch > 0:
            X, y = sample_data_and_gen(G, xt_train, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)
            X_past, y_past = X, y
            X = np.vstack((X[:int(len(X)/2),:],X_past[int(len(X_past)*(3/4)):,:],X[int(len(X)*(3/4)):,:]))
            y = np.vstack((y[:int(len(y)/2),:],y_past[int(len(y_past)*(3/4)):,:],y[int(len(y)*(3/4)):,:]))
        else:
            X, y = sample_data_and_gen(G, xt_train, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)
        """

        X, y = sample_data_and_gen(G, xt_train, encoder, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)

        """
        if epoch%5 == 0 or epoch == 0:
            # train networks
            set_trainability(D, True)
            d_loss.append(D.train_on_batch(X, y))
        else:
            d_loss.append(d_loss[-1])
        """

        # train networks
        set_trainability(D, True)
        d_loss.append(D.train_on_batch(X, y))

        X, y = sample_noise(G, xt_train, encoder, n_samples=noise_samples, noise_dim=noise_dim)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch(X, y))

        """
        if epoch%25 == 0 or epoch == 0:
            # train networks
            set_trainability(D, False)
            g_loss.append(GAN.train_on_batch(X, y))
        else:
            g_loss.append(g_loss[-1])
        """

        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
    return d_loss, g_loss

def test_data_and_gen(G, xt_train, encoder, noise_dim=10, n_samples=10000, noise_samples=100):
    # produce latent variables
    #latent_noise = np.random.normal(0, 1, size=[noise_samples, 1, 50])
    #XN_noise = encoder.predict(latent_noise)

    XT = np.random.normal(0, 1, size=[n_samples, hyperparams.outdim])
    XN_noise = np.random.normal(0, 1, size=[noise_samples, 1, noise_dim])

    XN = G.predict(XN_noise)
    residuals = []
    for s in range(noise_samples):
        #XN[s] = XN[s] / np.max(XN[s])
        #XN[s] = XN[s][::-1]
        #residuals.append(XN[s] - xt_train)
        residuals.append(xt_train - XN[s])
        #residuals[s] = residuals[s] / np.max(residuals[s])
    residuals = np.array(residuals)



    X = np.vstack((XT, XN))
    y = np.zeros((n_samples+len(XN_noise), 2))
    y[:n_samples, 1] = 1
    y[n_samples:, 0] = 1
    return X, residuals

def get_generative(G_in, dense_dim=128, drate=0.1, out_dim=50, lr=1e-3):
    # original network
    """
    x = Dense(512, activation='relu')(G_in)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    G_out = Dense(out_dim, activation='tanh')(x)
    G = Model(G_in, G_out)
    opt = SGD(lr=lr)
    G.compile(loss='binary_crossentropy', optimizer=opt)
    """

    # transpose convolutional network
    act = 'elu'

    x = Reshape((-1,1,1))(G_in)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128,(1,4), activity_regularizer=regularizers.l1(0.001), kernel_regularizer=regularizers.l2(0.01), strides=(1,1),padding='valid',activation=act)(x)
    x = BatchNormalization()(x)
    #x = Dropout(drate)(x)

    x = Conv2DTranspose(64,(1,8),strides=(1,1),padding='valid',activation=act)(x)
    x = BatchNormalization()(x)
    #x = Dropout(drate)(x)

    x = Conv2DTranspose(32,(1,16),strides=(1,1),padding='valid',activation=act)(x)
    x = BatchNormalization()(x)
    #x = Dropout(drate)(x)

    x = Conv2DTranspose(16,(1,32),strides=(1,1),padding='valid',activation=act)(x)
    x = BatchNormalization()(x)
    #x = Dropout(drate)(x)

    x = Flatten()(x)
    x = BatchNormalization()(x)

    x = Dense(out_dim, activation=act)(x)
    x = BatchNormalization()(x)

    G_out = Dense(out_dim, activation='linear')(x)
    #G_out = Conv2DTranspose(1,(1,out_dim))(x)
    G = Model(G_in, G_out)
    #opt = SGD(lr=lr)
    opt = Adam(lr=lr, beta_1=0.5)
    G.compile(loss='binary_crossentropy', optimizer=opt)


    return G, G_out

def get_discriminative(D_in, lr=1e-3, drate=.3, n_channels=50, conv_sz=5, leak=.2):
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
    x = Conv1D(50, 16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = BatchNormalization()(x)
    x = Dropout(drate)(x)
    
    x = Conv1D(128, 8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = BatchNormalization()(x)
    #x = Conv1D(256, 4)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    #x = BatchNormalization()(x)
    #x = Conv1D(512, 4)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    #x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dense(n_channels)(x)
    x = Dropout(drate)(x)

    D_out = Dense(2, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr, beta_1=0.5)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D, D_out

def main():
    outdir = '/home/hunter.gabbard/public_html/Burst/Gauss_pulse_testing/sineGauss_subtract/'

    # print out input waveforms
    ax = pd.DataFrame(np.transpose(sample_data(25))).plot(legend=False)
    ax = ax.get_figure()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    ax.savefig('%sinput_waveforms.png' % outdir)
    plt.close(ax)

    #ht_train = sample_data(hyperparams.n_samples)
    ht_train = sample_data(1)
    xt_train = ht_train + np.random.normal(0, 1, size=[1, ht_train.shape[1]])

    # initialize subplot figure
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)

    ax = pd.DataFrame(np.transpose(ht_train[0]))
    ax1.plot(ax, color='cyan', linewidth=0.5)
    ax1.set_title('signal + (sig+noise)')
    #ax1 = ax1.get_figure()
    #ax.savefig('%sinput_waveforms.png' % outdir)
    #plt.close(ax)

    ax1.plot(xt_train[0], color='green', alpha=0.5, linewidth=0.5)
    #plt.savefig('%sinput_waveforms_plusnoise.png' % outdir)
    #plt.close(ax)

    
    #G = keras.models.load_model('g_model.hdf5')
    G_in = Input(shape=(1,hyperparams.noise_dim))
    G, G_out = get_generative(G_in, lr=hyperparams.g_lr)
    G.load_weights('best_g_weights.hdf5')
    G.summary()

    #D = keras.models.load_model('d_model.hdf5')
    D_in = Input(shape=(hyperparams.outdim,))
    D, D_out = get_discriminative(D_in, lr=hyperparams.d_lr)
    #D.load_weights('best_d_weights.hdf5')
    D.summary()

    GAN_in = Input((1,hyperparams.noise_dim))
    GAN, GAN_out = make_gan(GAN_in, G, D)
    GAN.summary()

    # make auto XN_noise latent variables
    #autoencoder, encoder, decoder = make_autoencoder(hyperparams.noise_dim)
    #auto_xt_train = np.reshape(xt_train, (xt_train.shape[0],1,xt_train.shape[1]))
    #autoencoder = train_autoencoder(autoencoder, auto_xt_train)
    encoder = []

    pretrain(G, D, xt_train, encoder, n_samples=hyperparams.n_samples, noise_samples=hyperparams.noise_samples, noise_dim=hyperparams.noise_dim, batch_size=hyperparams.batch_size)


    d_loss, g_loss = train(GAN, G, D, xt_train, encoder, epochs=hyperparams.epochs, n_samples=hyperparams.n_samples,
                           noise_samples=hyperparams.noise_samples, noise_dim=hyperparams.noise_dim, batch_size=hyperparams.batch_size, verbose=True)

    # plot several generated waveforms, noise-free signal, signal+noise
    N_VIEWED_SAMPLES = 25
    data_and_gen, residuals = test_data_and_gen(G, xt_train, encoder, noise_samples=N_VIEWED_SAMPLES, n_samples=N_VIEWED_SAMPLES, noise_dim=hyperparams.noise_dim)
    sig = pd.DataFrame(np.transpose(ht_train[0])) #.plot(legend=False, color='cyan')
    #for i in range(data_and_gen[N_VIEWED_SAMPLES:].shape[0]):
    #    data_and_gen[N_VIEWED_SAMPLES:][i] = data_and_gen[N_VIEWED_SAMPLES:][i][::-1]

    gen_sig = pd.DataFrame(np.transpose(data_and_gen[N_VIEWED_SAMPLES:])) #.plot(legend=False, color='blue', alpha=0.25, ax=sig)
    ax3.plot(sig, color='cyan', linewidth=0.5)
    ax3.plot(gen_sig, color='blue', alpha=0.25, linewidth=0.5)
    ax3.plot(xt_train[0], color='green', linewidth=0.5)
    ax3.set_title('gen + sig + (sig+noise)')
    #ax = ax.get_figure()
    #plt.savefig('%sgen_waveform.png' % outdir)
    #plt.close(ax)

    # plot all noise training samples
    ax2.plot(np.transpose(data_and_gen[:N_VIEWED_SAMPLES]), alpha=0.25, color='blue', linewidth=0.5)
    ax2.set_title('Noise Samples')

    # plot residuals. signal+noise - generated
    residuals = np.resize(residuals, (residuals.shape[0], residuals.shape[2]))
    ax4.plot(np.transpose(residuals), color='red', alpha=0.25, linewidth=0.5)
    ax4.set_title('Residuals')

    plt.savefig('%swaveform_results.png' % outdir, dpi=900)
    plt.close()


    # plot loss functions
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
    ax.savefig('%sloss.png' % outdir)
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
