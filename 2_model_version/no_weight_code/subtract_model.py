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
from keras.layers import Input, Reshape, Conv2DTranspose, GaussianDropout, Activation, GaussianNoise, GlobalAveragePooling2D, GlobalAveragePooling1D, GlobalMaxPooling2D, GlobalMaxPooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D, UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from scipy import signal

class hyperparams():
    def __init__(self,n_total,n_samples,noise_samples,noise_dim,
                 batch_size,epochs,g_lr,d_lr,loss,outdim,noise_level,
                 outdir):
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
        self.outdir = outdir

# set hyperparamters
hyperparams.n_total = 100
hyperparams.n_samples = int(hyperparams.n_total*0.5)
hyperparams.noise_dim = 10
hyperparams.noise_samples = int(hyperparams.n_total*0.5)
hyperparams.batch_size = 8
hyperparams.epochs = 1000
hyperparams.g_lr = 1e-4 #4e-3
hyperparams.d_lr = 1e-4#1e-2
hyperparams.loss = 'binary_crossentropy'
hyperparams.snr = 5
hyperparams.outdim = 50
hyperparams.outdir = '/home/hunter.gabbard/public_html/Burst/Gauss_pulse_testing/sineGauss_subtract/'

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
    GAN.compile(loss=hyperparams.loss, optimizer=G.optimizer,
                metrics=['accuracy'])
    return GAN, GAN_out

def sample_data_and_gen(G, xt_train, encoder, epoch, noise_dim=10, n_samples=10000, noise_samples=100):
    # produce latent variables
    #latent_noise = np.random.normal(0, 1, size=[noise_samples, 1, 50])
    #XN_noise = encoder.predict(latent_noise)

    XT = np.random.normal(0, hyperparams.snr, size=[int(hyperparams.batch_size/2), hyperparams.outdim])
    XN_noise = np.random.normal(0, 1, size=[int(hyperparams.batch_size/2), 1, noise_dim])
    XN = G.predict(XN_noise)

    # plot 1 generated waveform for each epoch
    #plt.plot(XN[0])
    #plt.savefig('{}waveforms/wvf_{}.png'.format(hyperparams.outdir,epoch))
    #plt.close()

    for s in range(int(hyperparams.batch_size/2)):
        #XN[s] = XN[s] / np.max(XN[s])
        XN[s] =  np.subtract(XN[s], xt_train[0])
        #XN[s] = np.subtract(xt_train[0], XN[s])

    X = np.vstack((XT, XN))
    y = np.zeros((hyperparams.batch_size, 2))

    # set true labels
    y[:int(hyperparams.batch_size/2), 0] = np.random.uniform(0.7,1)
    y[int(hyperparams.batch_size/2):, 1] = np.random.uniform(0.7,1)

    # set false labels
    y[:int(hyperparams.batch_size/2), 1] = np.random.uniform(0,0.3)
    y[int(hyperparams.batch_size/2):, 0] = np.random.uniform(0,0.3)

    # randomly change labels
    

    return X, y

def pretrain(G, D, xt_train, encoder, noise_dim=10, n_samples=10000, noise_samples=10000, batch_size=32):
    epoch = 1
    X, y = sample_data_and_gen(G, xt_train, encoder, epoch, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)
    set_trainability(D, True)


    D.fit(X, y, epochs=1, batch_size=batch_size)


def sample_noise(G, xt_train, encoder, noise_dim=10, n_samples=10000):
    # produce latent variables
    #latent_noise = np.random.normal(0, 1, size=[n_samples, 1, 50])
    #X = encoder.predict(latent_noise)

    X = np.random.normal(0, 1, size=[hyperparams.batch_size, 1, noise_dim])
    y = np.zeros((hyperparams.batch_size, 2))
    y[:, 0] = 1 #np.random.uniform(0.8,1)
    y[:, 1] = 0 #np.random.uniform(0,0.2)
    return X, y

def train(GAN, G, D, xt_train, encoder, epochs=500, n_samples=10000, noise_samples=hyperparams.noise_samples, noise_dim=10, batch_size=32, verbose=False, v_freq=1):
    d_loss = []
    d_acc = []
    g_loss = []
    g_acc = []
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

        X, y = sample_data_and_gen(G, xt_train, encoder, epoch, n_samples=n_samples, noise_samples=noise_samples, noise_dim=noise_dim)

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
        d_loss.append(D.train_on_batch(X, y)[0])
        d_acc.append(D.train_on_batch(X, y)[1])

        X, y = sample_noise(G, xt_train, encoder, n_samples=noise_samples, noise_dim=noise_dim)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch(X, y)[0])
        g_acc.append(GAN.train_on_batch(X, y)[1])

        """
        if epoch%25 == 0 or epoch == 0:
            # train networks
            set_trainability(D, False)
            g_loss.append(GAN.train_on_batch(X, y))
        else:
            g_loss.append(g_loss[-1])
        """

        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Acc: {} Discriminative Loss: {}, Acc: {}".format(epoch + 1, g_loss[-1], g_acc[-1], d_loss[-1], d_acc[-1]))
    plot_model(GAN, show_shapes=True, to_file='{}model.png'.format(hyperparams.outdir))
    return d_loss, g_loss, d_acc, g_acc

def test_data_and_gen(G, xt_train, encoder, noise_dim=10, n_samples=10000, noise_samples=100):
    # produce latent variables
    #latent_noise = np.random.normal(0, 1, size=[noise_samples, 1, 50])
    #XN_noise = encoder.predict(latent_noise)

    XT = np.random.normal(0, hyperparams.snr, size=[n_samples, hyperparams.outdim])
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

def get_generative(G_in, dense_dim=128, drate=0.5, out_dim=50, lr=1e-3):
    """
    # original network
    x = Reshape((-1,1,1))(G_in)
    x = BatchNormalization()(x) 
    x = Conv2DTranspose(128,(1,4),strides=(1,1),padding='valid',activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dropout(drate)(x)
    x = Conv2DTranspose(64,(1,8),strides=(1,1),padding='valid',activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dropout(drate)(x)
    x = Conv2DTranspose(32,(1,16),strides=(1,1),padding='valid',activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dropout(drate)(x)
    x = Conv2DTranspose(16,(1,32),strides=(1,1),padding='valid',activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dropout(drate)(x)
    #x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(out_dim, activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dropout(drate)(x)
    G_out = Dense(out_dim, activation='linear')(x)
    #G_out = Conv2DTranspose(1,(1,out_dim))(x)
    G = Model(G_in, G_out)
    opt = SGD(lr=lr)
    G.compile(loss='binary_crossentropy', optimizer=opt)
    """
    
    
    # transpose convolutional network
    act = 'tanh'
    padding = 'same'

    x = Reshape((-1,1,1))(G_in)
    x = BatchNormalization(axis=1)(x)
    #x = Conv2DTranspose(128,(1,4), activity_regularizer=regularizers.l1(0.01), kernel_regularizer=regularizers.l2(0.01), strides=(1,1),padding='valid',activation=act)(x)
    #x = Conv2DTranspose(64,(1,128), strides=(1,2), dilation_rate=(1,1), padding='valid',activation=act)(x)
    #x = GaussianNoise(1)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(drate)(x)
    """
    x = Conv2DTranspose(512,(1,1),strides=(1,2), dilation_rate=(1,1), padding=padding,activation=act)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = GaussianNoise(1)(x)
    x = BatchNormalization(axis=1)(x)
    #x = Dropout(drate)(x)
    
    x = Conv2DTranspose(256,(1,1),strides=(1,1), dilation_rate=(1,1), padding=padding,activation=act)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = GaussianNoise(1)(x)
    x = BatchNormalization(axis=1)(x)
    #x = Conv2D(128, (1,32), activation=act)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(drate)(x)
    """
    x = Conv2DTranspose(128,(1,3),strides=(1,1), dilation_rate=(1,2), padding=padding,activation=act)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = GaussianNoise(1)(x)
    x = BatchNormalization(axis=1)(x)
    #x = Dropout(drate)(x)
    
    x = Conv2DTranspose(64,(1,2),strides=(1,1),dilation_rate=(1,3), padding=padding,activation=act)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=1)(x)
    
    #x = GlobalAveragePooling2D()(x)
    x = GlobalAveragePooling2D()(x)
    #x = Flatten()(x)
    #x = BatchNormalization(axis=1)(x)

    #x = Dense(256, activation='tanh')(x)
    #x = GaussianNoise(1)(x)
    #x = BatchNormalization()(x)

    G_out = Dense(out_dim, activation='linear')(x)
    #G_out = LeakyReLU(alpha=0.2)(G_out)
    #G_out = Conv2DTranspose(1,(1,out_dim))(x)
    G = Model(G_in, G_out)
    #opt = SGD(lr=lr)
    opt = Adam(lr=lr, beta_1=0.5, decay=1e-4)
    G.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy'])
    

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
    padding='valid'
    act='tanh'
    strides=1
    gauss_noise = 1.6


    x = Reshape((-1, 1))(D_in)
    #x = Conv1D(128, 8, activity_regularizer=regularizers.l1(0.0001), kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv1D(128, 8, padding=padding, strides=strides, activation=act)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = GaussianNoise(gauss_noise)(x)
    x = BatchNormalization(axis=1)(x)
    #x = Dropout(drate)(x)
    
    x = Conv1D(256, 8, padding=padding, strides=strides, activation=act)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = Dropout(drate)(x)
    x = GaussianNoise(gauss_noise)(x)
    x = BatchNormalization(axis=1)(x)
    
    x = Conv1D(512, 8, padding=padding, strides=strides, activation=act)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = Dropout(drate)(x)
    x = GaussianNoise(gauss_noise)(x)
    x = BatchNormalization(axis=1)(x)
    """
    x = Conv1D(1024, 8, padding=padding, strides=strides, activation=act)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = Dropout(drate)(x)
    x = GaussianNoise(gauss_noise)(x)
    x = BatchNormalization(axis=1)(x)

    x = Conv1D(2048, 8, padding=padding, strides=strides, activation=act)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = Dropout(drate)(x)
    x = GaussianNoise(gauss_noise)(x)
    x = BatchNormalization(axis=1)(x)
    """
    #x = Conv1D(2048, 4)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    
    #x = Flatten()(x)
    x = GlobalAveragePooling1D()(x)
    #x = BatchNormalization(axis=1)(x)

    #x = Dense(n_channels)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    #x = BatchNormalization(momentum=0.99)(x)
    #x = Dropout(drate)(x)

    D_out = Dense(2, activation='sigmoid')(x)
    #D_out = BatchNormalization(axis=1)(D_out) 
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr, beta_1=0.5, decay=1e-4)
    D.compile(loss='binary_crossentropy', optimizer=dopt,
              metrics=['accuracy'])
    return D, D_out

def main():
    # print out input waveforms
    ax = pd.DataFrame(np.transpose(sample_data(25))).plot(legend=False)
    ax = ax.get_figure()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    ax.savefig('%sinput_waveforms.png' % hyperparams.outdir)
    plt.close(ax)

    #ht_train = sample_data(hyperparams.n_samples)
    ht_train = sample_data(1)
    xt_train = ht_train + np.random.normal(0, hyperparams.snr, size=[1, ht_train.shape[1]])

    #G = keras.models.load_model('g_model.hdf5')
    G_in = Input(shape=(1,hyperparams.noise_dim))
    G, G_out = get_generative(G_in, lr=hyperparams.g_lr)
    #G.load_weights('best_g_weights.hdf5')
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


    d_loss, g_loss, d_acc, g_acc = train(GAN, G, D, xt_train, encoder, epochs=hyperparams.epochs, n_samples=hyperparams.n_samples,
                           noise_samples=hyperparams.noise_samples, noise_dim=hyperparams.noise_dim, batch_size=hyperparams.batch_size, verbose=True)

    # plot several generated waveforms, noise-free signal, signal+noise
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

    plt.savefig('%swaveform_results.png' % hyperparams.outdir, dpi=500)
    plt.close()


    # plot loss and acc functions
    # initialize subplot figure
    f, (ax1, ax2) = plt.subplots(2, 1)
    loss_curve = pd.DataFrame(
        {
            'Generative Loss': g_loss,
            'Discriminative Loss': d_loss,
        }
    ) #.plot(title='Training loss', logy=True)
    ax1.plot(loss_curve)
    ax1.set_title('Training loss')
    #ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    #ax = ax.get_figure()

    acc_curve = pd.DataFrame(
        {
            'Generative Acc': g_acc,
            'Discriminative Acc': d_acc,
        }
    )
    ax2.plot(acc_curve)
    ax2.set_title('Training acc')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Acc")
    #ax2.set_yscale("log")

    plt.savefig('%sloss_and_acc.png' % hyperparams.outdir)
    plt.close()

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
