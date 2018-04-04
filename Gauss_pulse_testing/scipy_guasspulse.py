
# coding: utf-8

# In[10]:


import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from keras.models import Model
from keras.layers import Input, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from scipy import signal


# In[60]:

# set variables
n_samples = 500
batch_size = 32
epochs = 200
g_lr = 1e-3 #0.5e-3
d_lr = 1e-3 #9e-4


def sample_data(n_samples=100, x_vals=np.arange(0, 5, .1), max_offset=0.25, mul_range=[1, 2]):
    waveforms = []
    for i in range(n_samples):
        t = np.linspace(-1, 1, 4 * 100, endpoint=False)
        offset = np.random.random() * max_offset
        i, q, e = signal.gausspulse(t+offset, fc=15, bw=0.12, retquad=True, retenv=True)
        waveforms.append(i)
    return np.array(waveforms)

ax = pd.DataFrame(np.transpose(sample_data(1))).plot()
ax = ax.get_figure()
ax.savefig('/home/hunter.gabbard/public_html/Burst/Gauss_pulse_testing/input_waveforms.png')
plt.close(ax)
sys.exit()


# In[61]:


def get_generative(G_in, dense_dim=200, out_dim=400, lr=1e-2):
    x = Dense(dense_dim)(G_in)
    LeakyReLU(alpha=0.3)(x)
    #x = Dropout(0.5)(x)
    G_out = Dense(out_dim, activation='tanh')(x)
    G = Model(G_in, G_out)
    opt = SGD(lr=lr)
    G.compile(loss='binary_crossentropy', optimizer=opt)
    return G, G_out

G_in = Input(shape=[10])
G, G_out = get_generative(G_in, lr=g_lr)
G.summary()


# In[75]:


def get_discriminative(D_in, lr=1e-4, drate=.25, n_channels=50, conv_sz=5, leak=.2):
    x = Reshape((-1, 1))(D_in)
    x = Conv1D(n_channels, conv_sz)(x)
    LeakyReLU(alpha=0.3)(x)
    x = Dropout(drate)(x)
    x = Flatten()(x)
    x = Dense(n_channels)(x)
    #x = Dropout(drate)(x)
    D_out = Dense(2, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D, D_out

D_in = Input(shape=[400])
D, D_out = get_discriminative(D_in, lr=d_lr)
D.summary()


# In[70]:


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

GAN_in = Input([10])
GAN, GAN_out = make_gan(GAN_in, G, D)
GAN.summary()


# In[71]:


def sample_data_and_gen(G, noise_dim=10, n_samples=100):
    XT = sample_data(n_samples=n_samples)
    XN_noise = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    XN = G.predict(XN_noise)
    X = np.concatenate((XT, XN))
    y = np.zeros((2*n_samples, 2))
    y[:n_samples, 1] = 1
    y[n_samples:, 0] = 1
    return X, y

def pretrain(G, D, noise_dim=10, n_samples=100, batch_size=32):
    X, y = sample_data_and_gen(G, n_samples=n_samples, noise_dim=noise_dim)
    set_trainability(D, True)
    D.fit(X, y, epochs=1, batch_size=batch_size)

pretrain(G, D, n_samples=n_samples, batch_size=batch_size)


# In[72]:


def sample_noise(G, noise_dim=10, n_samples=100):
    X = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    y = np.zeros((n_samples, 2))
    y[:, 1] = 1
    return X, y

def train(GAN, G, D, epochs=500, n_samples=100, noise_dim=10, batch_size=32, verbose=False, v_freq=1):
    d_loss = []
    g_loss = []
    e_range = range(epochs)
    if verbose:
        e_range = tqdm(e_range)
    for epoch in e_range:
        X, y = sample_data_and_gen(G, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, True)
        d_loss.append(D.train_on_batch(X, y))
        
        X, y = sample_noise(G, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch(X, y))
        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
    return d_loss, g_loss

d_loss, g_loss = train(GAN, G, D, epochs=epochs, n_samples=n_samples, 
                       batch_size=batch_size, verbose=True)


# In[73]:


ax = pd.DataFrame(
    {
        'Generative Loss': g_loss,
        'Discriminative Loss': d_loss,
    }
).plot(title='Training loss', logy=True)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax = ax.get_figure()
ax.savefig('/home/hunter.gabbard/public_html/Burst/Gauss_pulse_testing/loss.png')
plt.close(ax)

# In[74]:


N_VIEWED_SAMPLES = 2
data_and_gen, _ = sample_data_and_gen(G, n_samples=N_VIEWED_SAMPLES)
ax = pd.DataFrame(np.transpose(data_and_gen[N_VIEWED_SAMPLES:]))[5:].plot()
ax = ax.get_figure()
ax.savefig('/home/hunter.gabbard/public_html/Burst/Gauss_pulse_testing/gen_waveform.png')
plt.close(ax)
