from __future__ import division
from keras.models import Sequential, Model
from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.layers import Reshape, AlphaDropout, Dropout, GaussianDropout, GaussianNoise
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, UpSampling1D, Conv2DTranspose
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, AveragePooling1D, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ThresholdedReLU, ReLU
from keras.layers.core import Flatten
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.examples.tutorials.mnist import input_data
from scipy.stats import multivariate_normal as mvn
from scipy.special import logit, expit
from itertools import product as itprod
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
import glob
import random
import string
from sys import exit
import pandas as pd
import pickle, cPickle
import scipy
from scipy.stats import uniform, gaussian_kde, ks_2samp, anderson_ksamp
from scipy.signal import resample
from gwpy.table import EventTable
import keras
import h5py
from sympy import Eq, Symbol, solve
#import statsmodels.api as sm
from scipy import stats

cuda_dev = "4"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=cuda_dev
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

# define some global params
n_colors = 2		# greyscale = 1 or colour = 3 (multi-channel not supported yet)
n_pix = 1024	        # the rescaled image size (n_pix x n_pix)
n_sig = 1.0          # the noise standard deviation (if None then use noise images)
batch_size = 8        # the batch size (twice this when testing discriminator)
pe_batch_size = 64
max_iter = 500*1000 	# the maximum number of steps or epochs
pe_iter = 5*100000         # the maximum number of steps or epochs for pe network 
cadence = 100		# the cadence of output images
save_models = True	# save the generator and discriminator models
do_pe = True		# perform parameter estimation? 
pe_cadence = 1000  	# the cadence of PE outputs
pe_grain = 95           # fineness of pe posterior grid
npar = 2 		# the number of parameters to estimate (PE not supported yet)
N_VIEWED = 5           # number of samples to view when plotting
chi_loss = False        # set whether or not to use custom loss function
lr = 9e-5              # learning rate for all networks
GW150914 = True        # run on lalinference produced GW150914 waveform 
gw150914_tmp = True    # run on gw150914-like template
do_old_model = False     # run previously saved model for all models
do_contours = True      # plot credibility contours on pe estimates
do_only_old_pe_model = True # run previously saved pe model only
retrain_pe_mod = False
contour_cadence = 100   # the cadence of PE contour plot outputs
n_noise_real = 1       # number of noise realizations per training sample
event_name = 'gw150914'

# the locations of signal files and output directory
signal_path = '/home/hunter.gabbard/CBC/GenNet/BBH_version/data/event_%s_psd.pkl' % event_name
#pars_path = '/home/hunter.gabbard/Burst/GenNet/tests/data/burst/data_pars.pkl'
if gw150914_tmp:
    out_path = '/home/hunter.gabbard/public_html/CBC/mahoGANy/%s_template' % event_name 
if not GW150914 and not gw150914_tmp:
    out_path = '/home/hunter.gabbard/public_html/CBC/mahoGANy/rand_bbh_results/cuda_dev_%s' % cuda_dev

class bbhparams:
    def __init__(self,mc,M,eta,m1,m2,ra,dec,iota,phi,psi,idx,fmin,snr,SNR):
        self.mc = mc
        self.M = M
        self.eta = eta
        self.m1 = m1
        self.m2 = m2
        self.ra = ra
        self.dec = dec
        self.iota = iota
        self.phi = phi
        self.psi = psi
        self.idx = idx
        self.fmin = fmin
        self.snr = snr
        self.SNR = SNR

def chisquare_Loss(yTrue,yPred):
    #K.sum( K.square(K.log(data) - K.log(wvm)/n_sig ))
    #K.categorical_crossentropy(wvm, data)
    #return K.sum( K.square(K.log(yTrue) - K.log(yPred)/n_sig ), axis=-1)
    #return K.sqrt(K.sum(K.square(yPred - yTrue), axis=-1))
    return K.sum( K.square(yTrue - yPred)/(n_sig**2), axis=-1)

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

class MyLayer(Layer):
    """
    This layer just computes 
    a) the mean of the differences between the input image and the measured image
    b) the mean of the squared differences between the input image and the measured image
    Calling this layer requires you to pass the measured image (const)
    """
    def __init__(self, const, **kwargs):
        self.const = K.constant(const)		# the input measured image i.e., h(t)
        print(self.const)
        self.output_dim = (n_pix,2,1)			# the output dimension
        super(MyLayer,self).__init__(**kwargs)

    def build(self, input_shape):
	super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # computes the mean of the difference between the meausured image and the generated signal
        # and returns it as a Keras object
        # add in K cube of diff as third option
        diff = self.const - x
        return K.stack([x,diff], axis=2)

    def compute_output_shape(self, input_shape):
        # the output shape which seems to be (None,2) since the 2 is the number of 
        # outputs and the None needs to be there?
        return (input_shape[0],n_pix,2,1)

def data_subtraction_model(noise_signal,npix):
    """
    This model simply applies the signal subtraction from the measured image
    You must pass it the measured image
    """
    model = Sequential()
    model.add(MyLayer(noise_signal,input_shape=(npix,1))) # used to be another element for n_colors
   
    return model

def generator_model():
    """
    The generator that should train itself to generate noise free signals
    """
    model = Sequential()
    act = 'tanh'
    momentum = 0.99
    drate = 0.2
    padding = 'same'
    weights = 'glorot_uniform'
    alpha = 0.5
    filtsize = 5 # 10 is best
    num_lays = 5
    batchnorm = True
    
    # the first dense layer converts the input (100 random numbers) into
    # 1024 numbers and outputs with a tanh activation
    #model.add(Dense(128, input_shape=(100,)))
    #if batchnorm: model.add(BatchNormalization(momentum=momentum))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(GaussianDropout(0.3))
 
    #model.add(Dense(1024))
    #if batchnorm: model.add(BatchNormalization(momentum=momentum))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(GaussianDropout(0.3))

    #model.add(Dense(256))
    #if batchnorm: model.add(BatchNormalization(momentum=momentum))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(GaussianDropout(0.3))

    # the second dense layer expands this up to 32768 and again uses a
    # tanh activation function
    model.add(Dense(256 * 1 * int(n_pix/2), kernel_initializer=weights, input_shape=(100,)))
    if batchnorm: model.add(BatchNormalization(momentum=momentum))
    if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
    elif act == 'prelu': model.add(PReLU())
    else: model.add(Activation(act))
    model.add(Dropout(drate))

    # then we reshape into a cube, upsample by a factor of 2 in each of
    # 2 dimensions and apply a 2D convolution with filter size 5x5
    # and 64 neurons and again the activation is tanh 
    model.add(Reshape((int(n_pix/2), 256)))
    for i in range(num_lays):
        i += 1
        if i == 1:
            model.add(UpSampling1D(size=2))
            model.add(Conv1D(64, filtsize, kernel_initializer=weights, strides=2, padding=padding))
            #model.add(MaxPooling1D(pool_size=2))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
    
        elif i == 2:
            model.add(UpSampling1D(size=2))
            model.add(Conv1D(128, filtsize, kernel_initializer=weights, strides=1, padding=padding))
            #model.add(MaxPooling1D(pool_size=2))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))

        elif i == 3:
            #model.add(UpSampling1D(size=2))
            model.add(Conv1D(256, filtsize, kernel_initializer=weights, strides=1, padding=padding))
            #model.add(MaxPooling1D(pool_size=2))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))

        elif i == 4:
            #model.add(UpSampling1D(size=2))
            model.add(Conv1D(512, filtsize, kernel_initializer=weights, strides=1, padding=padding))
            #model.add(MaxPooling1D(pool_size=2))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))

        elif i == 5:
            #model.add(UpSampling1D(size=2))
            model.add(Conv1D(1024, filtsize, kernel_initializer=weights, strides=1, padding=padding))
            #model.add(MaxPooling1D(pool_size=2))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))

    
    #model.add(UpSampling1D(size=2))
    # if we have a 64x64 pixel dataset then we upsample once more 
    #if n_pix==64:
    #    model.add(UpSampling2D(size=(1, 2)))
    # apply another 2D convolution with filter size 5x5 and a tanh activation
    # the output shape should be n_colors x n_pix x n_pix
    model.add(Conv1D(1, filtsize, padding=padding))
    model.add(Activation('linear')) # this should be tanh
    model.summary()

    return model

def signal_pe_model():
    
    #The PE network that learns how to convert images into parameters

    """    
    model = Sequential()
    act = 'linear'
    momentum = 0.9

    # the first layer is a 2D convolution with filter size 5x5 and 64 neurons
    # the activation is tanh and we apply a 2x2 max pooling
    model.add(Conv1D(512, 64, strides=2, input_shape=(n_pix,1), padding='valid'))
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(PReLU())
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(MaxPooling2D(pool_size=(1, 2)))
    #model.add(BatchNormalization(momentum=momentum))

    # the next layer is another 2D convolution with 128 neurons and a 5x5
    # filter. More 2x2 max pooling and a tanh activation. The output is flattened
    # for input to the next dense layer
    model.add(Conv1D(512, 32, strides=2))
    model.add(Activation(act))
    #model.add(Dropout(0.5))
    #model.add(LeakyReLU(alpha=0.2))
    model.add(PReLU())
    #model.add(MaxPooling2D(pool_size=(1, 2)))
    #model.add(BatchNormalization(momentum=momentum))

    model.add(Conv1D(512, 16, strides=2))
    model.add(Activation(act))
    #model.add(Dropout(0.5))
    model.add(PReLU())
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))

    model.add(Conv1D(512, 8, strides=2))
    #model.add(Dropout(0.5))
    model.add(Activation(act))
    #model.add(PReLU())
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))

    model.add(Flatten())

    # we now use a dense layer with 1024 outputs and a tanh activation
    model.add(Dense(1024))
    #model.add(Dropout(0.5))
    model.add(Activation(act))
    #model.add(PReLU())
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))
    model.add(PReLU())

    # the final dense layer has a linear activation and 2 outputs
    # we are currently testing with only 2 outputs - can be generalised
    model.add(Dense(2))
    model.add(Activation('relu'))
    #model.add(PReLU())
    """    

    
    inputs = Input(shape=(n_pix,1))
    act = 'relu'
    drate = 0.3

    mc_branch = Conv1D(64, 5, strides=2, padding='same')(inputs)
    mc_branch = Activation(act)(mc_branch)

    mc_branch = Conv1D(128, 5, strides=2)(mc_branch)
    mc_branch = Activation(act)(mc_branch)

    mc_branch = Conv1D(256, 5, strides=2)(mc_branch)
    mc_branch = Activation(act)(mc_branch)

    mc_branch = Conv1D(512, 5, strides=2)(mc_branch)
    mc_branch = Activation(act)(mc_branch)

    mc_branch = Flatten()(mc_branch)

    #mc_branch = Dense(1024)(mc_branch)
    #mc_branch = Activation(act)(mc_branch)

    mc_branch = Dense(1)(mc_branch)
    mc_branch = Activation('relu')(mc_branch)
    
    act = 'relu' 
    q_branch = Conv1D(64, 5, strides=2, padding='same')(inputs)
    q_branch = Activation(act)(q_branch)
    #q_branch = LeakyReLU(alpha=0.2)(q_branch)
    #q_branch = ReLU(max_value=1)(q_branch)
    #q_branch = GaussianDropout(drate)(q_branch)

    q_branch = Conv1D(128, 5, strides=2)(q_branch)
    q_branch = Activation(act)(q_branch)
    #q_branch = LeakyReLU(alpha=0.2)(q_branch)
    #q_branch = ReLU(max_value=1)(q_branch)
    #q_branch = GaussianDropout(drate)(q_branch)

    q_branch = Conv1D(256, 5, strides=2)(q_branch)
    q_branch = Activation(act)(q_branch)
    #q_branch = LeakyReLU(alpha=0.2)(q_branch)
    #q_branch = ReLU(max_value=1)(q_branch)
    #q_branch = GaussianDropout(drate)(q_branch)

    q_branch = Conv1D(512, 5, strides=2)(q_branch)
    q_branch = Activation(act)(q_branch)
    #q_branch = LeakyReLU(alpha=0.2)(q_branch)
    #q_branch = ReLU(max_value=1)(q_branch)
    #q_branch = GaussianDropout(drate)(q_branch)
    
    q_branch = Conv1D(1024, 5, strides=2)(q_branch)
    q_branch = Activation(act)(q_branch)

    q_branch = Flatten()(q_branch)

    #q_branch = Dense(1024)(q_branch)
    #q_branch = Activation(act)(q_branch)
    #q_branch = PReLU()(q_branch)

    q_branch = Dense(1)(q_branch)
    #q_branch = Activation('sigmoid')(q_branch)
    q_branch = ReLU(max_value=1.0)(q_branch)
    model = Model(
        inputs=inputs,
        outputs=[mc_branch, q_branch],
        name="pe net")
    
    return model

def signal_discriminator_model():
    """
    The discriminator that should train itself to recognise generated signals
    from real signals
    """

    
    #act='tanh'
    momentum=0.99
    weights = 'glorot_uniform'
    drate = 0.4
    act = 'leakyrelu'
    alpha = 0.2
    padding = 'same'
    num_lays = 2
    batchnorm = False
    # so 16x2 gives best waveform reconstruction, but not best pe results? Kinda weird ...
    # around 4000 epochs getting ~46% beta result.
    # around 10000 epochs getting ~60% beta result.
    filtsize = (16,2) # 5x2 is the best

    model = Sequential()

    # the first layer is a 2D convolution with filter size 5x5 and 64 neurons
    # the activation is tanh and we apply a 2x2 max pooling
    for i in range(num_lays):
        i += 1
        if i == 1:
            model.add(Conv2D(64, (5,5), kernel_initializer=weights, input_shape=(n_pix,2,1), strides=(2,1), padding=padding))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            #model.add(MaxPooling1D(pool_size=2))
    
        # the next layer is another 2D convolution with 128 neurons and a 5x5 
        # filter. More 2x2 max pooling and a tanh activation. The output is flattened
        # for input to the next dense layer
        elif i == 2:
            model.add(Conv2D(128, (5,5), kernel_initializer=weights, strides=(2,1), padding=padding))
            #if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            model.add(Dropout(drate))
            #model.add(MaxPooling1D(pool_size=2))

        elif i == 3:
            model.add(Conv2D(256, (5,5), kernel_initializer=weights, strides=(2,1), padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            #model.add(MaxPooling1D(pool_size=2))

        elif i == 4:
            model.add(Conv2D(512, (5,5), kernel_initializer=weights, strides=(2,1), padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            #model.add(MaxPooling1D(pool_size=2))

        elif i == 5:
            model.add(Conv2D(1024, (5,5), kernel_initializer=weights, strides=(2,1), padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            # model.add(MaxPooling1D(pool_size=2))

        elif i == 6:
            model.add(Conv2D(1024, (5,5), kernel_initializer=weights, strides=(2,1), padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            # model.add(MaxPooling1D(pool_size=2))    

    model.add(Flatten())

    # we now use a dense layer with 1024 outputs and a tanh activation
    #model.add(Dense(1024))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))

    # the final dense layer has a sigmoid activation and a single output
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
 
    return model

def generator_after_subtracting_noise(generator, data_subtraction):
    """
    This is the sequence used for training the generator to make signals, that 
    when subtracted from the measured data, give Gaussian noise of zero mean and 
    expected variance
    """
    model = Sequential()
    model.add(generator)		# the trainable parameters are in this model
    model.add(data_subtraction)		# there are no parameters ro train in this model
    return model

def generator_containing_signal_discriminator(generator, signal_discriminator):
    """
    This is the sequence used to train the signal generator such that the signals
    it generates are consistent with the training signals seen by the disciminator
    """
    model = Sequential()
    model.add(generator)		# the trainable parameters are in this model
    model.add(signal_discriminator)	# the discriminator has been set to not be trainable
    return model

def plot_losses(losses,filename,logscale=False,legend=None,chi_loss=chi_loss):
    """
    Make loss and accuracy plots and output to file.
    Plot with x and y log-axes is desired
    """

    # plot losses
    fig = plt.figure()
    losses = np.array(losses)
    ax1 = fig.add_subplot(211)	
    #if chi_loss:
    #    losses = np.reshape(losses, (losses.shape[0],1)) 
    ax1.plot(losses[:,0],'b')
    if losses.shape[1]>2:
        ax1.plot(losses[:,2],'r')
    if losses.shape[1]>4:
        ax1.plot(losses[:,4],'g')
    ax1.set_xlabel(r'epoch')
    ax1.set_ylabel(r'loss')
    if legend is not None:
    	ax1.legend(legend,loc='upper left')
    
    # plot accuracies
    ax2 = fig.add_subplot(212)
    ax2.plot(logit(losses[:,1]),'b')
    if losses.shape[1]>3:
        ax2.plot(logit(losses[:,3]),'r')
    if losses.shape[1]>5:
        ax2.plot(logit(losses[:,5]),'g')
    # rescale axis using a logistic function so that we see more detail
    # close to 0 and close 1
    ax2.set_yticks(logit([0.001,0.01,0.1,0.5,0.9,0.99,0.999]))
    ax2.set_yticklabels(['0.001','0.01','0.1','0.5','0.9','0.99','0.999'])
    ax2.set_xlabel(r'epoch')
    ax2.set_ylabel(r'accuracy')
    if logscale==True:
        ax1.set_xscale("log", nonposx='clip')
        #ax2.set_xscale("log", nonposx='clip')
        ax1.set_yscale("log", nonposy='clip')
    plt.savefig(filename)
    plt.close('all')

def plot_pe_accuracy(true_pars,est_pars,outfile):
    """
    Plots the true vs the estimated paranmeters from the PE training
    Change est_pars to est_pars[0] or est_pars[1] if doing 
    multiple pe network.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(121,aspect=1.0)
    #ax1.plot(true_pars[:,0],est_pars[:,0],'.b', markersize=0.5)
    ax1.plot(true_pars[:,0],est_pars[0],'.b', markersize=0.5)
    ax1.plot([0,np.max(true_pars[:,0])],[0,np.max(true_pars[:,0])],'--k')
    ax1.set_xlabel(r'True parameter 1')
    ax1.set_ylabel(r'Estimated parameter 1')
    ax2 = fig.add_subplot(122,aspect=1.0)
    #ax2.plot(true_pars[:,1],est_pars[:,1],'.b', markersize=0.5)
    ax2.plot(true_pars[:,1],est_pars[1],'.b', markersize=0.5)
    ax2.plot([0,np.max(true_pars[:,1])],[0,np.max(true_pars[:,1])],'--k')
    ax2.set_xlabel(r'True parameter 2')
    ax2.set_ylabel(r'Estimated parameter 2')
    plt.savefig(outfile)
    plt.savefig('%s/latest/pe_accuracy.png' % out_path)
    plt.close('all')

def convert_q_mc_to_pars(pe_samples):

    # plot other representations of mass parameters for GAN pe samples
    post_m1 = []
    post_m2 = []
    eta = []
    M = []
    for comp in range(pe_samples[0].shape[0]):
        m1 = Symbol('m1')
        eqn_m1 = Eq((m1 + (m1/pe_samples[1][comp])) * (m1*(m1/pe_samples[1][comp])/(m1+(m1/pe_samples[1][comp]))**2)**(3.0/5.0), pe_samples[0][comp])
        m1 = float(solve(eqn_m1)[0])
        post_m1.append(m1)

        m2 = Symbol('m2')
        eqn_m2 = Eq((pe_samples[1][comp]*m2 + m2) * ((pe_samples[1][comp]*m2)*m2/((pe_samples[1][comp]*m2)+m2)**2)**(3.0/5.0), pe_samples[1][comp])
        m2 = float(solve(eqn_m2)[0])
        post_m2.append(m2)

        M.append(m1 + m2)
        eta.append(m1*m2/(m1+m2)**2)

    return post_m1, post_m2, eta, M

def plot_pe_samples(pe_samples,truth,like,outfile,index,x,y,lalinf_dist=None,pe_std=None):
    """
    Makes scatter plot of samples estimated from PE model
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(223)
    
    if like is not None:
        # compute enclose probability contours
        enc_post = get_enclosed_prob(like,1.0/pe_grain)
        X, Y = np.meshgrid(x,y)
        cmap = plt.cm.get_cmap("Greys")
        ax1.contourf(X, Y, enc_post, 100, cmap=cmap) 
        ax1.contour(X, Y, enc_post, [1.0-0.68], colors='b',linestyles='solid')
        ax1.contour(X, Y, enc_post, [1.0-0.9], colors='b',linestyles='dashed')
        ax1.contour(X, Y, enc_post, [1.0-0.99], colors='b',linestyles='dotted') 
    # plot pe samples
    if pe_samples is not None:
        #ax1.plot(pe_samples[:,0],pe_samples[:,1],'.r',markersize=0.8)
        ax1.plot(pe_samples[0],pe_samples[1],'.r',markersize=0.8)
        #ax1.plot(signal_train_pars[:,0],signal_train_pars[:,1],'+g',markersize=0.8)  

        if do_contours and (index>0): # and ((index % contour_cadence == 0) 
            # plot contours for generated samples
            #contour_y = np.reshape(pe_samples[:,1], (pe_samples[:,1].shape[0]))
            #contour_x = np.reshape(pe_samples[:,0], (pe_samples[:,0].shape[0]))
            contour_y = np.reshape(pe_samples[1], (pe_samples[1].shape[0]))
            contour_x = np.reshape(pe_samples[0], (pe_samples[0].shape[0]))
            contour_dataset = np.array([contour_x,contour_y])
            kernel_cnn = make_contour_plot(ax1,contour_x,contour_y,contour_dataset,'red',flip=False)

    # plot contours of lalinf distribution
    if lalinf_dist is not None:
        # plot lalinference parameters
        ax1.plot(lalinf_dist[0][:],lalinf_dist[1][:],'.b', markersize=0.8)

        if do_contours and (index>0): # and ((index % contour_cadence == 0)
            # plot lalinference parameter contours
            kernel_lalinf = make_contour_plot(ax1,lalinf_dist[0][:],lalinf_dist[1][:],lalinf_dist,'blue',flip=False)

    # plot pe_std error bars
    if pe_std:
        print('pe std: %s, %s'% (str(pe_std[0]),str(pe_std[1])))
        ax1.plot([truth[0]-pe_std[0],truth[0]+pe_std[0]],[truth[1],truth[1]], '-c')
        ax1.plot([truth[0], truth[0]],[truth[1]-pe_std[1],truth[1]+pe_std[1]], '-c')

    ax1.plot([truth[0],truth[0]],[np.min(y),np.max(y)],'-k', alpha=0.5)
    ax1.plot([np.min(x),np.max(x)],[truth[1],truth[1]],'-k', alpha=0.5)

    # add histograms to corner plot
    # chrip mass hist
    ax2 = fig.add_subplot(221)
    # inverse mass ratio hist
    ax3 = fig.add_subplot(224)

    # plot histograms
    ax2.hist(pe_samples[0], bins=100, normed=True)
    ax2.hist(lalinf_dist[0][:],bins=100, normed=True)
    ax2.set_xticks([])
    ax3.hist(pe_samples[1],bins=100,orientation=u'horizontal',normed=True)
    ax3.hist(lalinf_dist[1][:],bins=100,orientation=u'horizontal', normed=True) 
    ax3.set_yticks([])

    ks_score, ad_score, beta_score = overlap_tests(pe_samples,lalinf_dist,truth,kernel_cnn,kernel_lalinf)
    #print('mc KS result: {0}'.format(ks_score[0]))
    #print('q KS result: {0}'.format(ks_score[1]))
    #print('mc AD result: {0}'.format(ad_score[0]))
    #print('q AD result: {0}'.format(ad_score[1]))
    print('beta result: {0}'.format(beta_score))

    ax1.set_xlabel(r'mc')
    ax1.set_ylabel(r'mass ratio')
    ax1.legend(['Overlap: %s' % str(np.round(beta_score,3))])
    #ax1.set_xlim([np.min(all_pars[:,0]),np.max(all_pars[:,0])])
    #ax1.set_ylim([np.min(all_pars[:,1]),np.max(all_pars[:,1])])
    plt.savefig('%s/pe_samples%05d.png' % (outfile,index))
    
    if do_pe and not do_only_old_pe_model or retrain_pe_mod:
        plt.savefig('%s/latest/pe_samples_cnn.png' % (outfile), dpi=400)
    else: plt.savefig('%s/latest/pe_samples_gan.png' % (outfile), dpi=400)
    plt.close('all')

    return beta_score


def make_contour_plot(ax,x,y,dataset,color='red',flip=False):

    # Make a 2d normed histogram
    H,xedges,yedges=np.histogram2d(x,y,bins=20,normed=True)

    if flip == True:
        H,xedges,yedges=np.histogram2d(y,x,bins=20,normed=True)
        dataset = np.array([dataset[1,:],dataset[0,:]])

    norm=H.sum() # Find the norm of the sum
    # Set contour levels
    contour1=0.99
    contour2=0.90
    contour3=0.68

    # Set target levels as percentage of norm
    target1 = norm*contour1
    target2 = norm*contour2
    target3 = norm*contour3

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process
    def objective(limit, target):
        w = np.where(H>limit)
        count = H[w]
        return count.sum() - target

    # Find levels by summing histogram to objective
    level1= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
    level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
    level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))

    # For nice contour shading with seaborn, define top level
    level4=H.max()
    levels=[level1,level2,level3,level4]

    # Pass levels to normed kde plot
    #sns.kdeplot(x,y,shade=True,ax=ax,n_levels=levels,cmap=color,alpha=0.5,normed=True)
    X, Y = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = gaussian_kde(dataset)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.contour(X,Y,Z,levels=levels,alpha=0.5,colors=color)
    #ax.set_aspect('equal')

    return kernel

def get_enclosed_prob(x,dx):
    """
    Generates contour data for enclosed probability
    """
    s = x.shape
    x = x.flatten()
    idx = np.argsort(x)[::-1]
    y = np.zeros(x.shape)
    y[idx] = np.cumsum(x[idx])*dx*dx
    y /= np.max(y)
    return 1.0 - y.reshape(s)

def set_trainable(model, trainable):
    """
    Allows us to switch off models in sequences as trainable
    """
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def load_data(signal_path,pars_path,Ngauss_sig):
    """
    Truncates input GW waveform data file into a numpy array called data.
    """
    print('Using data for: {0}'.format(signal_path))
    print('Using parameters for: {0}'.format(pars_path))

    # load in time series dataset
    with open(signal_path, 'rb') as rfp:
        data = pickle.load(rfp)

    data = data[:Ngauss_sig]

    # load in parameter dataset
    with open(pars_path, 'rb') as rfp:
        pars = pickle.load(rfp)

    pars = pars[:Ngauss_sig]

    return data,pars

def load_gw_event(path):
    """
    Truncates input GW waveform data file into a numpy array called data.
    Will also resample data if desired.
    """
    print('Using data for: {0}'.format(path))

    # load in time series dataset
    with open(signal_path, 'rb') as rfp:
        data = pickle.load(rfp)

    return data

def overlap_tests(pred_samp,lalinf_samp,true_vals,kernel_cnn,kernel_lalinf):
    """
    Perform Anderson-Darling, K-S, and overlap tests
    to get quantifiable values for accuracy of GAN
    PE method
    """

    # do k-s test
    ks_mc_score = ks_2samp(pred_samp[0].reshape(pred_samp[0].shape[0],),lalinf_samp[0][:])
    ks_q_score = ks_2samp(pred_samp[1].reshape(pred_samp[1].shape[0],),lalinf_samp[1][:])
    ks_score = np.array([ks_mc_score,ks_q_score])

    # do anderson-darling test
    ad_mc_score = anderson_ksamp([pred_samp[0].reshape(pred_samp[0].shape[0],),lalinf_samp[0][:]])
    ad_q_score = anderson_ksamp([pred_samp[1].reshape(pred_samp[1].shape[0],),lalinf_samp[1][:]])
    ad_score = [ad_mc_score,ad_q_score]

    # compute overlap statistic
    comb_mc = np.concatenate((pred_samp[0],lalinf_samp[0][:].reshape(lalinf_samp[0][:].shape[0],1)))
    comb_q = np.concatenate((pred_samp[1],lalinf_samp[1][:].reshape(lalinf_samp[1][:].shape[0],1)))
    X, Y = np.mgrid[np.min(comb_mc):np.max(comb_mc):100j, np.min(comb_q):np.max(comb_q):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    #cnn_pdf = np.reshape(kernel_cnn(positions).T, X.shape)
    cnn_pdf = kernel_cnn.pdf(positions)

    #X, Y = np.mgrid[np.min(lalinf_samp[0][:]):np.max(lalinf_samp[0][:]):100j, np.min(lalinf_samp[1][:]):np.max(lalinf_samp[1][:]):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    #lalinf_pdf = np.reshape(kernel_lalinf(positions).T, X.shape)
    lalinf_pdf = kernel_lalinf.pdf(positions)

    beta_score = np.divide(np.sum( cnn_pdf*lalinf_pdf ),
                              np.sqrt(np.sum( cnn_pdf**2 ) * 
                              np.sum( lalinf_pdf**2 )))
    

    return ks_score, ad_score, beta_score

def plot_waveform_est(signal_image,noise_signal,generated_images,out_path,i,plot_lalinf_wvf=False):
    # plot original waveform
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
    ax = signal_image
    ax1.plot(ax, color='cyan', alpha=0.5, linewidth=0.5)
    ax1.plot(noise_signal, color='green', alpha=0.35, linewidth=0.5)
    #ax1.set_title('signal + (sig+noise)')

    # plotable generated signals
    generated_images = np.random.permutation(generated_images)
    if not plot_lalinf_wvf:
        gen_sig = np.reshape(generated_images[:N_VIEWED], (generated_images[:N_VIEWED].shape[0],generated_images[:N_VIEWED].shape[1]))
    if plot_lalinf_wvf:
        gen_sig = np.reshape(generated_images[:N_VIEWED], (generated_images[:N_VIEWED].shape[0],generated_images[:N_VIEWED].shape[1]))

    # compute percentile curves
    perc_90 = []
    perc_75 = []
    perc_25 = []
    perc_5 = []
    for n in range(gen_sig.shape[1]):
        perc_90.append(np.percentile(gen_sig[:,n], 90))
        perc_75.append(np.percentile(gen_sig[:,n], 75))
        perc_25.append(np.percentile(gen_sig[:,n], 25))
        perc_5.append(np.percentile(gen_sig[:,n], 5))

    # plot generated signals - first image is the noise-free true signal
    ax2.plot(signal_image, color='cyan', linewidth=0.5, alpha=0.5)
    ax2.fill_between(np.linspace(0,len(perc_90),num=len(perc_90)),perc_90, perc_5, lw=0,facecolor='#d5d8dc')
    ax2.fill_between(np.linspace(0,len(perc_75),num=len(perc_75)),perc_75, perc_25, lw=0,facecolor='#808b96')
    #ax2.set_title('gen + sig + (sig+noise)')
    ax2.set(ylabel='Amplitude (counts)')

    # plot residuals - generated images subtracted from the measured image
    # the first image is the true noise realisation
    residuals = np.transpose(np.transpose(noise_signal)-gen_sig)
    ax3.plot((residuals[:,0]), color='black', linewidth=0.5)
    ax3.plot((residuals), color='red', alpha=0.25, linewidth=0.5)
    #ax3.plot((noise_signal - ax), color='black', alpha=0.5, linewidth=0.5)

    #ax3.set_title('Residuals')
    ax3.set(xlabel='Time')
    # save waveforms plot
    if not plot_lalinf_wvf:
        plt.savefig('%s/waveform_results%05d.png' % (out_path,i), dpi=500)
        plt.savefig('%s/latest/most_recent_waveform.png' % out_path, dpi=400)
        print('Completed waveform plotting routine!')
    if plot_lalinf_wvf:
        plt.savefig('%s/latest/most_recent_lalinf_waveform.png' % out_path, dpi=400)
        print('Completed lalinf waveform plotting routine!')
    plt.close()

def main():

    ################################################
    # READ/GENERATE DATA ###########################

    # setup output directory - make sure it exists
    os.system('mkdir -p %s' % out_path) 

    template_dir = 'templates/'
    training_num = 50000 

    # load in lalinference m1 and m2 parameters
    pickle_lalinf_pars = open("data/%s_mc_q_lalinf_post.sav" % event_name)
    lalinf_pars = pickle.load(pickle_lalinf_pars)


    # load hplus and hcross pickle file
    #pickle_hp = open("%shp.pkl" % template_dir,"rb")
    #hp = pickle.load(pickle_hp)
    #pickle_hc = open("%shc.pkl" % template_dir,"rb")
    #hc = pickle.load(pickle_hc)
    #pickle_fmin = open("%sfmin.pkl" % template_dir,"rb")
    #fmin_bank = pickle.load(pickle_fmin)

    # load time series template pickle file
    file_idx_list = []
    pickle_ts = open("%s%s_ts_0_%sSamp.sav" % (template_dir,event_name,training_num),"rb")
    ts = pickle.load(pickle_ts)
    pickle_par = open("%s%s_params_0_%sSamp.sav" % (template_dir,event_name,training_num),"rb")
    par = pickle.load(pickle_par)
    if len(file_idx_list) > 0:
        ts = np.array(ts[0][:-1])
        par = np.array(par[:-1])
    else:
        ts = np.array(ts[0])
        par = np.array(par)
    par = np.reshape(par,(par.shape[0],1))
    print("loading file: _ts_0_%sSamp.sav" % (training_num))
    print("loading file: _params_0_%sSamp.sav" % (training_num))
    for idx in file_idx_list:
        pickle_ts = open("%s_ts_%s_%sSamp.sav" % (template_dir,str(idx),training_num),"rb")
        ts_new = pickle.load(pickle_ts)
        ts = np.vstack((ts,ts_new[0]))

        # load corresponding parameters template pickle file
        pickle_par = open("%s_params_%s_%sSamp.sav" % (template_dir,str(idx),training_num),"rb")
        par_new = np.array(pickle.load(pickle_par))
        par_new = np.reshape(par_new,(par_new.shape[0],1))
        par = np.vstack((par,par_new))

        print("loading file: _ts_%s_%sSamp.sav" % (str(idx),training_num))
        print("loading file: _params_%s_%sSamp.sav" % (str(idx),training_num))

        if idx < file_idx_list[-1]:
            ts = ts[:-1]
            par = par[:-1]

    par = par.reshape(par.shape[0])
    par = list(par)
    ts = [ts]

    signal_train_images = np.reshape(ts[0], (ts[0].shape[0],ts[0].shape[2]))

    #plt.plot(signal_train_images[0])
    #plt.savefig('/home/hunter.gabbard/public_html/CBC/mahoGANy/gw150914_template/input_waveform.png')
    #plt.close()
    #exit()

    signal_train_pars = []
    for k in par:
        signal_train_pars.append([k.mc,(k.m2/k.m1)])

    signal_train_pars = np.array(signal_train_pars)
    """
    # not really sure what this is for???
    if do_pe:
	tmp_signal_images, signal_train_noisy_pars = make_burst_waveforms(Ngauss_sig,rand=True)	
	tmp_noise_images = np.random.normal(0.0,n_sig,size=(Ngauss_sig,1,n_pix))
        signal_train_noisy_images = np.array([a + b for a,b in zip(tmp_signal_images,tmp_noise_images)]) #.reshape(Ngauss_sig,1,n_pix,n_colors)
    """

    """ 
    # print out input waveforms
    ax = pd.DataFrame(np.transpose(sample_data(25))).plot(legend=False)
    ax = ax.get_figure()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    ax.savefig('%s/input_waveforms.png' % out_path)
    plt.close(ax)
    """

    # randomly extract single image as the true signal
    # IMPORTANT - make sure to delete it from the training set
    if not GW150914 and not gw150914_tmp:
        i = np.random.randint(0,signal_train_images.shape[0],size=1)
        signal_image = signal_train_images[i,:]
        signal_train_images = np.delete(signal_train_images,i,axis=0)


    # choose fixed signal
    # pars will be default params in function
    if GW150914:
        pickle_gw150914 = open("data/%s.sav" % event_name,"rb")
        noise_signal = np.reshape(pickle.load(pickle_gw150914) * 817.98,(n_pix,1)) # 817.98
        signal_image = pickle.load(open("data/%s_data.pkl" % event_name,"rb")) * 817.98 # 1079.22
        noise_image = np.random.normal(0, n_sig, size=[1, signal_image.shape[0]])
        gan_noise_signal = np.transpose(signal_image + noise_image)

        #plt.plot(lal_noise_signal, alpha=0.5,label='lal_noise')
        #plt.plot(gan_noise_signal, alpha=0.5,label='numpy_noise')
        #plt.legend()
        #plt.hist(noise_signal,100)
        #plt.savefig('/home/hunter.gabbard/public_html/CBC/mahoGANy/gw150914_template/input_waveform.png')
        #plt.close()
        #exit()
        #signal_image = signal_train_images[-1,:]
        signal_train_images = np.delete(signal_train_images,-1,axis=0)

    if gw150914_tmp and not GW150914:
        signal_image = signal_train_images[-1,:]
        signal_train_images = np.delete(signal_train_images,-1,axis=0)

    if do_pe and not GW150914 and not gw150914_tmp:
        signal_pars = signal_train_pars[i,:][0]
        signal_train_pars = np.delete(signal_train_pars,i,axis=0)    

    if do_pe and gw150914_tmp:
        signal_pars = signal_train_pars[-1,:]
        signal_train_pars = np.delete(signal_train_pars,-1,axis=0)

    # combine signal and noise - this is the measured data i.e., h(t)
    if GW150914:
        #noise_signal = noise_signal[int((4*512/2)-(0.5*512)):int((4*512/2)+(0.5*512))]
        #signal_image = signal_image[int((32*4096/2)-(0.5*4096)):int((32*4096/2)+(0.5*4096))]

        # resample GW150914
        #noise_signal = resample(noise_signal[0],n_pix)
        signal_image = resample(signal_image,n_pix)

        peak_diff = np.abs(np.argmax(noise_signal)-np.argmax(signal_image))
        signal_image = np.roll(signal_image,peak_diff)

        # set signal_pars m1 and m2
        if not gw150914_tmp:
            signal_pars = [30.0,0.79]
        #signal_pars = signal_train_pars[-1,:]
        #signal_train_pars = np.delete(signal_train_pars,-1,axis=0)

    if not GW150914:
        # Generate single noise image
        signal_image = np.reshape(signal_image, (1, 512))
        noise_image = np.random.normal(0, n_sig, size=[1, signal_image.shape[1]])

        # combine signal and noise - this is the measured data i.e., h(t)
        noise_signal = np.transpose(signal_image + noise_image)
        signal_image = signal_image[0]

    plt.plot(signal_image)
    plt.plot(noise_signal, alpha=0.5)
    plt.savefig('%s/input_waveform.png' % out_path)
    plt.close()

    ################################################
    # SETUP MODELS #################################
   
    # initialise all models
    generator = generator_model()
    signal_discriminator = signal_discriminator_model()
    data_subtraction = data_subtraction_model(noise_signal,n_pix)
    if do_pe:
        signal_pe = signal_pe_model()    

    """
    setup generator training for when we subtract from the 
    measured data and expect Gaussian residuals.
    We use a mean squared error here since we want it to find 
    the situation where the residuals have the known mean=0, std=n_sig properties
    """
    # setup extra layer on generator
    data_subtraction_on_generator = generator_after_subtracting_noise(generator, data_subtraction)
    data_subtraction_on_generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])
    #data_subtraction_on_generator.compile(loss='binary_crossentropy', optimizer=Nadam(lr=lr), metrics=['accuracy'])

    # setup generator training when we pass the output to the signal discriminator
    signal_discriminator_on_generator = generator_containing_signal_discriminator(data_subtraction_on_generator, signal_discriminator)
    set_trainable(signal_discriminator, False)	# set the discriminator as not trainable for this step
    if not chi_loss:
        signal_discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])
        #signal_discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=Nadam(lr=lr), metrics=['accuracy'])
    elif chi_loss:
        signal_discriminator_on_generator.compile(loss=chisquare_Loss, optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    # setup trainin on signal discriminator model
    # This uses a binary cross entropy loss since we are just 
    # discriminating between real and fake signals
    set_trainable(signal_discriminator, True)	# set it back to being trainable
    signal_discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])
    #signal_discriminator.compile(loss='binary_crossentropy', optimizer=Nadam(lr=lr), metrics=['accuracy'])
    #elif chi_loss:
    #    signal_discriminator.compile(loss=chisquare_Loss, optimizer=Adam(lr=9e-5, beta_1=0.5), metrics=['accuracy'])

    if do_pe:
        signal_pe.compile(loss='mean_squared_error', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    # print the model summaries
    print(generator.summary())
    print(signal_discriminator_on_generator.summary())
    print(signal_discriminator.summary())
    if do_pe:
        print(signal_pe.summary())

    ################################################
    # DO PARAMETER ESTIMATION ######################

    if do_old_model:
        if do_pe:
            #signal_pe = keras.models.load_model('best_models/best_signal_pe.h5')
            signal_pe = keras.models.load_model('best_models/signal_pe.h5')
        signal_discriminator.load_weights('discriminator.h5')
        signal_discriminator_on_generator.load_weights('signal_dis_on_gen.h5')
        generator.load_weights('generator.h5')

    if do_only_old_pe_model:
        # load old pe model by default
        #signal_pe = keras.models.load_model('best_models/best_signal_pe.h5')
        signal_pe = keras.models.load_model('best_models/signal_pe.h5')

    if do_pe and not do_only_old_pe_model or retrain_pe_mod: #and not do_only_old_pe_model and not do_old_model:

        """
        # redefine training and input data
        # load signal training images and save examples
        signal_train_images, signal_train_pars = make_burst_waveforms(Ngauss_sig,rand=True)

        # randomly extract single image as the true signal
        # IMPORTANT - make sure to delete it from the training set
        i = np.random.randint(0,signal_train_images.shape[0],size=1)
        signal_image = signal_train_images[i,:]
        plt.plot(signal_image[0])
        plt.savefig('%s/input_waveform.png' % out_path)
        plt.close()
        signal_train_images = np.delete(signal_train_images,i,axis=0)

        if do_pe:
            signal_pars = signal_train_pars[i,:]
            print(signal_pars)
            signal_train_pars = np.delete(signal_train_pars,i,axis=0)

        # Generate single noise image
        noise_image = np.random.normal(0, n_sig, size=[1, signal_image.shape[1]])

        # combine signal and noise - this is the measured data i.e., h(t)
        noise_signal = signal_image #+ noise_image
        """
        
        # first compute true PE on a grid
        #x = np.linspace(0.25,0.75,pe_grain)
        #y = np.linspace(1.0/60.0,1.0/15.0,pe_grain)
        #xy = np.array([k for k in itprod(x,y)]).reshape(pe_grain*pe_grain,2)
        #L = []
        #for count,pars in enumerate(xy): # used to be x
        #    template,_ = make_burst_waveforms(1,tau=pars[1],t_0=pars[0]) #.reshape(1,n_pix)
	#    L.append(-0.5*np.sum(((np.transpose(noise_signal)-template)/n_sig)**2))
        #L = np.array(L).reshape(pe_grain,pe_grain).transpose()
        #L = np.exp(L-np.max(L)) 
        #plot_pe_samples(None,signal_pars[0],L,'%s/pe_truelike.png' % out_path,x,y)
        #print('Completed true grid PE')

        pe_losses = []         # initialise the losses for plotting
        pe_avg_losses = []
        i = 0
        rms = [1.0,1.0]

        for i in range(pe_iter):
	
            # get random batch from images
            idx = random.sample(np.arange(signal_train_images.shape[0]),pe_batch_size)
            signal_batch_images = signal_train_images[idx]
            signal_batch_images = np.reshape(signal_batch_images, (signal_batch_images.shape[0],signal_batch_images.shape[1],1))
            #signal_batch_images /= np.max(signal_batch_images)
            # add some noise to 50% of batch help generalization to non-ideal input 
            signal_batch_images[:int(signal_batch_images.shape[0]*(1.0/8.0))] += np.random.normal(size=[int(signal_batch_images.shape[0]*(1.0/8.0)), 1024, 1],loc=0.0,scale=np.random.uniform(0,5)) 
	    signal_batch_pars = signal_train_pars[idx]
   
            # normalize pars to be between 0 and 1
            #par1_norm = np.max(signal_batch_pars[:,0])
            #par2_norm = np.max(signal_batch_pars[:,1])
            #signal_batch_pars[:,0] /= par1_norm
            #signal_batch_pars[:,1] *= 50.0
            #signal_batch_pars[:,1] *= np.max(signal_train_pars[:,0])
            

            # train only the signal PE model on the data
            pe_loss = signal_pe.train_on_batch(signal_batch_images,[signal_batch_pars[:,0],signal_batch_pars[:,1]])
            #pe_loss = signal_pe.train_on_batch(signal_batch_images,signal_batch_pars)
	    pe_losses.append(pe_loss)
            pe_avg_losses.append(np.mean([pe_loss[0],pe_loss[1]]))
            
            # save model only if loss is less than previous best loss
            if np.mean([pe_loss[0],pe_loss[1]]) <= np.min(np.array(pe_avg_losses)[:]) and save_models and do_pe and i>5000:
                signal_pe.save('best_models/signal_pe.h5', True)   
            
	    # output status and save images
            if ((i % pe_cadence == 0) & (i>0)):

		# plot loss curves - non-log and log
                plot_losses(pe_losses,'%s/pe_losses.png' % out_path,legend=['PE-GEN'])
                plot_losses(pe_losses,'%s/pe_losses_logscale.png' % out_path,logscale=True,legend=['PE-GEN'])

		# plot true vs predicted values for all training data
                #pe_samples = signal_pe.predict(np.reshape(signal_train_images, (signal_train_images.shape[0],signal_train_images.shape[1],1)))
                idx = random.sample(np.arange(signal_train_images.shape[0]),4000)
                pe_samples = signal_pe.predict(np.reshape(signal_train_images[idx], (signal_train_images[idx].shape[0],signal_train_images[idx].shape[1],1)))

                # rescale output
                #pe_samples[:,0] *= par1_norm
                #pe_samples[:,0] *= 50.0
                #pe_samples[:,1] /= 50.0

                # plot pe accuracy
		#plot_pe_accuracy(signal_train_pars[idx],pe_samples,'%s/pe_accuracy%05d.png' % (out_path,i))

	        # compute RMS difference
                #rms = [np.mean((signal_train_pars[idx][:,k]-pe_samples[:,k])**2) for k in np.arange(2)]
                rms = [np.mean((signal_train_pars[idx][k]-pe_samples[k])**2) for k in np.arange(2)]

                pe_mesg = "%d: [PE loss: %f, acc: %f, RMS: %f,%f]" % (i, pe_loss[0], pe_loss[1], rms[0], rms[1])
                print(pe_mesg)

                #pe_std = [np.mean(np.abs(signal_train_pars[idx][:,0]-pe_samples[:,0].reshape(pe_samples[:,0].shape[0]))),
                #          np.mean(np.abs(signal_train_pars[idx][:,1]-pe_samples[:,1].reshape(pe_samples[:,0].shape[0])))]

                pe_std = [np.mean(np.abs(signal_train_pars[idx][:,0]-pe_samples[0].reshape(pe_samples[0].shape[0]))),
                          np.mean(np.abs(signal_train_pars[idx][:,1]-pe_samples[1].reshape(pe_samples[0].shape[0])))]
 
            
            # if new best network, plot output of PE 
            if np.mean([pe_loss[0],pe_loss[1]]) <= np.min(np.array(pe_avg_losses)[:]) and i>10000:
                # plot loss curves - non-log and log
                plot_losses(pe_losses,'%s/pe_losses.png' % out_path,legend=['PE-GEN'])
                plot_losses(pe_losses,'%s/pe_losses_logscale.png' % out_path,logscale=True,legend=['PE-GEN'])
                idx = random.sample(np.arange(signal_train_images.shape[0]),4000)
                pe_samples = signal_pe.predict(np.reshape(signal_train_images[idx], (signal_train_images[idx].shape[0],signal_train_images[idx].shape[1],1)))
                # plot pe accuracy
                plot_pe_accuracy(signal_train_pars[idx],pe_samples,'%s/pe_accuracy%05d.png' % (out_path,i))

                # compute RMS difference
                #rms = [np.mean((signal_train_pars[idx][:,k]-pe_samples[:,k])**2) for k in np.arange(2)]
                rms = [np.mean((signal_train_pars[idx][k]-pe_samples[k])**2) for k in np.arange(2)]
                pe_mesg = "%d: [PE loss: %f, acc: %f, RMS: %f,%f]" % (i, pe_loss[0], pe_loss[1], rms[0], rms[1])
                print(pe_mesg)


                pe_std = [np.mean(np.abs(signal_train_pars[idx][:,0]-pe_samples[0].reshape(pe_samples[0].shape[0]))),
                          np.mean(np.abs(signal_train_pars[idx][:,1]-pe_samples[1].reshape(pe_samples[0].shape[0])))]

                # test accuracy of parameter estimation against lalinf results during training
                L, x, y = None, None, None
                more_generated_images = pickle.load(open('data/%s_cnn_sanity_check_ts.sav' % event_name)) # TAKE OUT when not doing cnn sanity check!!!
                more_generated_images = np.reshape(more_generated_images, (more_generated_images.shape[0],more_generated_images.shape[1],1))
                pe_samples = signal_pe.predict(more_generated_images)
                #pe_samples[:,1] /= 50.0
                plot_pe_samples(pe_samples,signal_pars,L,out_path,i,x,y,lalinf_pars,pe_std)

    # load old pe model by default
    #signal_pe = keras.models.load_model('signal_pe.h5')
    #signal_pe.load_weights('signal_pe.h5')
    print('Completed CNN PE')

    ################################################
    # LOOP OVER BATCHES ############################

    losses = []		# initailise the losses for plotting 
    beta_score_hist = []
    for i in range(max_iter):

	# get random batch from images, should be real signals
        signal_batch_images = np.array(random.sample(signal_train_images, batch_size))
        #for waveform in signal_batch_images:
        #    plt.plot(waveform)
        #plt.savefig('/home/hunter.gabbard/public_html/CBC/mahoGANy/gw150914_template/latest/template_waveforms.png')
        #plt.close()
        #print(signal_batch_images.shape)
        #exit()

	# first use the generator to make fake images - this is seeded with a size 100 random vector
        noise = np.random.uniform(size=[batch_size*n_noise_real, 100], low=-1.0, high=1.0)
        generated_images = generator.predict(noise)

        # making generated signal a 2d image
        subtracted_signals = noise_signal - generated_images
        subtracted_signals_2d = np.array([])
        for _idx in range(subtracted_signals.shape[0]):
            subtracted_signals_2d = np.append(np.concatenate((generated_images[_idx],subtracted_signals[_idx]), axis=1),subtracted_signals_2d)
        subtracted_signals_2d = subtracted_signals_2d.reshape(subtracted_signals.shape[0],subtracted_signals.shape[1],2)
        

	# make set of real and fake signal mages with labels
        signal_batch_images = np.reshape(signal_batch_images, (signal_batch_images.shape[0], signal_batch_images.shape[1], 1))
        signal_batch_images_noise = np.random.normal(loc=0,scale=1,size=[batch_size*n_noise_real,signal_batch_images.shape[1],1])
        # make signal copy for every noise realization
        signal_batch_images_orig = signal_batch_images
        for _idx in range(n_noise_real-1):
            z = np.copy(signal_batch_images_orig)
            signal_batch_images = np.concatenate((signal_batch_images,z), axis=0)
        signal_batch_images = np.concatenate((signal_batch_images,signal_batch_images_noise), axis=2)
        sX = np.concatenate((signal_batch_images, subtracted_signals_2d))
        sX = np.reshape(sX, (sX.shape[0],sX.shape[1],sX.shape[2],1))
        sy = [1.0] * (batch_size*n_noise_real) + [0.0] * (batch_size*n_noise_real)

        # train only the signal discriminator on the data
        sd_loss = signal_discriminator.train_on_batch(sX, sy)

	# finally train the generator to make images that look like signals
        noise = np.random.uniform(size=[batch_size*n_noise_real, 100], low=-1.0, high=1.0)
        sg_loss = signal_discriminator_on_generator.train_on_batch(noise, [1] * (batch_size*n_noise_real))

        # fill in the loss vector for plotting
        losses.append([sg_loss[0],sg_loss[1],sd_loss[0],sd_loss[1]])

	# output status and save images
	if ((i % cadence == 0) & (i>0)) or (i == max_iter):
            log_mesg = "%d: [sD loss: %f, acc: %f]" % (i, sd_loss[0], sd_loss[1])
	    log_mesg = "%s  [sG loss: %f, acc: %f]" % (log_mesg, sg_loss[0], sg_loss[1])
            print(log_mesg)

            # plot waveform estimates
            lalinf_prod_waveforms = pickle.load(open('data/cnn_sanity_check_ts.sav'))
            plot_waveform_est(signal_image,noise_signal,generated_images,out_path,i)
            if i == cadence:
                plot_waveform_est(signal_image,noise_signal,lalinf_prod_waveforms,out_path,i,plot_lalinf_wvf=True)


            """
            # plot mean and standard-dev of generated images from last batch
            tmp = []
            tmp.append(renorm(noise_signal).reshape(n_pix,n_pix,n_colors))
            tmp.append(signal_image.reshape(n_pix,n_pix,n_colors))
	    tmp.append(noise_image.reshape(n_pix,n_pix,n_colors))
            tmp.append(np.mean(generated_images,axis=0).reshape(n_pix,n_pix,n_colors))
            tmp.append(np.std(generated_images,axis=0).reshape(n_pix,n_pix,n_colors))
	    tmp.append(renorm(noise_signal.reshape(n_pix,n_pix,n_colors) - np.mean(generated_images,axis=0)).reshape(n_pix,n_pix,n_colors))
            tmp = np.array(tmp).reshape(6,n_pix,n_pix,n_colors)
	    ms_out = combine_images(tmp,cols=3,rows=3,randomize=False)
            ms_out.save('%s/mean_std_res_signal%05d.png' % (out_path,i))
            """

            # plot loss curves - noMlog and log
            plot_losses(losses,'%s/losses.png' % out_path,legend=['S-GEN','S-DIS'])
            plot_losses(losses,'%s/losses_logscale.png' % out_path,logscale=True,legend=['S-GEN','S-DIS'])

            
	    # plot posterior samples
            
            if do_pe:
                L, x, y = None, None, None
                # first use the generator to make MANY fake images
        	noise = np.random.uniform(size=[4000, 100], low=-1.0, high=1.0)
        	more_generated_images = generator.predict(noise)
                #more_generated_images = pickle.load(open('data/cnn_sanity_check_ts.sav')) + np.random.uniform(size=[3800,1024],low=0,high=0.01)
                #test_imgs = pickle.load(open('data/cnn_sanity_check_ts.sav','rb'))
                #test_imgs = test_imgs.reshape(test_imgs.shape[0],test_imgs.shape[1],1)
                #for i in range(more_generated_images.shape[0]):
                #    plt.plot(more_generated_images[i],alpha=0.5)
                #plt.savefig('%s/test_waveform.png' % out_path)
                #plt.close()
                #exit()

                #more_generated_images = np.reshape(more_generated_images, (more_generated_images.shape[0],more_generated_images.shape[1],1))
                pe_samples = signal_pe.predict(np.reshape(more_generated_images, (more_generated_images.shape[0],more_generated_images.shape[1],1)))
                #idx = random.sample(np.arange(signal_train_images.shape[0]),4000)
                #pe_std = [np.mean(np.abs(signal_train_pars[idx][:,0]-pe_samples[0].reshape(pe_samples[0].shape[0]))),
                #          np.mean(np.abs(signal_train_pars[idx][:,1]-pe_samples[1].reshape(pe_samples[0].shape[0])))]
                #pe_samples[:,1] /= 50.0
                pe_std = [0.02185649964844209, 0.005701401364171313]
                var_par1 = np.var(pe_samples[0])
                var_par2 = np.var(pe_samples[1])
                #try:
                if var_par1 != 0 and var_par2 != 0:
                    beta_score_hist.append([plot_pe_samples(pe_samples,signal_pars,L,out_path,i,x,y,lalinf_pars,pe_std)])
                    plt.plot(np.linspace(cadence,i,len(beta_score_hist)),beta_score_hist)
                    plt.savefig('%s/latest/beta_hist.png' % out_path)
                    plt.close()
                #except:
                #    print('Skipping algebra error')
                f = open('gan_pe_samples.sav', 'wb')
                cPickle.dump(pe_samples, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()
                f = open('gan_pe_waveforms.sav', 'wb')
                cPickle.dump(more_generated_images, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()
                print('{}: saved GAN PE samples data to file'.format(time.asctime()))
                print('{}: Completed PE plotting routine!'.format(time.asctime()))
            

	    # save trained models
            
            if save_models:
	        generator.save_weights('generator.h5', True)
                signal_discriminator.save_weights('discriminator.h5', True)
                signal_discriminator_on_generator.save_weights('signal_dis_on_gen.h5', True)
            

            # save posterior samples
            f = open('GAN_posterior_samples/posterior_samples_%05d.sav' % i, 'wb')
            pickle.dump(pe_samples, f)
            f.close()
            print '{}: saved posterior data to file'.format(time.asctime())

main()
