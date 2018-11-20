# Copyright (C) 2018  Hunter Gabbard, Chris Messenger
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

'''
This is a script which takes as input a trainig set composed of GW templates and then uses those
to return posterior estimates on any given GW waveform burried in noise. This script assumes 
that you are rusing Python 2.7
'''

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
from scipy import stats
from scipy.signal import butter, lfilter
from scipy.signal import freqs

cuda_dev = "3" # define GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cuda_dev

# allow GPU memory usage to grow as needed
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

# define some global params
n_pix = 1024	               # time series size
n_sig = 1.0                    # the noise standard deviation (default is 1)
batch_size = 8                 # the GAN batch size (twice this when testing discriminator)
pe_batch_size = 64             # The CNN batch size
max_iter = 500*1000 	       # the maximum number of steps or epochs for GAN waveform network
pe_iter = 5*100000             # the maximum number of steps or epochs for CNN pe network 
cadence = 100		       # the cadence of priting/saving output for GAN
save_models = True	       # save the generator and discriminator models
do_pe = True		       # perform parameter estimation 
pe_cadence = 1000  	       # the cadence of PE print satements and status plots
pe_grain = 95                  # fineness of pe posterior grid (leave this alone plz)
npar = 2 		       # the number of parameters to estimate TODO: make ability to increase this number
N_VIEWED = 25                  # number of allowed samples to view when plotting GAN estimated waveforms
chi_loss = False               # use chisquared loss function in GAN waveform generation
lr = 9e-5                      # learning rate for all networks
GW150914 = True                # use lalinference produced GW150914 waveform as event 
gw150914_tmp = True            # use gw150914-like template waveform as event
do_old_model = False           # use previously saved model for GAN
do_contours = True             # plot credibility contours on pe estimates
do_only_old_pe_model = True    # run previously saved pe model only
retrain_pe_mod = False         # retrain an old parameter estimation CNN model
comb_pe_model = False          # if true: use single NN for PE. if False: use multiple NNs for PE estimation
contour_cadence = 100          # the cadence of making contour plot outputs (lower cadence takes longer training)
n_noise_real = 1               # number of noise realizations per training sample (default is 1)
event_name = 'gw150914'        # event name
template_dir = 'templates/'    # location of training templates directory
training_num = 50000           # number of samples to use during training of either GAN or CNN 
tag = '_srate-1024hz_oversamp' # special tag for some files used
cnn_sanity_check_file = 'gw150914_cnn_sanity_check_ts_mass-time-vary_srate-1024hz.sav' # name of file used for checking absolute best performance of CNN
cnn_noise_frac = 1.0/8.0       # fraction of training set which are noisy to be used in CNN training

# load in lalinference m1 and m2 parameters
pickle_lalinf_pars = open("data/%s_mc_q_lalinf_post_srate-1024hz.sav" % (event_name))
lalinf_pars = pickle.load(pickle_lalinf_pars)

# the locations of signal files and output directory
if gw150914_tmp:
    out_path = '/home/hunter.gabbard/public_html/CBC/mahoGANy/%s_template' % event_name 

# define new output folder if not running on an event from lalinference
# this will be for running over my own generated templates if applicable
if not GW150914 and not gw150914_tmp:
    out_path = '/home/hunter.gabbard/public_html/CBC/mahoGANy/rand_bbh_results/cuda_dev_%s' % cuda_dev

# define bbh parameters class
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
    """Uses a chisquare loss instead of standard keras loss 
       function.

    Parameters
    ----------
    yTrue:
        labels for training set. Will be either zero or one
    yPred: 
        predictions from discriminator

    Returns
    -------
    loss: Keras function
        keras chisquared loss function
    """
    return K.sum( K.square(yTrue - yPred)/(n_sig**2), axis=-1)

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
        # computes the difference between the meausured image and the generated signal
        # and returns it as a Keras 2D object where top is waveform and bottom is noise
        diff = self.const - x
        return K.stack([x,diff], axis=2)

    def compute_output_shape(self, input_shape):
        # the output shape
        return (input_shape[0],n_pix,2,1)

def data_subtraction_model(noise_signal,npix):
    """
    This model simply applies the signal subtraction from the measured image
    You must pass it the measured image

    Parameters
    ----------
    noise_signal:
        event to do PE over (burried in noise)
    npix: 
        sampling frequency of signal

    Returns
    -------
    model: Keras model
        model which includes waveform subtraction layer
    """
    model = Sequential()
    model.add(MyLayer(noise_signal,input_shape=(npix,1)))
   
    return model

def generator_model():
    """
    The generator that should train itself to generate noise free signals

    Returns
    -------
    model: Keras model
        model which is the generator portion of the GAN
    """
    model = Sequential()       # type of model
    act = 'tanh'               # activation function for hidden layers
    momentum = 0.99            # momentum used in batch normalization
    drate = 0.2                # drop out rate percentage in hidden layers
    padding = 'same'           # padding used in hidden layers
    weights = 'glorot_uniform' # weight initialization
    alpha = 0.5                # slope for negative inputs to leaky relu activation function
    filtsize = 5 # 10 is best  # filter size for hidden layers
    num_lays = 5               # number of hidden layers in model
    batchnorm = True           # if True: use batch normalization. if False: do not use
    
    # the first dense layer converts the input (100 random numbers) into
    # 512 numbers. Upsampling applied in later layers
    model.add(Dense(256 * 1 * int(n_pix/2), kernel_initializer=weights, input_shape=(100,)))
    if batchnorm: model.add(BatchNormalization(momentum=momentum))
    if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
    elif act == 'prelu': model.add(PReLU())
    else: model.add(Activation(act))
    model.add(Dropout(drate))

    # reshape input to correct size
    model.add(Reshape((int(n_pix/2), 256)))

    # add requested number of hidden layers
    # TODO: automate this such that any number of layers are allowed
    for i in range(num_lays):
        i += 1
        if i == 1:
            model.add(UpSampling1D(size=2)) # perform upsampling to increase size of output by factor of 2
            model.add(Conv1D(64, filtsize, kernel_initializer=weights, strides=2, padding=padding)) # stration decreases output size by factor of 2
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
    
        elif i == 2:
            model.add(UpSampling1D(size=2))
            model.add(Conv1D(128, filtsize, kernel_initializer=weights, strides=1, padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))

        elif i == 3:
            model.add(Conv1D(256, filtsize, kernel_initializer=weights, strides=1, padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))

        elif i == 4:
            model.add(Conv1D(512, filtsize, kernel_initializer=weights, strides=1, padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))

        elif i == 5:
            model.add(Conv1D(1024, filtsize, kernel_initializer=weights, strides=1, padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))

    
    # the output shape should be (n_pix x 1)
    model.add(Conv1D(1, filtsize, padding=padding))
    model.add(Activation('linear'))

    return model

def signal_pe_model():
    """ The CNN PE network that learns how to convert
        noise-free time series into point estimates on parameters

    Returns
    -------
    model: Keras model
        model which is the CNN parameter point estimator
    """ 

    # if use wants to use one NN for multiple parameter point estimation
    if comb_pe_model:    
    	model = Sequential() # type of model to use
    	act = 'prelu'       # activation function to use
    	momentum = 0.9
        filtsize=5

   	model.add(Conv1D(64, filtsize, strides=2, input_shape=(n_pix,1), padding='valid'))
        if act=='prelu': model.add(PReLU())
        elif act=='leakyrelu': model.add(LeakyReLU(alpha=0.2))
   	else: model.add(Activation(act))
        if batchnorm==True: model.add(BatchNormalization(momentum=momentum))
  	model.add(Dropout(0.5))

    	# the next layer is another 2D convolution with 128 neurons and a 5x5
    	# filter. More 2x2 max pooling and a tanh activation. The output is flattened
    	# for input to the next dense layer
    	model.add(Conv1D(128, filtsize, strides=2))
        if act=='prelu': model.add(PReLU())
        elif act=='leakyrelu': model.add(LeakyReLU(alpha=0.2))
        else: model.add(Activation(act))
        if batchnorm==True: model.add(BatchNormalization(momentum=momentum))

    	model.add(Conv1D(256, filtsize, strides=2))
        if act=='prelu': model.add(PReLU())
        elif act=='leakyrelu': model.add(LeakyReLU(alpha=0.2))
        else: model.add(Activation(act))
        if batchnorm==True: model.add(BatchNormalization(momentum=momentum))

    	model.add(Conv1D(512, filtsize, strides=2))
        if act=='prelu': model.add(PReLU())
        elif act=='leakyrelu': model.add(LeakyReLU(alpha=0.2))
        else: model.add(Activation(act))
        if batchnorm==True: model.add(BatchNormalization(momentum=momentum))

    	model.add(Flatten())

    	# we now use a dense layer with 1024 outputs and a tanh activation
    	model.add(Dense(1024))
        if act=='prelu': model.add(PReLU())
        elif act=='leakyrelu': model.add(LeakyReLU(alpha=0.2))
        else: model.add(Activation(act))

    	# the final dense layer has a linear activation and 2 outputs
    	# we are currently testing with only 2 outputs - can be generalised
    	model.add(Dense(2))
    	model.add(Activation('relu'))
    	#model.add(PReLU())   

    if not comb_pe_model:
    	inputs = Input(shape=(n_pix,1))
    	act = 'relu'
    	drate = 0.3

        # define chirp mass branch of neural network
    	mc_branch = Conv1D(64, 5, strides=2, padding='same')(inputs)
    	mc_branch = Activation(act)(mc_branch)

    	mc_branch = Conv1D(128, 5, strides=2)(mc_branch)
    	mc_branch = Activation(act)(mc_branch)

    	mc_branch = Conv1D(256, 5, strides=2)(mc_branch)
    	mc_branch = Activation(act)(mc_branch)

    	mc_branch = Conv1D(512, 5, strides=2)(mc_branch)
    	mc_branch = Activation(act)(mc_branch)

    	mc_branch = Flatten()(mc_branch)


   	mc_branch = Dense(1)(mc_branch)
    	mc_branch = Activation('relu')(mc_branch)
    
        # define inverse mass ratio branch of neural network
    	act = 'relu' 
    	q_branch = Conv1D(64, 5, strides=1, padding='same')(inputs)
    	q_branch = Activation(act)(q_branch)

    	q_branch = Conv1D(128, 5, strides=1)(q_branch)
    	q_branch = Activation(act)(q_branch)

    	q_branch = Conv1D(256, 5, strides=1)(q_branch)
    	q_branch = Activation(act)(q_branch)

    	q_branch = Conv1D(512, 5, strides=2)(q_branch)
    	q_branch = Activation(act)(q_branch)
    
    	q_branch = Conv1D(1024, 5, strides=2)(q_branch)
    	q_branch = Activation(act)(q_branch)

    	q_branch = Flatten()(q_branch)

    	q_branch = Dense(1)(q_branch)
    	q_branch = ReLU(max_value=1.0)(q_branch)
    	model = Model(
            inputs=inputs,
            outputs=[mc_branch, q_branch],
            name="pe net")
    
    return model

def signal_discriminator_model():
    """ The discriminator that should train itself to recognise generated signals
    from real signals

    Returns
    -------
    model: Keras model
        model which is the discriminator of the GAN
    """
    #act='tanh'
    momentum=0.99              # momentum term in batch normalization
    weights = 'glorot_uniform' # weight initialization for hidden layers
    drate = 0.4                # dropout rate for dropout layers
    act = 'leakyrelu'          # activation function for hidden layers
    alpha = 0.2                # slope for negative input values in leaky relu act func
    padding = 'same'           # padding for hidden layers
    num_lays = 2               # number of hidden layers
    batchnorm = False          # if True: use batch normalization in hidden layers. if False: do not use
    maxpool = False            # if True: use max pooling in hidden layers. if False: do not use

    # so 5x2 gives best waveform reconstruction, but not best pe results? Kinda weird ...
    filtsize = (5,5)           # filter size of convolutional neurons
    n_neuron_scale = 4         # scale by neurons are increased from hidden layer to hidden layer

    model = Sequential()

    # iterate over requested number of hidden layers
    # TODO: automate to allow N number of hidden layers
    for i in range(num_lays):
        i += 1
        if i == 1:
            model.add(Conv2D(64 * n_neuron_scale, filtsize, kernel_initializer=weights, input_shape=(n_pix,2,1), strides=(2,1), padding=padding))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            if maxpool: model.add(MaxPooling2D(pool_size=(2,1)))
    
        elif i == 2:
            model.add(Conv2D(128*n_neuron_scale, filtsize, kernel_initializer=weights, strides=(2,1), padding=padding))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            model.add(Dropout(drate))
            if maxpool: model.add(MaxPooling2D(pool_size=(2,1)))

        elif i == 3:
            model.add(Conv2D(256, filtsize, kernel_initializer=weights, strides=(1,1), padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            if maxpool: model.add(MaxPooling2D(pool_size=(2,1)))

        elif i == 4:
            model.add(Conv2D(512, filtsize, kernel_initializer=weights, strides=(1,1), padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            if maxpool: model.add(MaxPooling2D(pool_size=(2,1)))

        elif i == 5:
            model.add(Conv2D(1024, filtsize, kernel_initializer=weights, strides=(1,1), padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            if maxpool: model.add(MaxPooling2D(pool_size=(2,1)))

        elif i == 6:
            model.add(Conv2D(1024, filtsize, kernel_initializer=weights, strides=(1,1), padding=padding))
            if batchnorm: model.add(BatchNormalization(momentum=momentum))
            if act == 'leakyrelu': model.add(LeakyReLU(alpha=alpha))
            elif act == 'prelu': model.add(PReLU())
            else: model.add(Activation(act))
            model.add(Dropout(drate))
            if maxpool: model.add(MaxPooling2D(pool_size=(2,1)))    

    model.add(Flatten())

    # the final dense layer has a sigmoid activation and a single output
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
 
    return model

def generator_after_subtracting_noise(generator, data_subtraction):
    """ This is the sequence used for training the generator to make signals, that 
    when subtracted from the measured data, give Gaussian noise of zero mean and 
    expected variance

    Parameters
    ----------
    generator: Keras model
        generator network
    data_subtraction: 
        waveform subtraction MyLayer keras layer

    Returns
    -------
    model: Keras model
    """
    model = Sequential()
    model.add(generator)		# the trainable parameters are in this model
    model.add(data_subtraction)		# there are no parameters ro train in this model
    return model

def generator_containing_signal_discriminator(generator, signal_discriminator):
    """ This is the sequence used to train the signal generator such that the signals
    it generates are consistent with the training signals seen by the disciminator

    Parameters
    ----------
    generator: Keras model
        generator network
    signal_discriminator: 
        discriminator network

    Returns
    -------
    model: Keras model
    """
    model = Sequential()
    model.add(generator)		# the trainable parameters are in this model
    model.add(signal_discriminator)	# the discriminator has been set to not be trainable
    return model

def plot_losses(losses,filename,logscale=False,legend=None,chi_loss=chi_loss):
    """ Make loss and accuracy plots and output to file.
    Plot with x and y log-axes is desired

    Parameters
    ----------
    losses: list
        list containing history of network loss and accuracy values
    filename: string
        string which specifies location of output directory and filename
    logscale: boolean
        if True: use logscale in plots, if False: do not use
    legend: boolean
        if True: apply legend, if False: do not
    chi_loss: boolean
        if True: we take into account fact that we will only have two loss curves 
    """
    # plot losses
    fig = plt.figure()
    losses = np.array(losses)
    ax1 = fig.add_subplot(211)	
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
        ax1.set_yscale("log", nonposy='clip')
    plt.savefig(filename)
    plt.close('all')

def plot_pe_accuracy(true_pars,est_pars,outfile):
    """ Plots the true vs the estimated parameters from the CNN PE training
    Change est_pars to est_pars[0] or est_pars[1] if doing 
    multiple pe network.

    Parameters
    ----------
    true_pars:
        array containing true values of PE sample parameters
    est_pars:
        array containing CNN estimated values of PE sample parameters
    outfile:
        path and filename of output image file
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(121,aspect=1.0)
    if comb_pe_model: ax1.plot(true_pars[:,0],est_pars[:,0],'.b', markersize=0.5)
    if not comb_pe_model: ax1.plot(true_pars[:,0],est_pars[0],'.b', markersize=0.5)
    ax1.plot([0,np.max(true_pars[:,0])],[0,np.max(true_pars[:,0])],'--k')
    ax1.set_xlabel(r'True parameter 1')
    ax1.set_ylabel(r'Estimated parameter 1')
    ax2 = fig.add_subplot(122,aspect=1.0)
    if comb_pe_model: ax2.plot(true_pars[:,1],est_pars[:,1],'.b', markersize=0.5)
    if not comb_pe_model: ax2.plot(true_pars[:,1],est_pars[1],'.b', markersize=0.5)
    ax2.plot([0,np.max(true_pars[:,1])],[0,np.max(true_pars[:,1])],'--k')
    ax2.set_xlabel(r'True parameter 2')
    ax2.set_ylabel(r'Estimated parameter 2')
    plt.savefig(outfile)
    plt.savefig('%s/latest/pe_accuracy.png' % out_path)
    plt.close('all')

def plot_pe_samples(pe_samples,truth,like,outfile,index,x,y,lalinf_dist=None,pe_std=None):
    """ Makes scatter plot of samples estimated from PE model with contours

    Parameters
    ----------
    pe_samples: array
        array containing the predicted pe estimates from the CNN
    truth: list
        list containing the true point estimate values of the parameters (taken from paper)
    like: function
        likelihood function for contour calculations. TODO: Should remove this. Not used.
    outfile: string
        output directory of files
    index: 
        current training epoch iteration
    x:
        another dummy variable. need to remove  
    y:
        another dummy variable. need to remove
    lalinf_dist: array
        array of lalinference posterior estimate results to compare CNN with       
    pe_std: list
        error of the CNN network for each parameter  

    Returns
    -------
    beta_score: scalar
        scalar value which ranges from 0-1 where 1 is 100% overlap of CNN samples
        with lalinference samples and 0 is 0% overlap. The higher, the better.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(223)
   
    # plot pe samples
    if pe_samples is not None:
        if comb_pe_model: ax1.plot(pe_samples[:,0],pe_samples[:,1],'.r',markersize=0.8)
        if not comb_pe_model: ax1.plot(pe_samples[0],pe_samples[1],'.r',markersize=0.8)

        # plot contours for generated samples
        if do_contours and (index>0): # and ((index % contour_cadence == 0) 
            if comb_pe_model: contour_y = np.reshape(pe_samples[:,1], (pe_samples[:,1].shape[0]))
            if comb_pe_model: contour_x = np.reshape(pe_samples[:,0], (pe_samples[:,0].shape[0]))
            if not comb_pe_model: contour_y = np.reshape(pe_samples[1], (pe_samples[1].shape[0]))
            if not comb_pe_model: contour_x = np.reshape(pe_samples[0], (pe_samples[0].shape[0]))
            contour_dataset = np.array([contour_x,contour_y])
            kernel_cnn = make_contour_plot(ax1,contour_x,contour_y,contour_dataset,'red',flip=False)

    # plot contours of lalinf distribution
    if lalinf_dist is not None:

        # plot lalinference parameters
        ax1.plot(lalinf_dist[0][:],lalinf_dist[1][:],'.b', markersize=0.8)

        # plot lalinference parameter contours
        if do_contours and (index>0): # and ((index % contour_cadence == 0)
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
    if comb_pe_model: ax2.hist(pe_samples[:,0], bins=100, alpha=0.5, normed=True)
    if not comb_pe_model: ax2.hist(pe_samples[0], bins=100, alpha=0.5, normed=True)
    ax2.hist(lalinf_dist[0][:],bins=100, alpha=0.5, normed=True)
    ax2.set_xticks([])
    if comb_pe_model: ax2.hist(pe_samples[:,0], bins=100, alpha=0.5, normed=True)
    if not comb_pe_model: ax3.hist(pe_samples[1], bins=100, orientation=u'horizontal', alpha=0.5, normed=True)
    ax3.hist(lalinf_dist[1][:],bins=100,orientation=u'horizontal', alpha=0.5, normed=True) 
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
    """ Module used to make contour plots in pe scatter plots.

    Parameters
    ----------
    ax: matplotlib figure
        a matplotlib figure instance
    x: 1D numpy array
        pe sample parameters for x-axis
    y: 1D numpy array
        pe sample parameters for y-axis
    dataset: 2D numpy array
        array containing both parameter estimates
    color:
        color of contours in plot
    flip:
        if True: transpose parameter estimates array. if False: do not transpose parameter estimates
        TODO: This is not used, so should remove

    Returns
    -------
    kernel: scipy kernel
        gaussian kde of the input dataset

    """
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

def set_trainable(model, trainable):
    """ Allows us to switch off models in sequences as trainable

    Parameters
    ----------
    model: keras model object
        this is the keras model to be set trainability
    trainable: boolean
        if True: set to be trainable. if False: set to be non-trainable
    """
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def overlap_tests(pred_samp,lalinf_samp,true_vals,kernel_cnn,kernel_lalinf):
    """ Perform Anderson-Darling, K-S, and overlap tests
    to get quantifiable values for accuracy of GAN
    PE method

    Parameters
    ----------
    pred_samp: numpy array
        predicted PE samples from CNN
    lalinf_samp: numpy array
        predicted PE samples from lalinference
    true_vals:
        true scalar point values for parameters to be estimated (taken from GW event paper)
    kernel_cnn: scipy kde instance
        gaussian kde of CNN results
    kernel_lalinf: scipy kde instance
        gaussian kde of lalinference results

    Returns
    -------
    ks_score:
        k-s test score
    ad_score:
        anderson-darling score
    beta_score:
        overlap score. used to determine goodness of CNN PE estimates
    """

    # do k-s test
    if comb_pe_model: ks_mc_score = ks_2samp(pred_samp[:,0].reshape(pred_samp[:,0].shape[0],),lalinf_samp[0][:])
    if comb_pe_model: ks_q_score = ks_2samp(pred_samp[:,1].reshape(pred_samp[:,1].shape[0],),lalinf_samp[1][:])
    if not comb_pe_model: ks_mc_score = ks_2samp(pred_samp[0].reshape(pred_samp[0].shape[0],),lalinf_samp[0][:])
    if not comb_pe_model: ks_q_score = ks_2samp(pred_samp[1].reshape(pred_samp[1].shape[0],),lalinf_samp[1][:])
    ks_score = np.array([ks_mc_score,ks_q_score])

    # do anderson-darling test
    if comb_pe_model: ad_mc_score = anderson_ksamp([pred_samp[:,0].reshape(pred_samp[:,0].shape[0],),lalinf_samp[0][:]])
    if comb_pe_model: ad_q_score = anderson_ksamp([pred_samp[:,1].reshape(pred_samp[:,1].shape[0],),lalinf_samp[1][:]])
    if not comb_pe_model: ad_mc_score = anderson_ksamp([pred_samp[0].reshape(pred_samp[0].shape[0],),lalinf_samp[0][:]])
    if not comb_pe_model: ad_q_score = anderson_ksamp([pred_samp[1].reshape(pred_samp[1].shape[0],),lalinf_samp[1][:]])
    ad_score = [ad_mc_score,ad_q_score]

    # compute overlap statistic
    if comb_pe_model: comb_mc = np.concatenate((pred_samp[:,0].reshape(pred_samp[:,0].shape[0],1),lalinf_samp[0][:].reshape(lalinf_samp[0][:].shape[0],1)))
    if comb_pe_model: comb_q = np.concatenate((pred_samp[:,1].reshape(pred_samp[:,1].shape[0],1),lalinf_samp[1][:].reshape(lalinf_samp[1][:].shape[0],1)))
    if not comb_pe_model: comb_mc = np.concatenate((pred_samp[0],lalinf_samp[0][:].reshape(lalinf_samp[0][:].shape[0],1)))
    if not comb_pe_model: comb_q = np.concatenate((pred_samp[1],lalinf_samp[1][:].reshape(lalinf_samp[1][:].shape[0],1)))
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

def plot_waveform_est(signal_image,noise_signal,generated_images,out_path,i,plot_lalinf_wvf=False,zoom=False):
    """ plotes the estimated wavforms, residuals of signal+noise minus estimated waveforms, 
    and the signal+noise and the signal without noise.

    Parameters
    ----------
    signal_image: numpy array
        timeseries containing signal to do PE on without noise
    noise_signal: numpy array
        timeseries containing signal to do PE on with noise
    generated_images: numpy array
        waveform estimates from generative adversarial network
    out_path: string
        location of output directory for figures to be saved
    i:
        index of current iteration of training epoch
    plot_lalinf_wvf: boolean
        if True: plot waveforms derrived from lalinference estimated parameters. if False:
        plot the GAN estimated waveforms.
    zoom:
        if True: plot a zoomed-in version of the waveform plots        
    """
    # plot original waveform
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
    ax = signal_image
    ax1.plot(ax, color='cyan', alpha=0.5, linewidth=0.5)
    ax1.plot(noise_signal, color='green', alpha=0.35, linewidth=0.5)
    if zoom==True: ax1.set_xlim((450,550))
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
    if zoom==True: ax2.set_xlim((450,550))

    # plot residuals - generated images subtracted from the measured image
    # the first image is the true noise realisation
    residuals = np.transpose(np.transpose(noise_signal)-gen_sig)
    ax3.plot((residuals[:,0]), color='black', linewidth=0.5)
    ax3.plot((residuals), color='red', alpha=0.25, linewidth=0.5)
    if zoom==True: ax3.set_xlim((450,550))
    #ax3.plot((noise_signal - ax), color='black', alpha=0.5, linewidth=0.5)

    #ax3.set_title('Residuals')
    ax3.set(xlabel='Time')
    # save waveforms plot
    if not plot_lalinf_wvf and zoom==False:
        plt.savefig('%s/waveform_results%05d.png' % (out_path,i), dpi=500)
        plt.savefig('%s/latest/most_recent_waveform.png' % out_path, dpi=400)
        print('Completed waveform plotting routine!')
    elif plot_lalinf_wvf and zoom==False:
        plt.savefig('%s/latest/most_recent_lalinf_waveform.png' % out_path, dpi=400)
        print('Completed lalinf waveform plotting routine!')
    elif plot_lalinf_wvf and zoom==True:
        plt.savefig('%s/latest/most_recent_zoomed_lalinf_waveform.png' % out_path, dpi=400)
        print('Completed lalinf waveform plotting routine!')
    elif not plot_lalinf_wvf and zoom==True:
        plt.savefig('%s/waveform_zoomed_results%05d.png' % (out_path,i), dpi=500)
        plt.savefig('%s/latest/most_recent_zoomed_waveform.png' % out_path, dpi=400)
        print('Completed waveform plotting routine!')

    plt.close("all")

def main():

    ################################################
    # READ/GENERATE DATA ###########################
    ################################################

    # setup output directory - make sure it exists
    os.system('mkdir -p %s' % out_path) 

    # load first time series / pars template pickle file
    file_idx_list = []
    pickle_ts = open("%s%s_ts_0_%sSamp%s.sav" % (template_dir,event_name,training_num,tag),"rb")
    ts = pickle.load(pickle_ts)
    pickle_par = open("%s%s_params_0_%sSamp%s.sav" % (template_dir,event_name,training_num,tag),"rb")
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

    # iterate over all other data files and load them
    for idx in file_idx_list:
        pickle_ts = open("%s_ts_%s_%sSamp%s.sav" % (template_dir,str(idx),training_num,tag),"rb")
        ts_new = pickle.load(pickle_ts)
        ts = np.vstack((ts,ts_new[0]))

        # load corresponding parameters template pickle file
        pickle_par = open("%s_params_%s_%sSamp%s.sav" % (template_dir,str(idx),training_num,tag),"rb")
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

    # transform training parameters into a numpy array rather than class instance
    signal_train_pars = []
    for k in par:
        signal_train_pars.append([k.mc,(k.m2/k.m1)])

    signal_train_pars = np.array(signal_train_pars)

    # randomly extract single image as the true signal if 
    # template not otherwise specified
    if not GW150914 and not gw150914_tmp:
        i = np.random.randint(0,signal_train_images.shape[0],size=1)
        signal_image = signal_train_images[i,:]
        signal_train_images = np.delete(signal_train_images,i,axis=0)


    # pick lalinference event time series as true signal
    # pars will be default params in function
    if GW150914:
        pickle_gw150914 = open("data/%s0%s.sav" % (event_name,tag),"rb")
        noise_signal = np.reshape(pickle.load(pickle_gw150914) * 817.98,(n_pix,1)) # 817.98
        signal_image = pickle.load(open("data/%s_data%s.pkl" % (event_name,tag),"rb")) * 817.98 # 1079.22
        noise_image = np.random.normal(0, n_sig, size=[1, signal_image.shape[0]])
        gan_noise_signal = np.transpose(signal_image + noise_image)

        # delete last template in training set. this is done because the
        # last template is always a reproduction of the event to be trained over
        # this is coded in the templated bank generation process
        signal_train_images = np.delete(signal_train_images,-1,axis=0)

    # pick event-like template from training set as true signal
    if gw150914_tmp and not GW150914:
        signal_image = signal_train_images[-1,:]
        signal_train_images = np.delete(signal_train_images,-1,axis=0)

    # if choosing to run over random template event, set template pars
    # will only be used if setting the true signal to be a random
    # template in training set
    if do_pe and not GW150914 and not gw150914_tmp:
        signal_pars = signal_train_pars[i,:][0]
        signal_train_pars = np.delete(signal_train_pars,i,axis=0)    

    # use last template in training samples as true signal params
    # training sets are made such that the true template is 
    # always included as the last template
    if do_pe and gw150914_tmp:
        signal_pars = signal_train_pars[-1,:]
        signal_train_pars = np.delete(signal_train_pars,-1,axis=0)

    # if setting true event to be from lalinference run THIS IS DEFAULT
    # set true signal pars
    if GW150914:
        # set signal_pars mc and q
        # parameters taken from event paper
        # these are only used in pe results scatter plot cross hairs
        if not gw150914_tmp:
            signal_pars = [30.0,0.79]

    # if choosing a random template as signal to do PE on
    # add that signal to noise
    if not GW150914:
        # Generate single noise image
        signal_image = np.reshape(signal_image, (1, 512))
        noise_image = np.random.normal(0, n_sig, size=[1, signal_image.shape[1]])

        # combine signal and noise - this is the measured data i.e., h(t)
        noise_signal = np.transpose(signal_image + noise_image)
        signal_image = signal_image[0]

    # plots what waveform we are going to be doing PE on
    # mainly just for debugging purposes
    plt.plot(signal_image)
    plt.plot(noise_signal, alpha=0.5)
    plt.savefig('%s/input_waveform.png' % out_path)
    plt.close()

    ################################################
    # SETUP MODELS #################################
    ################################################  
 
    # initialise all models
    generator = generator_model()
    signal_discriminator = signal_discriminator_model()
    data_subtraction = data_subtraction_model(noise_signal,n_pix)
    if do_pe:
        signal_pe = signal_pe_model()    

    """
    setup generator training for when we subtract from the 
    measured data and expect Gaussian residuals.
    """
    # setup extra layer on generator
    data_subtraction_on_generator = generator_after_subtracting_noise(generator, data_subtraction)
    data_subtraction_on_generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    # setup generator training when we pass the output to the signal discriminator
    signal_discriminator_on_generator = generator_containing_signal_discriminator(data_subtraction_on_generator, signal_discriminator)
    set_trainable(signal_discriminator, False)	# set the discriminator as not trainable for this step
    if not chi_loss:
        signal_discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])
    elif chi_loss:
        signal_discriminator_on_generator.compile(loss=chisquare_Loss, optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    # setup training on signal discriminator model
    # This uses a binary cross entropy loss since we are just 
    # discriminating between real and fake signals
    set_trainable(signal_discriminator, True)	# set it back to being trainable
    signal_discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    # compile CNN PE point estimater neural network
    if do_pe:
        signal_pe.compile(loss='mean_squared_error', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    # print the model summaries
    print(generator.summary())
    print(signal_discriminator_on_generator.summary())
    print(signal_discriminator.summary())
    if do_pe:
        print(signal_pe.summary())

    ################################################
    # DO CNN PARAMETER ESTIMATION TRAINING #########
    ################################################

    # if True: load a pre-trained NN for each NN
    if do_old_model:
        if do_pe:
            signal_pe = keras.models.load_model('best_models/signal_pe.h5')
        signal_discriminator.load_weights('discriminator.h5')
        signal_discriminator_on_generator.load_weights('signal_dis_on_gen.h5')
        generator.load_weights('generator.h5')

    # if true: load a pre-trained NN model ONLY for the CNN PE point estimater
    if do_only_old_pe_model:
        signal_pe = keras.models.load_model('best_models/signal_pe.h5')

    # train either an untrained CNN or an old pre-trained CNN to do point estimate parameter estimation
    if do_pe and not do_only_old_pe_model or retrain_pe_mod:

        pe_losses = []         # initialise the losses for plotting
        pe_avg_losses = []
        i = 0
        rms = [1.0,1.0]        # initialize values for root mean squared estimates of CNN goodness

        # iterate over desired number of CNN training iterations
        for i in range(pe_iter):
	
            # get random batch from images
            idx = random.sample(np.arange(signal_train_images.shape[0]),pe_batch_size)
            signal_batch_images = signal_train_images[idx]
            signal_batch_images = np.reshape(signal_batch_images, (signal_batch_images.shape[0],signal_batch_images.shape[1],1))

            # add some noise to fraction of batch to help generalization to non-ideal input 
            signal_batch_images[:int(signal_batch_images.shape[0]*(cnn_noise_frac))] += np.random.normal(size=[int(signal_batch_images.shape[0]*(cnn_noise_frac)), 1024, 1],loc=0.0,scale=np.random.uniform(0,5)) 
	    signal_batch_pars = signal_train_pars[idx]
   
            # train only the signal PE model on the data
            if not comb_pe_model: pe_loss = signal_pe.train_on_batch(signal_batch_images,[signal_batch_pars[:,0],signal_batch_pars[:,1]])
            if comb_pe_model: pe_loss = signal_pe.train_on_batch(signal_batch_images,signal_batch_pars)
	    pe_losses.append(pe_loss)
            pe_avg_losses.append(np.mean([pe_loss[0],pe_loss[1]]))
            
            # save CNN model
            if ((i % 5000 == 0) & (i>0)):
            #if np.mean([pe_loss[0],pe_loss[1]]) <= np.min(np.array(pe_avg_losses)[:]) and save_models and do_pe and i>5000:
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

                # calculate root-mean-squared value
                if not comb_pe_model: rms = [np.mean((signal_train_pars[idx][k]-pe_samples[k])**2) for k in np.arange(2)]

                pe_mesg = "%d: [PE loss: %f, acc: %f, RMS: %f,%f]" % (i, pe_loss[0], pe_loss[1], rms[0], rms[1])
                print(pe_mesg)

                # calculate average CNN error for each parameter
                if comb_pe_model: pe_std = [np.mean(np.abs(signal_train_pars[idx][:,0]-pe_samples[:,0].reshape(pe_samples[:,0].shape[0]))),
                          np.mean(np.abs(signal_train_pars[idx][:,1]-pe_samples[:,1].reshape(pe_samples[:,0].shape[0])))]

                if not comb_pe_model: pe_std = [np.mean(np.abs(signal_train_pars[idx][:,0]-pe_samples[0].reshape(pe_samples[0].shape[0]))),
                          np.mean(np.abs(signal_train_pars[idx][:,1]-pe_samples[1].reshape(pe_samples[0].shape[0])))]
 
            
            # if new best network, plot output of PE 
            #if np.mean([pe_loss[0],pe_loss[1]]) <= np.min(np.array(pe_avg_losses)[:]) and i>5000:
            if ((i % 5000 == 0) & (i>0)):

                # plot loss curves - non-log and log
                plot_losses(pe_losses,'%s/pe_losses.png' % out_path,legend=['PE-GEN'])
                plot_losses(pe_losses,'%s/pe_losses_logscale.png' % out_path,logscale=True,legend=['PE-GEN'])
                idx = random.sample(np.arange(signal_train_images.shape[0]),4000)
                pe_samples = signal_pe.predict(np.reshape(signal_train_images[idx], (signal_train_images[idx].shape[0],signal_train_images[idx].shape[1],1)))

                # plot pe accuracy
                plot_pe_accuracy(signal_train_pars[idx],pe_samples,'%s/pe_accuracy%05d.png' % (out_path,i))

                # compute RMS difference
                if comb_pe_model: rms = [np.mean((signal_train_pars[idx][:,k]-pe_samples[:,k])**2) for k in np.arange(2)]
                if not comb_pe_model: rms = [np.mean((signal_train_pars[idx][k]-pe_samples[k])**2) for k in np.arange(2)]
                pe_mesg = "%d: [PE loss: %f, acc: %f, RMS: %f,%f]" % (i, pe_loss[0], pe_loss[1], rms[0], rms[1])
                print(pe_mesg)
               
                # calculate average CNN error for each parameter
                if comb_pe_model: pe_std = [np.mean(np.abs(signal_train_pars[idx][:,0]-pe_samples[:,0].reshape(pe_samples[:,0].shape[0]))),
                          np.mean(np.abs(signal_train_pars[idx][:,1]-pe_samples[:,1].reshape(pe_samples[:,0].shape[0])))]
                if not comb_pe_model: pe_std = [np.mean(np.abs(signal_train_pars[idx][:,0]-pe_samples[0].reshape(pe_samples[0].shape[0]))),
                          np.mean(np.abs(signal_train_pars[idx][:,1]-pe_samples[1].reshape(pe_samples[0].shape[0])))]

                # test accuracy of parameter estimation against lalinf results during training
                L, x, y = None, None, None
                more_generated_images = pickle.load(open('data/%s' % (cnn_sanity_check_file)))
                more_generated_images = np.reshape(more_generated_images, (more_generated_images.shape[0],more_generated_images.shape[1],1))
                pe_samples = signal_pe.predict(more_generated_images)
                plot_pe_samples(pe_samples,signal_pars,L,out_path,i,x,y,lalinf_pars,pe_std)

    print('Completed CNN PE')

    ################################################
    # TRAIN GAN WAVEFORM ESTIMATER #################
    ################################################

    losses = []		      # initailise the losses for plotting 
    beta_score_hist = []      # initialize overlap score history for plotting
    for i in range(max_iter):

	# get random batch from images, should be real signals
        signal_batch_images = np.array(random.sample(signal_train_images, batch_size))

	# first use the generator to make fake images - this is seeded with a size 100 random vector
        noise = np.random.uniform(size=[batch_size*n_noise_real, 100], low=-1.0, high=1.0)
        generated_images = generator.predict(noise)
        
        #Butterworth filter testing

        #def butter_lowpass(cutOff, fs, order=5):
        #    nyq = 0.5 * fs
        #    normalCutoff = cutOff / nyq
        #    b, a = butter(order, normalCutoff, btype='high', analog = True)
        #    return b, a
        #def butter_lowpass_filter(data, cutOff, fs, order=4):
        #    b, a = butter_lowpass(cutOff, fs, order=order)
        #    y = lfilter(b, a, data)
        #    return y
        #cutOff = 256 #cutoff frequency in rad/s
        #fs = 1024.0 #sampling frequency in rad/s
        #order = 20 #order of filter
        #generated_images = butter_lowpass_filter(generated_images, cutOff, fs, order)[:]
        

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

        # make labels for real and fake images
        sy = [1.0] * (batch_size*n_noise_real) + [0.0] * (batch_size*n_noise_real)

        # train only the signal discriminator on the data
        sd_loss = signal_discriminator.train_on_batch(sX, sy)

	# finally train the generator to make images that look like signals
        noise = np.random.uniform(size=[batch_size*n_noise_real, 100], low=-1.0, high=1.0)
        sg_loss = signal_discriminator_on_generator.train_on_batch(noise, [1] * (batch_size*n_noise_real))

        # fill in the loss vector for plotting
        losses.append([sg_loss[0],sg_loss[1],sd_loss[0],sd_loss[1]])

	# output status of GAN training and save results images
	if ((i % cadence == 0) & (i>0)) or (i == max_iter):
            log_mesg = "%d: [sD loss: %f, acc: %f]" % (i, sd_loss[0], sd_loss[1])
	    log_mesg = "%s  [sG loss: %f, acc: %f]" % (log_mesg, sg_loss[0], sg_loss[1])
            print(log_mesg)

            # load waveforms derived from lalinference esimated parameters
            lalinf_prod_waveforms = pickle.load(open('data/%s' % (cnn_sanity_check_file)))

            # make new generator images
            noise = np.random.uniform(size=[1000, 100], low=-1.0, high=1.0)
            generated_images = generator.predict(noise)

            # plot GAN waveform estimates
            plot_waveform_est(signal_image,noise_signal,generated_images,out_path,i,zoom=False)
            plot_waveform_est(signal_image,noise_signal,generated_images,out_path,i,zoom=True)
            if i == cadence:
                plot_waveform_est(signal_image,noise_signal,lalinf_prod_waveforms,out_path,i,plot_lalinf_wvf=True,zoom=False)
                plot_waveform_est(signal_image,noise_signal,lalinf_prod_waveforms,out_path,i,plot_lalinf_wvf=True,zoom=True)

            # plot loss curves - noMlog and log
            plot_losses(losses,'%s/losses.png' % out_path,legend=['S-GEN','S-DIS'])
            plot_losses(losses,'%s/losses_logscale.png' % out_path,logscale=True,legend=['S-GEN','S-DIS'])
            
	    # plot posterior samples
            if do_pe:
                L, x, y = None, None, None

                # first use the generator to make MANY fake images
        	noise = np.random.uniform(size=[4000, 100], low=-1.0, high=1.0)
        	more_generated_images = generator.predict(noise)
                #more_generated_images = butter_lowpass_filter(more_generated_images, cutOff, fs, order)[:]
                #more_generated_images = np.pad(more_generated_images, ((0,0),(256,256),(0,0)), 'constant', constant_values=(0, 0))
                #more_generated_images = pickle.load(open('data/cnn_sanity_check_ts.sav')) + np.random.uniform(size=[3800,1024],low=0,high=0.01)
                #test_imgs = pickle.load(open('data/cnn_sanity_check_ts.sav','rb'))
                #test_imgs = test_imgs.reshape(test_imgs.shape[0],test_imgs.shape[1],1)
                #for i in range(more_generated_images.shape[0]):
                #    plt.plot(more_generated_images[i],alpha=0.5)
                #plt.savefig('%s/test_waveform.png' % out_path)
                #plt.close()
                #exit()

                pe_samples = signal_pe.predict(np.reshape(more_generated_images, (more_generated_images.shape[0],more_generated_images.shape[1],1)))
                # average error of CNN pars after training. TODO: Need to automate this.
                pe_std = [0.02185649964844209, 0.005701401364171313]

                if not comb_pe_model: 
                    var_par1 = np.var(pe_samples[0])
                    var_par2 = np.var(pe_samples[1])
                if comb_pe_model:
                    var_par1 = np.var(pe_samples[:,0])
                    var_par2 = np.var(pe_samples[:,1])

                # if results aren't terrible, make scatter plot of PE results
                if var_par1 != 0 and var_par2 != 0:
                    beta_score_hist.append([plot_pe_samples(pe_samples,signal_pars,L,out_path,i,x,y,lalinf_pars,pe_std)])
                    plt.plot(np.linspace(cadence,i,len(beta_score_hist)),beta_score_hist)
                    plt.savefig('%s/latest/beta_hist.png' % out_path)
                    plt.close()

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
