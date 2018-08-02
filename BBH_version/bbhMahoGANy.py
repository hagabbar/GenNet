from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Reshape, AlphaDropout, Dropout, GaussianDropout, GaussianNoise
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, UpSampling1D, Conv2DTranspose
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, AveragePooling1D, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import Flatten
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import Adam
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
import pickle
import scipy
from scipy.stats import uniform, gaussian_kde
from scipy.signal import resample
from gwpy.table import EventTable
import keras
import h5py
from sympy import Eq, Symbol, solve
#import statsmodels.api as sm

cuda_dev = "6"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=cuda_dev

# define some global params
n_colors = 1		# greyscale = 1 or colour = 3 (multi-channel not supported yet)
n_pix = 512	        # the rescaled image size (n_pix x n_pix)
n_sig = 1.0            # the noise standard deviation (if None then use noise images)
batch_size = 16         # the batch size (twice this when testing discriminator)
max_iter = 50*1000 	# the maximum number of steps or epochs
pe_iter = 1*10000         # the maximum number of steps or epochs for pe network 
cadence = 100		# the cadence of output images
save_models = True	# save the generator and discriminator models
do_pe = True		# perform parameter estimation? 
pe_cadence = 100  	# the cadence of PE outputs
pe_grain = 95           # fineness of pe posterior grid
npar = 2 		# the number of parameters to estimate (PE not supported yet)
N_VIEWED = 25           # number of samples to view when plotting
chi_loss = False        # set whether or not to use custom loss function
lr = 2e-4               # learning rate for all networks
GW150914 = False        # run on lalinference produced GW150914 waveform 
gw150914_tmp = True    # run on gw150914-like template
do_old_model = True     # run previously saved model for all models
do_contours = True      # plot credibility contours on pe estimates
do_only_old_pe_model = False # run previously saved pe model only
contour_cadence = 100   # the cadence of PE contour plot outputs

# the locations of signal files and output directory
signal_path = '/home/hunter.gabbard/Burst/GenNet/BBH_version/data/event_gw150914_psd.pkl'
#pars_path = '/home/hunter.gabbard/Burst/GenNet/tests/data/burst/data_pars.pkl'
if gw150914_tmp:
    out_path = '/home/hunter.gabbard/public_html/CBC/mahoGANy/gw150914_template' 
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
        self.output_dim = 2			# the output dimension
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
	super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # computes the mean of the difference between the meausured image and the generated signal
        # and returns it as a Keras object
        # add in K cube of diff as third option
        diff = self.const - x
        return K.stack([K.mean(diff), K.mean(K.square(diff))])

    def compute_output_shape(self, input_shape):
        # the output shape which seems to be (None,2) since the 2 is the number of 
        # outputs and the None needs to be there?
        return (input_shape[0],2)

def generator_model():
    """
    The generator that should train itself to generate noise free signals
    """
    model = Sequential()
    act = 'relu'
    momentum = 0.9
    drate = 0.01

    
    # the first dense layer converts the input (100 random numbers) into
    # 1024 numbers and outputs with a tanh activation
    #model.add(Dense(128, input_shape=(100,)))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(GaussianDropout(0.3))
 
    #model.add(Dense(512))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(GaussianDropout(0.3))

    #model.add(Dense(256))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(GaussianDropout(0.3))

    # the second dense layer expands this up to 32768 and again uses a
    # tanh activation function
    model.add(Dense(256 * 1 * int(n_pix/2), input_shape=(100,)))
    model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(PReLU())
    #model.add(BatchNormalization(momentum=momentum))
    model.add(GaussianDropout(drate))

    # then we reshape into a cube, upsample by a factor of 2 in each of
    # 2 dimensions and apply a 2D convolution with filter size 5x5
    # and 64 neurons and again the activation is tanh 
    model.add(Reshape((int(n_pix/2), 256)))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(64, 5, strides=1, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Activation(act))
    #model.add(PReLU())
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))
    model.add(GaussianDropout(drate))

    model.add(UpSampling1D(size=2))
    model.add(Conv1D(128, 5, strides=1, padding='same'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Activation(act))
    #model.add(PReLU())
    #model.add(LeakyReLU(alpha=0.2))
    model.add(GaussianDropout(drate))

    #model.add(UpSampling1D(size=2))
    model.add(Conv1D(512, 5, strides=2, padding='same'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Activation(act))
    #model.add(PReLU())
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(GaussianDropout(drate))

    model.add(UpSampling1D(size=2))
    model.add(Conv1D(512, 5, strides=1, padding='same'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Activation(act))
    #model.add(PReLU())
    #model.add(LeakyReLU(alpha=0.2))
    model.add(GaussianDropout(drate))

    # if we have a 64x64 pixel dataset then we upsample once more 
    #if n_pix==64:
    #    model.add(UpSampling2D(size=(1, 2)))
    # apply another 2D convolution with filter size 5x5 and a tanh activation
    # the output shape should be n_colors x n_pix x n_pix
    model.add(Conv1D(n_colors, 5, padding='same'))
    model.add(Activation('linear')) # this should be tanh
    
    return model

def data_subtraction_model(noise_signal,npix):
    """
    This model simply applies the signal subtraction from the measured image
    You must pass it the measured image
    """
    model = Sequential()
    model.add(MyLayer(noise_signal,input_shape=(npix,1))) # used to be another element for n_colors
   
    return model

def signal_pe_model():
    """
    The PE network that learns how to convert images into parameters
    
    model = Sequential()
    act = 'tanh'

    # the first layer is a 2D convolution with filter size 5x5 and 64 neurons
    # the activation is tanh and we apply a 2x2 max pooling
    model.add(Conv1D(64, 5, strides=2, input_shape=(n_pix,1), padding='same'))
    model.add(Activation(act))
    #model.add(PReLU())
    #model.add(MaxPooling2D(pool_size=(1, 2)))

    # the next layer is another 2D convolution with 128 neurons and a 5x5
    # filter. More 2x2 max pooling and a tanh activation. The output is flattened
    # for input to the next dense layer
    model.add(Conv1D(128, 5, strides=2))
    model.add(Activation(act))
    #model.add(PReLU())
    #model.add(MaxPooling2D(pool_size=(1, 2)))

    model.add(Conv1D(256, 5, strides=2))
    model.add(Activation(act))

    model.add(Conv1D(512, 5, strides=2))
    model.add(Activation(act))

    model.add(Flatten())

    # we now use a dense layer with 1024 outputs and a tanh activation
    model.add(Dense(1024))
    model.add(Activation(act))
    #model.add(PReLU())

    # the final dense layer has a linear activation and 2 outputs
    # we are currently testing with only 2 outputs - can be generalised
    model.add(Dense(2))
    model.add(Activation('relu'))
    """

    inputs = Input(shape=(n_pix,1))
    act = 'tanh'

    mc_branch = Conv1D(64, 5, strides=2, padding='same')(inputs)
    mc_branch = Activation(act)(mc_branch)

    mc_branch = Conv1D(128, 5, strides=2)(mc_branch)
    mc_branch = Activation(act)(mc_branch)

    mc_branch = Conv1D(256, 5, strides=2)(mc_branch)
    mc_branch = Activation(act)(mc_branch)

    mc_branch = Conv1D(512, 5, strides=2)(mc_branch)
    mc_branch = Activation(act)(mc_branch)

    mc_branch = Flatten()(mc_branch)

    mc_branch = Dense(1024)(mc_branch)
    mc_branch = Activation(act)(mc_branch)

    mc_branch = Dense(1)(mc_branch)
    mc_barnch = Activation('relu')(mc_branch)
    
    act = 'sigmoid' 
    q_branch = Conv1D(64, 5, strides=2, padding='same')(inputs)
    q_branch = Activation(act)(q_branch)

    q_branch = Conv1D(128, 5, strides=2)(q_branch)
    q_branch = Activation(act)(q_branch)

    q_branch = Conv1D(256, 5, strides=2)(q_branch)
    q_branch = Activation(act)(q_branch)

    q_branch = Conv1D(512, 5, strides=2)(q_branch)
    q_branch = Activation(act)(q_branch)

    q_branch = Flatten()(q_branch)

    q_branch = Dense(1024)(q_branch)
    q_branch = Activation(act)(q_branch)

    q_branch = Dense(1)(q_branch)
    q_barnch = Activation('relu')(q_branch)
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

    
    act='tanh'
    momentum=0.8

    model = Sequential()

    
    # the first layer is a 2D convolution with filter size 5x5 and 64 neurons
    # the activation is tanh and we apply a 2x2 max pooling
    model.add(Conv1D(64, 5, input_shape=(n_pix,1), strides=1, padding='same'))
    model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=2))

    # the next layer is another 2D convolution with 128 neurons and a 5x5 
    # filter. More 2x2 max pooling and a tanh activation. The output is flattened
    # for input to the next dense layer
    model.add(Conv1D(128, 5, strides=1))
    model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Dropout(0.3))
    #model.add(MaxPooling1D(pool_size=2))

    #model.add(Conv1D(256, 5, strides=1))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Dropout(0.3))
    #model.add(MaxPooling1D(pool_size=2))

    #model.add(Conv1D(512, 5, strides=1))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Dropout(0.3))
    #model.add(MaxPooling1D(pool_size=2))

    #model.add(Conv1D(1024, 5))
    #model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Dropout(0.3))
    # model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    # we now use a dense layer with 1024 outputs and a tanh activation
    model.add(Dense(1024))
    #model.add(BatchNormalization(momentum=momentum))
    model.add(Activation(act))
    #model.add(LeakyReLU(alpha=0.2))

    # the final dense layer has a sigmoid activation and a single output
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
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

def plot_losses(losses,filename,logscale=False,legend=None):
    """
    Make loss and accuracy plots and output to file.
    Plot with x and y log-axes is desired
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
        #ax2.set_xscale("log", nonposx='clip')
        ax1.set_yscale("log", nonposy='clip')
    plt.savefig(filename)
    plt.close('all')

def plot_pe_accuracy(true_pars,est_pars,outfile):
    """
    Plots the true vs the estimated paranmeters from the PE training
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(121,aspect=1.0)
    ax1.plot(true_pars[:,0],est_pars[0],'.b')
    ax1.plot([0,np.max(true_pars[:,0])],[0,np.max(true_pars[:,0])],'--k')
    ax1.set_xlabel(r'True parameter 1')
    ax1.set_ylabel(r'Estimated parameter 1')
    ax1.set_xlim([0,np.max(true_pars[:,0])])
    ax1.set_ylim([0,np.max(true_pars[:,0])])
    ax2 = fig.add_subplot(122,aspect=1.0)
    ax2.plot(true_pars[:,1],est_pars[1],'.b')
    ax2.plot([0,np.max(true_pars[:,1])],[0,np.max(true_pars[:,1])],'--k')
    ax2.set_xlabel(r'True parameter 2')
    ax2.set_ylabel(r'Estimated parameter 2')
    ax2.set_xlim([0,np.max(true_pars[:,1])])
    ax2.set_ylim([0,np.max(true_pars[:,1])])
    plt.savefig(outfile)
    plt.close('all')

def plot_pe_samples(pe_samples,truth,like,outfile,index,x,y,lalinf_dist=None,pe_std=None):
    """
    Makes scatter plot of samples estimated from PE model
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
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
        # uncomment if you want to plot scatter points
        ax1.plot(pe_samples[0],pe_samples[1],'.r',markersize=0.8)
  
        if do_contours and ((index % contour_cadence == 0) and (index>0)): 
            # plot contours for generated samples
            contour_y = np.reshape(pe_samples[1], (pe_samples[0].shape[0]))
            contour_x = np.reshape(pe_samples[0], (pe_samples[0].shape[0]))
            contour_dataset = np.array([contour_x,contour_y])
            make_contour_plot(ax1,contour_x,contour_y,contour_dataset,'Reds',flip=False)

    # plot contours of lalinf distribution
    if lalinf_dist is not None:
        # plot lalinference parameters
        ax1.plot(lalinf_dist[0][:],lalinf_dist[1][:],'.b', markersize=0.8)

        if do_contours and ((index % contour_cadence == 0) and (index>0)):
            # plot lalinference parameter contours
            make_contour_plot(ax1,lalinf_dist[0][:],lalinf_dist[1][:],lalinf_dist,'Blues',flip=False)

    # plot pe_std error bars
    if pe_std:
        ax1.plot([truth[0]-pe_std[0],truth[0]+pe_std[0]],[truth[1],truth[1]], '-c')
        ax1.plot([truth[0], truth[0]],[truth[1]-pe_std[1],truth[1]+pe_std[1]], '-c')

    ax1.plot([truth[0],truth[0]],[np.min(y),np.max(y)],'-k', alpha=0.5)
    ax1.plot([np.min(x),np.max(x)],[truth[1],truth[1]],'-k', alpha=0.5)


    ax1.set_xlabel(r'mc')
    ax1.set_ylabel(r'mass ratio')
    #ax1.set_xlim([np.min(all_pars[:,0]),np.max(all_pars[:,0])])
    #ax1.set_ylim([np.min(all_pars[:,1]),np.max(all_pars[:,1])])
    plt.savefig('%s/pe_samples%05d.png' % (outfile,index))
    plt.savefig('%s/latest/pe_samples.png' % (outfile))
    plt.close('all')


def make_contour_plot(ax,x,y,dataset,color='Reds_d',flip=False):

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
    ax.contour(X,Y,Z,levels=levels,alpha=0.5)
    #ax.set_aspect('equal')

    return

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

def main():

    ################################################
    # READ/GENERATE DATA ###########################

    # setup output directory - make sure it exists
    os.system('mkdir -p %s' % out_path) 

    template_dir = 'templates/'   

    # load in lalinference m1 and m2 parameters
    pickle_lalinf_pars = open("data/gw150914_mc_q_lalinf_post.sav")
    lalinf_pars = pickle.load(pickle_lalinf_pars)


    # load hplus and hcross pickle file
    #pickle_hp = open("%shp.pkl" % template_dir,"rb")
    #hp = pickle.load(pickle_hp)
    #pickle_hc = open("%shc.pkl" % template_dir,"rb")
    #hc = pickle.load(pickle_hc)
    #pickle_fmin = open("%sfmin.pkl" % template_dir,"rb")
    #fmin_bank = pickle.load(pickle_fmin)

    # load time series template pickle file
    pickle_ts = open("%s_ts_0_8000Samp.sav" % template_dir,"rb")
    ts = pickle.load(pickle_ts)

    # load corresponding parameters template pickle file
    pickle_par = open("%s_params_0_8000Samp.sav" % template_dir,"rb")
    par = pickle.load(pickle_par)

    signal_train_images = np.reshape(ts[0], (ts[0].shape[0],ts[0].shape[2]))

    signal_train_pars = []
    for k in par:
        signal_train_pars.append([k.mc,1.0/(k.m1/k.m2)])

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
        pickle_gw150914 = open("data/gw150914.sav","rb")
        noise_signal = pickle.load(pickle_gw150914)
        signal_image = pickle.load(open("data/GW150914_data.pkl","rb"))[1]

    if gw150914_tmp:
        signal_image = signal_train_images[-1,:]
        signal_train_images = np.delete(signal_train_images,-1,axis=0)

    if do_pe and not GW150914 and not gw150914_tmp:
        signal_pars = signal_train_pars[i,:][0]
        print(signal_pars)
        signal_train_pars = np.delete(signal_train_pars,i,axis=0)    

    if do_pe and gw150914_tmp:
        signal_pars = signal_train_pars[-1,:]
        signal_train_pars = np.delete(signal_train_pars,-1,axis=0)

    # combine signal and noise - this is the measured data i.e., h(t)
    if GW150914:
        noise_signal = noise_signal[int((4*512/2)-(0.5*512)):int((4*512/2)+(0.5*512))]
        signal_image = signal_image[int((32*4096/2)-(0.5*4096)):int((32*4096/2)+(0.5*4096))]

        # resample GW150914
        noise_signal = resample(noise_signal,n_pix)
        signal_image = resample(signal_image,n_pix)

        peak_diff = np.abs(np.argmax(noise_signal)-np.argmax(signal_image))
        signal_image = np.roll(signal_image,-peak_diff)

        # set signal_pars m1 and m2
        signal_pars = [36.0,29.0]

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
    signal_discriminator = signal_discriminator_model()
    data_subtraction = data_subtraction_model(noise_signal,n_pix)	# need to pass the measured data here
    generator = generator_model()
    if do_pe:
        signal_pe = signal_pe_model()    

    """
    setup generator training for when we subtract from the 
    measured data and expect Gaussian residuals.
    We use a mean squared error here since we want it to find 
    the situation where the residuals have the known mean=0, std=n_sig properties
    """
    if not chi_loss:
        data_subtraction_on_generator = generator_after_subtracting_noise(generator, data_subtraction)
        data_subtraction_on_generator.compile(loss='mean_squared_error', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    # setup generator training when we pass the output to the signal discriminator
    signal_discriminator_on_generator = generator_containing_signal_discriminator(generator, signal_discriminator)
    set_trainable(signal_discriminator, False)	# set the discriminator as not trainable for this step
    if not chi_loss:
        signal_discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])
    elif chi_loss:
        signal_discriminator_on_generator.compile(loss=chisquare_Loss, optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    # setup trainin on signal discriminator model
    # This uses a binary cross entropy loss since we are just 
    # discriminating between real and fake signals
    set_trainable(signal_discriminator, True)	# set it back to being trainable
    signal_discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])
    #elif chi_loss:
    #    signal_discriminator.compile(loss=chisquare_Loss, optimizer=Adam(lr=9e-5, beta_1=0.5), metrics=['accuracy'])

    if do_pe:
        signal_pe.compile(loss='mean_squared_error', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    # print the model summaries
    print(generator.summary())
    if not chi_loss:
        print(data_subtraction_on_generator.summary())
    print(signal_discriminator_on_generator.summary())
    print(signal_discriminator.summary())
    if do_pe:
        print(signal_pe.summary())

    ################################################
    # DO PARAMETER ESTIMATION ######################

    if do_old_model:
        if do_pe:
            signal_pe = keras.models.load_model('signal_pe.h5')
            #signal_pe.load_weights('signal_pe.h5')
        signal_discriminator.load_weights('discriminator.h5')
        signal_discriminator_on_generator.load_weights('signal_dis_on_gen.h5')
        data_subtraction_on_generator.load_weights('data_subtract_on_gen.h5')
        generator.load_weights('generator.h5')

    if do_only_old_pe_model:
        # load old pe model by default
        signal_pe = keras.models.load_model('best_models/signal_pe.h5')
        #signal_pe.load_weights('signal_pe.h5')

    if do_pe: #and not do_only_old_pe_model and not do_old_model:

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
        i = 0
        rms = [1.0,1.0]
        
        for i in range(pe_iter):
	
            # get random batch from images
            idx = random.sample(np.arange(signal_train_images.shape[0]),batch_size)
            signal_batch_images = signal_train_images[idx]
            signal_batch_images = np.reshape(signal_batch_images, (signal_batch_images.shape[0],signal_batch_images.shape[1],1))
            #signal_batch_images /= np.max(signal_batch_images)
	    signal_batch_pars = signal_train_pars[idx]


            # train only the signal PE model on the data
            pe_loss = signal_pe.train_on_batch(signal_batch_images,[signal_batch_pars[:,0],signal_batch_pars[:,1]])
	    pe_losses.append(pe_loss)

	    # output status and save images
            if ((i % pe_cadence == 0) & (i>0)):

		# plot loss curves - non-log and log
                plot_losses(pe_losses,'%s/pe_losses.png' % out_path,legend=['PE-GEN'])
                plot_losses(pe_losses,'%s/pe_losses_logscale.png' % out_path,logscale=True,legend=['PE-GEN'])

		# plot true vs predicted values for all training data
                pe_samples = signal_pe.predict(np.reshape(signal_train_images, (signal_train_images.shape[0],signal_train_images.shape[1],1)))

                # plot pe accuracy
		plot_pe_accuracy(signal_train_pars,pe_samples,'%s/pe_accuracy%05d.png' % (out_path,i))

	        # compute RMS difference
                rms = [np.mean((signal_train_pars[:,k]-pe_samples[k])**2) for k in np.arange(2)]

                pe_mesg = "%d: [PE loss: %f, acc: %f, RMS: %f,%f]" % (i, pe_loss[0], pe_loss[1], rms[0], rms[1])
                print(pe_mesg)

                pe_std = [np.mean(np.abs(signal_train_pars[:,0]-pe_samples[0].reshape(pe_samples[0].shape[0]))),
                          np.mean(np.abs(signal_train_pars[:,1]-pe_samples[1].reshape(pe_samples[0].shape[0])))]
        
    # load old pe model by default
    #signal_pe = keras.models.load_model('signal_pe.h5')
    #signal_pe.load_weights('signal_pe.h5')
    print('Completed CNN PE')

    ################################################
    # LOOP OVER BATCHES ############################

    losses = []		# initailise the losses for plotting 
    for i in range(max_iter):

	# get random batch from images, should be real signals
        signal_batch_images = np.array(random.sample(signal_train_images, batch_size))

	# first use the generator to make fake images - this is seeded with a size 100 random vector
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        generated_images = generator.predict(noise)

	# make set of real and fake signal mages with labels
        signal_batch_images = np.reshape(signal_batch_images, (signal_batch_images.shape[0], signal_batch_images.shape[1], 1))
        sX = np.concatenate((signal_batch_images, generated_images))
        sy = [1] * batch_size + [0] * batch_size

        # train only the signal discriminator on the data
        sd_loss = signal_discriminator.train_on_batch(sX, sy)

 	# next train the generator to make signals that have residuals (after
        # subtracting from the measured data) that have the correct Gaussian properties
	noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        ny = np.zeros((batch_size,2))	# initialise the expected residual means as zero 
        ny[:,1] = n_sig**2		# initialise the expected variances as n_sig squared
        #ny[:,2] = 0                     # initialise the expected 3rd moment of n_sig as zero
        #ny[:,3] = 3                     # initialise the expected 4th moment of n_sig as 3.
        if not chi_loss:
            ng_loss = data_subtraction_on_generator.train_on_batch(noise, ny)
        else:
            ng_loss = data_subtraction_on_generator.train_on_batch(noise, [noise_signal] * batch_size)

	# finally train the generator to make images that look like signals
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        sg_loss = signal_discriminator_on_generator.train_on_batch(noise, [1] * batch_size)

        # fill in the loss vector for plotting
        if not chi_loss:
            losses.append([sg_loss[0],sg_loss[1],sd_loss[0],sd_loss[1],ng_loss[0],ng_loss[1]])
        #elif chi_loss:
        #    losses.append([sg_loss[0],sg_loss[1],sd_loss[0],sd_loss[1]])

	# output status and save images
	if ((i % cadence == 0) & (i>0)) or (i == max_iter):
            log_mesg = "%d: [sD loss: %f, acc: %f]" % (i, sd_loss[0], sd_loss[1])
	    log_mesg = "%s  [sG loss: %f, acc: %f]" % (log_mesg, sg_loss[0], sg_loss[1])
            if not chi_loss:
                log_mesg = "%s  [nG loss: %f, acc: %f]" % (log_mesg, ng_loss[0], ng_loss[1])
            print(log_mesg)

            # plot original waveform
            f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
            ax = signal_image
            ax1.plot(ax, color='cyan', alpha=0.5, linewidth=0.5)
            ax1.plot(noise_signal, color='green', alpha=0.35, linewidth=0.5)
            ax1.set_title('signal + (sig+noise)')

            # plotable generated signals
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
            ax2.plot(signal_image, color='cyan', linewidth=0.5)
            ax2.fill_between(np.linspace(0,len(perc_90),num=len(perc_90)),perc_90, perc_5, lw=0,facecolor='#d5d8dc')
            ax2.fill_between(np.linspace(0,len(perc_75),num=len(perc_75)),perc_75, perc_25, lw=0,facecolor='#808b96')
            ax2.set_title('gen + sig + (sig+noise)')
	    
	    # plot residuals - generated images subtracted from the measured image
            # the first image is the true noise realisation
            residuals = np.transpose(np.transpose(noise_signal)-gen_sig)
            ax3.plot((residuals), color='red', alpha=0.25, linewidth=0.5)
            
            ax3.set_title('Residuals')

            # save waveforms plot
            plt.savefig('%s/waveform_results%05d.png' % (out_path,i), dpi=500)
            plt.savefig('%s/latest/most_recent_waveform.png' % out_path, dpi=400)
            plt.close()
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
            if not chi_loss:
                plot_losses(losses,'%s/losses.png' % out_path,legend=['S-GEN','S-DIS','N-GEN'])
                plot_losses(losses,'%s/losses_logscale.png' % out_path,logscale=True,legend=['S-GEN','S-DIS','N-GEN'])
            if chi_loss:
                plot_losses(losses,'%s/losses.png' % out_path,legend=['S-GEN','S-DIS'])
                plot_losses(losses,'%s/losses_logscale.png' % out_path,logscale=True,legend=['S-GEN','S-DIS'])

            
	    # plot posterior samples
            if do_pe:
                L, x, y = None, None, None
                # first use the generator to make MANY fake images
        	noise = np.random.uniform(size=[1000, 100], low=-1.0, high=1.0)
        	more_generated_images = generator.predict(noise)
                pe_samples = signal_pe.predict(more_generated_images)
                plot_pe_samples(pe_samples,signal_pars,L,out_path,i,x,y,lalinf_pars,pe_std)

                # make pp plot
                # plot contours for generated samples
                #pe_samples_y = np.reshape(pe_samples[1], (pe_samples[0].shape[0]))
                #pe_samples_x = np.reshape(pe_samples[0], (pe_samples[0].shape[0]))
                #pe_samples = np.array([pe_samples_x,pe_samples_y])
                #sm.ProbPlot.ppplot(sm.ProbPlot(lalinf_pars))
                #plt.savefig('%s/pp_plot.png' % out_path) 
                #plt.close()                          
                
	    # save trained models
            if save_models:
	        generator.save_weights('generator.h5', True)
                signal_discriminator.save_weights('discriminator.h5', True)
                if not chi_loss:
                    data_subtraction_on_generator.save_weights('data_subtract_on_gen.h5', True)
                signal_discriminator_on_generator.save_weights('signal_dis_on_gen.h5', True)
                if do_pe:
                    signal_pe.save('signal_pe.h5', True)

            # save posterior samples
            #f = open('GAN_posterior_samples/posterior_samples_%05d.sav' % i, 'wb')
            #pickle.dump(pe_samples, f)
            #f.close()
            #print '{}: saved posterior data to file'.format(time.asctime())

main()
