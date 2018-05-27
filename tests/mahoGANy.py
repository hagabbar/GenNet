from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape, Dropout
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
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
import glob
import random
import string
from sys import exit
import pandas as pd

# define some global params
mnist_sig = False	# use the mnist dataset in tensorflow?
Ngauss_sig = 10000	# Number of simple Gaussian blob signals to generate (<=0 means don't use)
n_colors = 1		# greyscale = 1 or colour = 3
n_pix = 50		# the rescaled image size (n_pix x n_pix)
n_sig = 0.25		# the noise standard deviation (if None then use noise images)
batch_size = 128	# the batch size (twice this when testing discriminator)
max_iter = 10*1000	# the maximum number of steps or epochs
cadence = 100 		# the cadence of output images
save_models = False	# save the generator and discriminator models
do_pe = False		# perform parameter estimation? 
pe_cadence = 100  	# the cadence of PE outputs
npar = 2 		# the number of parameters to estimate
blob_scale = 0.15	# the scale of the Gaussian blob widths (image spans 0-1)

# catch input errors
if n_pix != 50:
    print 'Sorry, only deals with 1x50 images'
    exit(0)
if mnist_sig==True and n_colors != 1:
    print 'Sorry, the mnist data is greyscale only'
    exit(0)
if mnist_sig==True and Ngauss_sig>0:
    print 'Sorry, can\'t use both mnist and gaussian signals'
    exit(0)
if Ngauss_sig>0 and n_colors==3:
    print 'Sorry, can\'t use colour images with the gaussian signals'
    exit(0)    
if do_pe==True and Ngauss_sig<=0:
    print 'Sorry, can only do parameter estimation if using a parameterised signal model'
    exit(0)

# the locations of signal files and output directory
signal_path = './data/gwbush/*.jpg'
out_path = '/home/hunter.gabbard/public_html/Burst/mahoGANy'

def sample_data(n_samples=10000, x_vals=np.arange(0, 5, .1), max_offset=2*np.pi, mul_range=[1, 2], snr=1):
    vectors = []
    for i in range(n_samples):
        offset = np.random.random() * max_offset
        #mul = mul_range[0] + np.random.random() * (mul_range[1] - mul_range[0])
        mul = (2 * np.pi) / 5
        vectors.append(
            np.sin(offset + x_vals * mul) * snr
        )
    return np.array(vectors)

class MyLayer(Layer):
    """
    This layer just computes 
    a) the mean of the differences between the input image and the measured image
    b) the mean of the squared differences between the input image and the measured image
    Calling this layer requires you to pass the measured image (const)
    """
    def __init__(self, const, **kwargs):
        self.const = K.constant(const)		# the input measured image i.e., h(t)
        self.output_dim = 2			# the output dimension
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
	super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # computes the mean of the difference between the meausured image and the generated signal
        # and returns it as a Keras object
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
    act = 'linear'

    # the first dense layer converts the input (100 random numbers) into
    # 1024 numbers and outputs with a tanh activation
    model.add(Dense(1024, input_shape=(100,)))
    model.add(Activation(act))
    model.add(LeakyReLU(alpha=0.2))

    # the second dense layer expands this up to 32768 and again uses a
    # tanh activation function
    model.add(Dense(128 * 1 * 25))
    model.add(Activation(act))
    model.add(LeakyReLU(alpha=0.2))

    # then we reshape into a cube, upsample by a factor of 2 in each of
    # 2 dimensions and apply a 2D convolution with filter size 5x5
    # and 64 neurons and again the activation is tanh 
    model.add(Reshape((1, 25, 128)))
    model.add(UpSampling2D(size=(1, 2)))
    model.add(Conv2D(64, (1, 5), padding='same'))
    model.add(Activation(act))
    model.add(LeakyReLU(alpha=0.2))

    # if we have a 64x64 pixel dataset then we upsample once more 
    #if n_pix==64:
    #    model.add(UpSampling2D(size=(1, 2)))
    # apply another 2D convolution with filter size 5x5 and a tanh activation
    # the output shape should be n_colors x n_pix x n_pix
    model.add(Conv2D(n_colors, (1, 5), padding='same'))
    model.add(Activation('tanh')) # this should be tanh

    return model

def data_subtraction_model(noise_signal):
    """
    This model simply applies the signal subtraction from the measured image
    You must pass it the measured image
    """
    model = Sequential()
    model.add(MyLayer(noise_signal,input_shape=(n_pix, n_pix, n_colors)))
   
    return model

def signal_pe_model():
    """
    The PE network that learns how to convert images into parameters
    """
    model = Sequential()

    # the first layer is a 2D convolution with filter size 5x5 and 64 neurons
    # the activation is tanh and we apply a 2x2 max pooling
    model.add(Conv2D(64, (5, 5), input_shape=(n_pix, n_pix, n_colors), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the next layer is another 2D convolution with 128 neurons and a 5x5
    # filter. More 2x2 max pooling and a tanh activation. The output is flattened
    # for input to the next dense layer
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # we now use a dense layer with 1024 outputs and a tanh activation
    model.add(Dense(1024))
    model.add(Activation('tanh'))

    # the final dense layer has a linear activation and 2 outputs
    # we are currently testing with only 2 outputs - can be generalised
    model.add(Dense(2))
    model.add(Activation('linear'))

    return model

def signal_discriminator_model():
    """
    The discriminator that should train itself to recognise generated signals
    from real signals
    """
    model = Sequential()

    # the first layer is a 2D convolution with filter size 5x5 and 64 neurons
    # the activation is tanh and we apply a 2x2 max pooling
    model.add(Conv2D(64, (1, 5), input_shape=(1, n_pix, n_colors), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(1, 2)))

    # the next layer is another 2D convolution with 128 neurons and a 5x5 
    # filter. More 2x2 max pooling and a tanh activation. The output is flattened
    # for input to the next dense layer
    model.add(Conv2D(128, (1, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Flatten())

    # we now use a dense layer with 1024 outputs and a tanh activation
    model.add(Dense(1024))
    model.add(Activation('tanh'))

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

def renorm(image):
    """
    This function simply rescales an image between -1 and 1.
    *IMPORTANT* This is for plotting purposes *ONLY*.
    """
    hspan = 0.5*(np.max(image)-np.min(image))
    mean = 0.5*(np.max(image)+np.min(image))
    return (image - mean)/hspan

def load_images(filepath,flip=True,mnist=False,Nsig=0):
    """
    This function loads in images from the filepath
    If flip is True then we make a horizontally flipped copy of each image read in.
    I mnist is True then we read in mnist data from tensorflow
    """
    res = []	# initialise the output 

    # if the mnist flag has been set then read in the mnist data from tensorflow
    if mnist:
        data = input_data.read_data_sets("mnist",one_hot=True).train.images
        data = np.array(data).reshape(-1,28,28,1).astype(np.float32)
	for tmp in data:
	    img = Image.fromarray(tmp.squeeze())
            arr = np.array(img.resize((n_pix, n_pix)))	# resize from 28x28 to n_pix x n_pix
            res.append(renorm(arr))			# normalise each image to be -1 to 1	

    elif Nsig>0:					# or make fake parameterised Gaussian blob signals    

	# make some Gaussian blob images
        # these are parameterised so we can do PE with them
	res,pars = gen_gauss_signals(Nsig)	
        return np.array(res).reshape(-1,n_pix,n_pix,n_colors), np.array(pars).reshape(-1,npar)

    else:						# otherwise read in from directory
        files = glob.glob(filepath)			# get filelist from path
        
	# loop over files
	for path in files:
            img = Image.open(path)			# open file
            img = img.resize((n_pix, n_pix))		# resize to n_pix x n_pix
            if n_colors==1:
                img = img.convert('L')			# if greyscale then convert it 
	    arr = np.array(img)
	    #arr = (arr - 127.5) / 127.5
	    arr = renorm(arr)				# rescale from -1 to 1
            arr.resize((n_pix, n_pix, n_colors))	# make sure its a 3D object
            res.append(arr)				# append to the read in images
	    if flip:
                res.append(arr[:,::-1,:])   		# add flipped image
    
    # return the reshaped array of images
    return np.array(res).reshape(-1,n_pix,n_pix,n_colors)

def gen_gauss_signals(N=1,pars=None):
    """
    This function generates Gaussian blob images with varying location parameters
    - takes in parameter values if provided (x and y means as fraction of image)
    """

    # initialise results and set up vector of locations on the image
    sig = []			# stores the output images
    params = []			# stores the output parameters (x,y means)
    cov = (blob_scale**2)*np.eye(2)	# the blob covariance (constant for now)
    x = np.arange(n_pix)
    xy = np.array([k for k in itprod(x,x)]).reshape(n_pix*n_pix,2)
    
    # loop over each generated signal
    for _ in range(N):
        if pars is not None:
            mean = pars		# if params provided then use them
        else:
	    mean = np.random.uniform(0,1.0,size=2)	# else draw random values
        	
	# compute Gaussian pdf value at all locations for this choice of mean
        # also renormalise to [-1,1]
	sig.append(renorm(mvn.pdf(xy, mean=n_pix*mean, cov=n_pix*n_pix*cov)))
        params.append([mean[0],mean[1]])	# record the params

    # return the images and the paramneters used to generate them
    return np.array(sig).reshape(N,n_pix,n_pix,n_colors), np.array(params).reshape(-1,npar)

def combine_images(generated_images, cols=4, rows=4,randomize=True,extra=None):
    """
    Function to generate tile plots of images.
    - cols and rows args set the grid size.
    - randomize randomly selects the order of the images to plot.
    - extra points to a special image that gets plotted in the top left tile
    """
    
    # setup the size of the output image
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    image = np.ones((rows * h,  cols * w, n_colors))

    # make index in order or randomize
    i = np.arange(generated_images.shape[0])
    if randomize:
	i = np.random.randint(0,generated_images.shape[0],size=rows*cols)
    
    # loop over the images until we've filled each tile
    # for each iteration we place the image in the correct tile
    for index, img in enumerate(generated_images[i]):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]

    # if we want to plot an extra special image then do so in the first tile
    if extra is not None:
        image[0:h, 0:w, :] = extra[:, :, :]

    # rescale back to [0,255] range for plotting and # resize to 512 x 512 for 
    # easy viewing 
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8).squeeze())
    image = image.resize([512,512])
    
    return image

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
        ax2.set_xscale("log", nonposx='clip')
        ax1.set_yscale("log", nonposx='clip')
    plt.savefig(filename)
    plt.close('all')

def plot_pe_accuracy(true_pars,est_pars,outfile):
    """
    Plots the true vs the estimated paranmeters from the PE training
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(121,aspect=1.0)
    ax1.plot(true_pars[:,0],est_pars[:,0],'.b')
    ax1.plot([0,1],[0,1],'--k')
    ax1.set_xlabel(r'True parameter 1')
    ax1.set_ylabel(r'Estimated parameter 1')
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    ax2 = fig.add_subplot(122,aspect=1.0)
    ax2.plot(true_pars[:,1],est_pars[:,1],'.b')
    ax2.plot([0,1],[0,1],'--k')
    ax2.set_xlabel(r'True parameter 2')
    ax2.set_ylabel(r'Estimated parameter 2')
    ax2.set_xlim([0,1])
    ax2.set_ylim([0,1])
    plt.savefig(outfile)
    plt.close('all')

def plot_pe_samples(pe_samples,truth,like,outfile):
    """
    Makes scatter plot of samples estimated from PE model
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if like is not None:
        # compute enclose probability contours
        enc_post = get_enclosed_prob(like,1.0/n_pix)
        x = np.linspace(0,1,n_pix)
        X, Y = np.meshgrid(x,x)
        cmap = plt.cm.get_cmap("Greys")
        ax1.contourf(X, Y, enc_post, 100, cmap=cmap) 
        ax1.contour(X, Y, enc_post, [1.0-0.68], colors='b',linestyles='solid')
        ax1.contour(X, Y, enc_post, [1.0-0.9], colors='b',linestyles='dashed')
        ax1.contour(X, Y, enc_post, [1.0-0.99], colors='b',linestyles='dotted') 
    if pe_samples is not None:
        ax1.plot(pe_samples[:,0],pe_samples[:,1],'.r',markersize=0.8)
    ax1.plot([truth[0],truth[0]],[0,1],'-k')
    ax1.plot([0,1],[truth[1],truth[1]],'-k')
    ax1.set_xlabel(r'Parameter 1')
    ax1.set_ylabel(r'Parameter 2')
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    plt.savefig(outfile)
    plt.close('all')

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

def main():

    ################################################
    # READ/GENERATE DATA ###########################
    # should add in PE stuff once you understand 
    # how it works

    # setup output directory - make sure it exists
    os.system('mkdir -p %s' % out_path)     
    
    
    # load signal training images and save examples
    #signal_train_images, signal_train_pars = load_images(signal_path,mnist=mnist_sig,Nsig=Ngauss_sig)
    #signal_train_out = combine_images(signal_train_images)
    #signal_train_out.save('%s/signal_train.png' % out_path)
   
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
    #i = np.random.randint(0,signal_train_images.shape[0],size=1)
    signal_image = sample_data(1)
    #signal_train_images = np.delete(signal_train_images,i,axis=0)
    if do_pe:
        signal_pars = signal_train_pars[i,:]
        signal_train_pars = np.delete(signal_train_pars,i,axis=0)    


    # Generate single noise image
    noise_image = np.random.normal(0, 0.25, size=[1, signal_image.shape[1]])

    # combine signal and noise - this is the measured data i.e., h(t)
    noise_signal = signal_image + noise_image

    # output combined true signal and noise image - normalise between -1,1 *ONLY* for plotting
    #tmp = np.array([signal_image,noise_image,renorm(noise_signal)]).reshape(3,n_pix,n_pix,n_colors)
    #true_out = combine_images(tmp,cols=2,rows=2,randomize=False)
    #true_out.save('%s/input.png' % out_path)

    ################################################
    # SETUP MODELS #################################
   
    # initialise all models
    signal_discriminator = signal_discriminator_model()
    data_subtraction = data_subtraction_model(noise_signal)	# need to pass the measured data here
    generator = generator_model()
    if do_pe:
        signal_pe = signal_pe_model()    

    # setup generator training for when we subtract from the 
    # measured data and expect Gaussian residuals.
    # We use a mean squared error here since we want it to find 
    # the situation where the residuals have the known mean=0, std=n_sig properties
    data_subtraction_on_generator = generator_after_subtracting_noise(generator, data_subtraction)
    data_subtraction_on_generator.compile(loss='mean_squared_error', optimizer=Adam(lr=2e-4, beta_1=0.5), metrics=['accuracy'])

    # setup generator training when we pass the output to the signal discriminator
    signal_discriminator_on_generator = generator_containing_signal_discriminator(generator, signal_discriminator)
    set_trainable(signal_discriminator, False)	# set the discriminator as not trainable for this step
    signal_discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-4, beta_1=0.5), metrics=['accuracy'])

    # setup trainin on signal discriminator model
    # This uses a binary cross entropy loss since we are just 
    # discriminating between real and fake signals
    set_trainable(signal_discriminator, True)	# set it back to being trainable
    signal_discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-4, beta_1=0.5), metrics=['accuracy'])

    if do_pe:
        signal_pe.compile(loss='mean_squared_error', optimizer=Adam(lr=2e-4, beta_1=0.5), metrics=['accuracy'])

    # print the model summaries
    print(generator.summary())
    print(data_subtraction_on_generator.summary())
    print(signal_discriminator_on_generator.summary())
    print(signal_discriminator.summary())
    if do_pe:
        print(signal_pe.summary())

    ################################################
    # DO PARAMETER ESTIMATION ######################

    if do_pe:

        # first compute true PE on a grid
        x = np.linspace(0,1,n_pix)
        xy = np.array([k for k in itprod(x,x)]).reshape(n_pix*n_pix,2)
        L = []
        for pars in xy:
            template = np.array(gen_gauss_signals(1,pars=pars)[0]).reshape(n_pix,n_pix)
	    L.append(-0.5*np.sum(((noise_signal.reshape(n_pix,n_pix)-template)/n_sig)**2))
        L = np.array(L).reshape(n_pix,n_pix).transpose()
        L = np.exp(L-np.max(L))
        plot_pe_samples(None,signal_pars[0],L,'%s/pe_truelike.png' % out_path)
        print('Completed true grid PE')

        pe_losses = []         # initialise the losses for plotting
        i = 0
        rms = [1.0,1.0]
        while rms[0]>5e-4 or rms[1]>5e-4:
	
            # get random batch from images
            idx = random.sample(np.arange(signal_train_images.shape[0]),batch_size)
            signal_batch_images = signal_train_images[idx]
	    signal_batch_pars = signal_train_pars[idx]

            # train only the signal PE model on the data
            pe_loss = signal_pe.train_on_batch(signal_batch_images,signal_batch_pars)
	    pe_losses.append(pe_loss)

	    # output status and save images
            if ((i % pe_cadence == 0) & (i>0)):

		# plot loss curves - non-log and log
                plot_losses(pe_losses,'%s/pe_losses.png' % out_path,legend=['PE-GEN'])
                plot_losses(pe_losses,'%s/pe_losses_logscale.png' % out_path,logscale=True,legend=['PE-GEN'])

		# plot true vs predicted values for all training data
                pe_samples = signal_pe.predict(signal_train_images)
		plot_pe_accuracy(signal_train_pars,pe_samples,'%s/pe_accuracy%05d.png' % (out_path,i))
            
	        # compute RMS difference
                rms = [np.mean((signal_train_pars[:,k]-pe_samples[:,k])**2) for k in np.arange(2)]

                pe_mesg = "%d: [PE loss: %f, acc: %f, RMS: %f,%f]" % (i, pe_loss[0], pe_loss[1], rms[0], rms[1])
                print(pe_mesg)

            i += 1
 
        print('Completed CNN PE')

    ################################################
    # LOOP OVER BATCHES ############################

    losses = []		# initailise the losses for plotting 
    for i in range(max_iter):

	# get random batch from images, should be real signals
        signal_batch_images = sample_data(batch_size)
        
	# first use the generator to make fake images - this is seeded with a size 100 random vector
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        generated_images = generator.predict(noise)

	# make set of real and fake signal mages with labels
        signal_batch_images = np.reshape(signal_batch_images, (signal_batch_images.shape[0], 1, signal_batch_images.shape[1], 1))
        sX = np.concatenate((signal_batch_images, generated_images))
        sy = [1] * batch_size + [0] * batch_size

        # train only the signal discriminator on the data
        sd_loss = signal_discriminator.train_on_batch(sX, sy)

 	# next train the generator to make signals that have residuals (after
        # subtracting from the measured data) that have the correct Gaussian properties
	noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        ny = np.zeros((batch_size,2))	# initialise the expected residual means as zero 
        ny[:,1] = n_sig**2		# initialise the expected variances as n_sig squared
        ng_loss = data_subtraction_on_generator.train_on_batch(noise, ny)

	# finally train the generator to make images that look like signals
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
        sg_loss = signal_discriminator_on_generator.train_on_batch(noise, [1] * batch_size)

        # fill in the loss vector for plotting
        losses.append([sg_loss[0],sg_loss[1],sd_loss[0],sd_loss[1],ng_loss[0],ng_loss[1]])

	# output status and save images
	if ((i % cadence == 0) & (i>0)) or (i == max_iter):
            log_mesg = "%d: [sD loss: %f, acc: %f]" % (i, sd_loss[0], sd_loss[1])
	    log_mesg = "%s  [sG loss: %f, acc: %f]" % (log_mesg, sg_loss[0], sg_loss[1])
            log_mesg = "%s  [nG loss: %f, acc: %f]" % (log_mesg, ng_loss[0], ng_loss[1])
            print(log_mesg)
	    
            # plot original waveform
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)
            ax = signal_image
            ax1.plot(ax[0], color='cyan', linewidth=0.5)
            ax1.plot(noise_signal[0], color='green', linewidth=0.5)
            ax1.set_title('signal + (sig+noise)')

            # plot all noise training samples
            ax2.plot(noise[:25], alpha=0.25, color='blue', linewidth=0.5)
            ax2.set_title('Noise Samples')

            # plot generated signals - first image is the noise-free true signal
            ax3.plot(signal_image[0], color='cyan', linewidth=0.5)
            ax3.plot(np.transpose(np.reshape(generated_images[:25], (generated_images[:25].shape[0],generated_images[:25].shape[2]))), color='blue', alpha=0.25, linewidth=0.5)
            ax3.plot(noise_signal[0], color='green', linewidth=0.5)
            ax3.set_title('gen + sig + (sig+noise)')
	    #image = combine_images(generated_images,extra=signal_image.reshape(n_pix,n_pix,n_colors))
            #image.save('%s/gen_signal%05d.png' % (out_path,i))
	    
	    # plot residuals - generated images subtracted from the measured image
            # the first image is the true noise realisation
            residuals = np.transpose(noise_signal-np.reshape(generated_images[:25], (generated_images[:25].shape[0],generated_images[:25].shape[2])))
            ax4.plot((residuals), color='red', alpha=0.25, linewidth=0.5)
            ax4.set_title('Residuals')
            #image = combine_images(renorm(noise_signal-generated_images),extra=noise_image.reshape(n_pix,n_pix,n_colors)) 
            #image.save('%s/residual%05d.png' % (out_path,i))

            # save waveforms plot
            plt.savefig('%s/waveform_results%05d.png' % (out_path,i), dpi=500)
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

            # plot loss curves - non-log and log
            plot_losses(losses,'%s/losses.png' % out_path,legend=['S-GEN','S-DIS','N-GEN'])
            plot_losses(losses,'%s/losses_logscale.png' % out_path,logscale=True,legend=['S-GEN','S-DIS','N-GEN'])
	    #plot_losses(losses,"%s/losses_logscale.png' % out_path,logscale=True,legend=['S-GEN','S-DIS','N-GEN'])

	    # plot posterior samples
            if do_pe:
                # first use the generator to make MANY fake images
        	noise = np.random.uniform(size=[1000, 100], low=-1.0, high=1.0)
        	more_generated_images = generator.predict(noise)
                pe_samples = signal_pe.predict(more_generated_images)
                plot_pe_samples(pe_samples,signal_pars[0],L,'%s/pe_samples%05d.png' % (out_path,i))

	    # save trained models
            if save_models:
	        generator.save_weights('generator.h5', True)
                discriminator.save_weights('discriminator.h5', True)
                if do_pe:
                    signal_pe.save_weights('signal_pe.h5', True)

main()
