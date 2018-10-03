#!/usr/local/bin/python
from __future__ import division
import pandas as pd
import cPickle
import numpy as np
from scipy import integrate, interpolate
from scipy.misc import imsave
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import lal
import lalsimulation
from pylal import antenna, cosmography
import argparse
import time
from scipy.signal import filtfilt, butter
from scipy.optimize import brentq
from lal import MSUN_SI, C_SI, G_SI
import os
from sys import exit
import scipy
import pickle
from sympy import Eq, Symbol, solve


safe = 2    # define the safe multiplication scale for the desired time length
verb = False
gw_tmp = True
batch_size = 3170
# load gan pe samples. TEMPORARY!!!
with open('gan_pe_samples.sav', 'rb') as rfp:
    gan_pe_post = np.array(pickle.load(rfp))
with open('gan_pe_waveforms.sav', 'rb') as rfp:
    gan_pe_waveforms = np.array(pickle.load(rfp))
# load in lalinference m1 and m2 parameters
pickle_lalinf_pars = open("data/gw150914_mc_q_lalinf_post.sav")
lalinf_pars = pickle.load(pickle_lalinf_pars)
post_q = gan_pe_post[1]
post_mc = gan_pe_post[0]

gan_post = pickle.load(open('/home/hunter.gabbard/Burst/GenNet/BBH_version/data/gw150914_m1_m2_lainf_post.sav'))
out_path = '/home/hunter.gabbard/public_html/CBC/mahoGANy/gw150914_template'
gw150914_posteriors = pd.read_hdf('/home/hunter.gabbard/Burst/GenNet/BBH_version/data/lalinf_nest_gw150914-like_allbutmassfixed/posterior_H1_1126259462-0.hdf5','lalinference/lalinference_nest/posterior_samples')

gan_post = np.transpose(gan_post)

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

def tukey(M,alpha=0.5):
    """
    Tukey window code copied from scipy
    """
    n = np.arange(0, M)
    width = int(np.floor(alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
    w = np.concatenate((w1, w2, w3))

    return np.array(w[:M])

def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='data_prep.py',description='generates GW data for application of deep learning networks.')

    # arguments for reading in a data file
    parser.add_argument('-N', '--Nsamp', type=int, default=3170, help='the number of samples')
    parser.add_argument('-Nn', '--Nnoise', type=int, default=0, help='the number of noise realisations per signal, if 0 then signal only')
    parser.add_argument('-Nb', '--Nblock', type=int, default=3170, help='the number of training samples per output file')
    parser.add_argument('-f', '--fsample', type=int, default=1024, help='the sampling frequency (Hz)')
    parser.add_argument('-T', '--Tobs', type=int, default=2, help='the observation duration (sec)')
    parser.add_argument('-s', '--snr', type=float, default=56, help='the signal integrated SNR')   
    parser.add_argument('-I', '--detectors', type=str, nargs='+',default=['H1'], help='the detectors to use')   
    parser.add_argument('-b', '--basename', type=str,default='templates/', help='output file path and basename')
    parser.add_argument('-m', '--mdist', type=str, default='astro', help='mass distribution for training (astro,gh,metric)')
    parser.add_argument('-z', '--seed', type=int, default=1, help='the random seed')

    return parser.parse_args()

def convert_beta(beta,fs,T_obs):
    """
    Converts beta values (fractions defining a desired period of time in
    central output window) into indices for the full safe time window
    """
    # pick new random max amplitude sample location - within beta fractions
    # and slide waveform to that location
    newbeta = np.array([(beta[0] + 0.5*safe - 0.5),(beta[1] + 0.5*safe - 0.5)])/safe
    low_idx = int(T_obs*fs*newbeta[0])
    high_idx = int(T_obs*fs*newbeta[1])

    return low_idx,high_idx

def gen_noise(fs,T_obs,psd):
    """
    Generates noise from a psd
    """

    N = T_obs * fs          # the total number of time samples
    Nf = N // 2 + 1
    dt = 1 / fs             # the sampling time (sec)
    df = 1 / T_obs

    amp = np.sqrt(0.25*T_obs*psd)
    idx = np.argwhere(psd==0.0)
    amp[idx] = 0.0
    re = amp*np.random.normal(0,1,Nf)
    im = amp*np.random.normal(0,1,Nf)
    re[0] = 0.0
    im[0] = 0.0
    x = N*np.fft.irfft(re + 1j*im)*df

    return x

def gen_psd(fs,T_obs,op='AdvDesign',det='H1'):
    """
    generates noise for a variety of different detectors
    """
    N = T_obs * fs          # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)
    df = 1 / T_obs          # the frequency resolution
    psd = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0, df,lal.HertzUnit, N // 2 + 1)    

    if det=='H1' or det=='L1':
        if op == 'AdvDesign':
            lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd, 10.0)
        elif op == 'AdvEarlyLow':
            lalsimulation.SimNoisePSDAdVEarlyLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvEarlyHigh':
            lalsimulation.SimNoisePSDAdVEarlyHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidLow':
            lalsimulation.SimNoisePSDAdVMidLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidHigh':
            lalsimulation.SimNoisePSDAdVMidHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateLow':
            lalsimulation.SimNoisePSDAdVLateLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateHigh':
            lalsimulation.SimNoisePSDAdVLateHighSensitivityP1200087(psd, 10.0)
        else:
            print 'unknown noise option'
            exit(1)
    else:
        print 'unknown detector - will add Virgo soon'
        exit(1) 

    return psd

def get_snr(data,T_obs,fs,psd,fmin):
    """
    computes the snr of a signal given a PSD starting from a particular frequency index
    """

    N = T_obs*fs
    df = 1.0/T_obs
    dt = 1.0/fs
    fidx = int(fmin/df)

    win = tukey(N,alpha=1.0/8.0)
    idx = np.argwhere(psd>0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0/psd[idx]

    xf = np.fft.rfft(data*win)*dt
    SNRsq = 4.0*np.sum((np.abs(xf[fidx:])**2)*invpsd[fidx:])*df
    return np.sqrt(SNRsq)

def whiten_data(data,duration,sample_rate,psd,flag='td'):
    """
    Takes an input timeseries and whitens it according to a psd
    """

    if flag=='td':
        # FT the input timeseries - window first
        win = tukey(duration*sample_rate,alpha=1.0/8.0)
        xf = np.fft.rfft(win*data)
    else:
        xf = data

    # deal with undefined PDS bins and normalise
    idx = np.argwhere(psd>0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0/psd[idx]
    xf *= np.sqrt(2.0*invpsd/sample_rate)

    # Detrend the data: no DC component.
    xf[0] = 0.0

    if flag=='td':
        # Return to time domain.
        x = np.fft.irfft(xf)
        return x
    else:
        return xf

def gen_masses(m_min=5.0,M_max=100.0,mdist='astro'):
    """
    function returns a pair of masses drawn from the appropriate distribution
    """
    
    flag = False
    if mdist=='astro':
        if verb: print '{}: using astrophysical logarithmic mass distribution'.format(time.asctime())
        new_m_min = m_min
        new_M_max = M_max
	log_m_max = np.log(new_M_max - new_m_min)
        while not flag:
            m12 = np.exp(np.log(new_m_min) + np.random.uniform(0,1,2)*(log_m_max-np.log(new_m_min)))
            flag = True if (np.sum(m12)<new_M_max) and (np.all(m12>new_m_min)) and (m12[0]>=m12[1]) else False
        eta = m12[0]*m12[1]/(m12[0]+m12[1])**2
        mc = np.sum(m12)*eta**(3.0/5.0)
        return m12, mc, eta
    elif mdist=='gh':
        if verb: print '{}: using George & Huerta mass distribution'.format(time.asctime())
	m12 = np.zeros(2)
        while not flag:
            q = np.random.uniform(1.0,10.0,1)
            m12[1] = np.random.uniform(5.0,75.0,1)
            m12[0] = m12[1]*q
            flag = True if (np.all(m12<75.0)) and (np.all(m12>5.0)) and (m12[0]>=m12[1]) else False
	eta = m12[0]*m12[1]/(m12[0]+m12[1])**2
    	mc = np.sum(m12)*eta**(3.0/5.0)
	return m12, mc, eta
    elif mdist=='metric':
	if verb: print '{}: using metric based mass distribution'.format(time.asctime())
        new_m_min = m_min
	new_M_max = M_max
	new_M_min = 2.0*new_m_min
	eta_min = m_min*(new_M_max-new_m_min)/new_M_max**2
	while not flag:
	    M = (new_M_min**(-7.0/3.0) - np.random.uniform(0,1,1)*(new_M_min**(-7.0/3.0) - new_M_max**(-7.0/3.0)))**(-3.0/7.0)
    	    eta = (eta_min**(-2.0) - np.random.uniform(0,1,1)*(eta_min**(-2.0) - 16.0))**(-1.0/2.0)
            m12 = np.zeros(2)
	    m12[0] = 0.5*M + M*np.sqrt(0.25-eta)
	    m12[1] = M - m12[0]
	    flag = True if (np.sum(m12)<new_M_max) and (np.all(m12>new_m_min)) and (m12[0]>=m12[1]) else False	
        mc = np.sum(m12)*eta**(3.0/5.0)
	return m12, mc, eta
    else:
	print '{}: ERROR, unknown mass distribution. Exiting.'.format(time.asctime())
	exit(1)

def get_fmin(M,eta,dt):
    """
    Compute the instantaneous frequency given a time till merger
    """
    M_SI = M*MSUN_SI

    def dtchirp(f):
        """
        The chirp time to 2nd PN order
        """
        v = ((G_SI/C_SI**3)*M_SI*np.pi*f)**(1.0/3.0)
        temp = (v**(-8.0) + ((743.0/252.0) + 11.0*eta/3.0)*v**(-6.0) -
                (32*np.pi/5.0)*v**(-5.0) + ((3058673.0/508032.0) + 5429*eta/504.0 +
                (617.0/72.0)*eta**2)*v**(-4.0))
        return (5.0/(256.0*eta))*(G_SI/C_SI**3)*M_SI*temp - dt

    # solve for the frequency between limits
    fmin = brentq(dtchirp, 1.0, 2000.0, xtol=1e-6)
    if verb: print '{}: signal enters segment at {} Hz'.format(time.asctime(),fmin)

    return fmin

def gen_par(fs,T_obs,index,mdist='astro',beta=[0.75,0.95],gw_tmp=False):
    """
    Generates a random set of parameters
    """
    # define distribution params
    m_min = 5.0         # 5 rest frame component masses
    M_max = 100.0       # 100 rest frame total mass
    log_m_max = np.log(M_max - m_min)

    #m12, mc, eta = gen_masses(m_min,M_max,mdist=mdist)
    m12 = [gan_post[index,1],gan_post[index,0]]
    #if index==0:
    #    idx = np.argsort(post_q,axis=0)[200][0]
    #    m1 = Symbol('m1')
    #    eqn_m1 = Eq((m1 + (m1/post_q[idx])) * (m1*(m1/post_q[idx])/(m1+(m1/post_q[idx]))**2)**(3.0/5.0), post_mc[idx])
    #    post_m1 = float(solve(eqn_m1)[0])

    #    post_m2 = ((float(post_q[idx]))*post_m1)
    #    print(post_m1,post_m2)
    #    m12 = [post_m1,post_m2]
    #if index==1:
    #    index = int(np.random.uniform(low=0,high=4390))
    #    index = np.argmin(gan_post[:,0]/gan_post[index,1])
    #    m12 = [gan_post[index,1],gan_post[index,0]]
    #    print(m12[1]/m12[0])
    #    exit()
    #m12 = [36.0, 29.0]
    eta = m12[0]*m12[1]/(m12[0]+m12[1])**2
    #mc = np.sum(m12)*eta**(3.0/5.0)
    mc = gw150914_posteriors['mc'][index]
    M = np.sum(m12)
    if verb: print '{}: selected bbh masses = {},{} (chirp mass = {})'.format(time.asctime(),m12[0],m12[1],mc)

    # generate iota
    #iota = np.arccos(-1.0 + 2.0*np.random.rand())
    #iota = np.arccos(gw150914_posteriors['costheta_jn'][index])
    if verb: print '{}: selected bbh cos(inclination) = {}'.format(time.asctime(),np.cos(iota))

    # generate polarisation angle
    #psi = 2.0*np.pi*np.random.rand()
    #psi = gw150914_posteriors['psi'][index]
    if verb: print '{}: selected bbh polarisation = {}'.format(time.asctime(),psi)

    # generate reference phase
    #phi = 2.0*np.pi*np.random.rand()
    #phi = gw150914_posteriors['phase_maxl'][index] # phase_maxl is best
    if verb: print '{}: selected bbh reference phase = {}'.format(time.asctime(),phi)

    # pick sky position - uniform on the 2-sphere
    #ra = 2.0*np.pi*np.random.rand()
    #ra = gw150914_posteriors['ra'][index]
    #dec = np.arcsin(-1.0 + 2.0*np.random.rand())
    #dec = gw150914_posteriors['dec'][index]
    if verb: print '{}: selected bbh sky position = {},{}'.format(time.asctime(),ra,dec)

    ra = 2.21535724066
    dec = -1.23649695537
    iota = 2.5#-0.8011436155469337
    phi = 1.5
    psi = 1.75

    # pick new random max amplitude sample location - within beta fractions
    # and slide waveform to that location
    low_idx,high_idx = convert_beta(beta,fs,T_obs)
    if low_idx==high_idx:
        idx = low_idx
    else:
        idx = int(np.random.randint(low_idx,high_idx,1)[0])
    if verb: print '{}: selected bbh peak amplitude time = {}'.format(time.asctime(),idx/fs)

    # the start index of the central region
    sidx = int(0.5*fs*T_obs*(safe-1.0)/safe)

    # compute SNR of pre-whitened data
    fmin = get_fmin(M,eta,int(idx-sidx)/fs)
    #print((((T_obs * fs)) * (1126259462.0 - gw150914_posteriors['time'][index])))
    #idx = int((T_obs * fs) / 2) - 4 
    #idx = int(((T_obs * fs) / 2) + (((T_obs * fs)/2) * (1126259462.0 - gw150914_posteriors['time'][index]))) - 4
    if verb: print '{}: computed starting frequency = {} Hz'.format(time.asctime(),fmin)

    # store params
    par = bbhparams(mc,M,eta,m12[0],m12[1],ra,dec,iota,phi,psi,idx,fmin,None,None)

    """
    Only if you want to gen template like gw150914. Will be first template in 
    set.
    """
    if gw_tmp:
        idx = int((T_obs * fs) / 2) - 4
        m1, m2 = 36.0, 29.0
        eta = m1*m2/(m1+m2)**2
        M = m1 + m2
        mc = M*eta**(3.0/5.0)
        mc = 30.0
        fmin = get_fmin(M,eta,int(idx-sidx)/fs)
        
        ra = 2.21535724066
        dec = -1.23649695537
        iota = 2.5#-0.8011436155469337  
        phi = 1.5
        psi = 1.75
        
        par = bbhparams(mc,M,eta,m1,m2,ra,dec,iota,phi,psi,idx,fmin,None,None)

    return par

def gen_bbh(fs,T_obs,idx,psds,snr=1.0,dets=['H1'],beta=[0.75,0.95],par=None, gw_tmp=False):
    """
    generates a BBH timedomain signal
    """
    N = T_obs * fs      # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)
    f_low = 40            # lowest frequency of waveform (Hz)
    amplitude_order = 0
    phase_order = 7
    f_max = 512
    approximant = lalsimulation.IMRPhenomPv2
    #dist = gw150914_posteriors['dist'][idx] * 1e6*lal.PC_SI
    dist = 410e6 * lal.PC_SI
    if gw_tmp:
        dist = 410e6 * lal.PC_SI    

    # make waveform
    # loop until we have a long enough waveform - slowly reduce flow as needed
    flag = False
    while not flag:
        hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
                    par.m1 * lal.MSUN_SI, par.m2 * lal.MSUN_SI,
                    0, 0, 0, 0, 0, 0,
                    dist,
                    par.iota, par.phi, 0,
                    0, 0,
                    1 / T_obs,
                    f_low,f_max,f_low,
                    lal.CreateDict(),
                    approximant)
        #flag = True if hp.data.length>2*N else False
        flag = True
        f_low -= 1       # decrease by 1 Hz each time
    #orig_hp = hp.data.data
    #orig_hc = hc.data.data
    whiten_hp = whiten_data(hp.data.data,T_obs,fs,psds,flag='fd')
    whiten_hc = whiten_data(hc.data.data,T_obs,fs,psds,flag='fd')

    #orig_hp = np.roll(np.fft.irfft(hp.data.data.real + 1j*hp.data.data.imag,T_obs*fs),-fs)
    #orig_hc = np.roll(np.fft.irfft(hc.data.data.real + 1j*hc.data.data.imag,T_obs*fs),-fs)
    orig_hp = np.roll(np.fft.irfft(whiten_hp,T_obs*fs),int(-fs))
    orig_hc = np.roll(np.fft.irfft(whiten_hc,T_obs*fs),int(-fs))

    # compute reference idx
    ref_idx = np.argmax(orig_hp**2 + orig_hc**2)

    # the start index of the central region
    sidx = int(0.5*fs*T_obs*(safe-1.0)/safe)

    # make aggressive window to cut out signal in central region
    # window is non-flat for 1/8 of desired Tobs
    # the window has dropped to 50% at the Tobs boundaries
    win = np.zeros(N)
    tempwin = tukey(int((16.0/15.0)*N/safe),alpha=1.0/8.0)
    win[int((N-tempwin.size)/2):int((N-tempwin.size)/2)+tempwin.size] = tempwin

    # loop over detectors
    ndet = 1
    ts = np.zeros((ndet,N))
    hp = np.zeros((ndet,N))
    hc = np.zeros((ndet,N))
    intsnr = []
    j = 0
    for det in dets:

    	# make signal - apply antenna and shifts
    	ht_shift, hp_shift, hc_shift = make_bbh(orig_hp,orig_hc,fs,par.ra,par.dec,par.psi,det,idx,gw_tmp)

    	# place signal into timeseries - including shift
    	ht_temp = ht_shift[int(ref_idx-par.idx):]
    	hp_temp = hp_shift[int(ref_idx-par.idx):]
    	hc_temp = hc_shift[int(ref_idx-par.idx):]
    	if len(ht_temp)<N:
            ts[j,:len(ht_temp)] = ht_temp
            hp[j,:len(ht_temp)] = hp_temp
            hc[j,:len(ht_temp)] = hc_temp
        else:
            ts[j,:] = ht_temp[:N]
            hp[j,:] = hp_temp[:N]
            hc[j,:] = hc_temp[:N]


    	# apply aggressive window to cut out signal in central region
    	# window is non-flat for 1/8 of desired Tobs
    	# the window has dropped to 50% at the Tobs boundaries
    	ts[j,:] *= win
    	hp[j,:] *= win
    	hc[j,:] *= win

        # compute SNR of pre-whitened data
        intsnr.append(get_snr(ts[j,:],T_obs,fs,psds,par.fmin))

    # normalise the waveform using either integrated or peak SNR
    #intsnr = np.array(intsnr)
    #scale = snr/np.sqrt(np.sum(intsnr**2))
    #ts *= scale
    #hp *= scale
    #hc *= scale
    #intsnr *= scale
    if verb: print '{}: computed the network SNR = {}'.format(time.asctime(),snr)

    return ts, hp, hc

def make_bbh(hp,hc,fs,ra,dec,psi,det,index,gw_tmp=False):
    """
    turns hplus and hcross into a detector output
    applies antenna response and
    and applies correct time delays to each detector
    """

    # make basic time vector
    tvec = np.arange(len(hp))/float(fs)

    # compute antenna response and apply gw150914_posteriors['time'][index]
    Fp,Fc,_,_ = antenna.response( 1126259462.0, ra, dec, 0, psi, 'radians', det )
    if gw_tmp:
        Fp,Fc,_,_ = antenna.response(1126259462.0, ra, dec, 0, psi, 'radians', det )
    ht = hp*Fp + hc*Fc     # overwrite the timeseries vector to reuse it

    # compute time delays relative to Earth centre
    frDetector =  lalsimulation.DetectorPrefixToLALDetector(det)
    tdelay = lal.TimeDelayFromEarthCenter(frDetector.location,ra,dec,1126259462.0)
    if gw_tmp:
        tdelay = lal.TimeDelayFromEarthCenter(frDetector.location,ra,dec,1126259462.0)
    if verb: print '{}: computed {} Earth centre time delay = {}'.format(time.asctime(),det,tdelay)

    # interpolate to get time shifted signal
    ht_tck = interpolate.splrep(tvec, ht, s=0)
    hp_tck = interpolate.splrep(tvec, hp, s=0)
    hc_tck = interpolate.splrep(tvec, hc, s=0)
    if gw_tmp: tnew = tvec - tdelay
    else: tnew = tvec - tdelay# + ((1126259462.0 - gw150914_posteriors['time'][index]))
    new_ht = interpolate.splev(tnew, ht_tck, der=0,ext=1)
    new_hp = interpolate.splev(tnew, hp_tck, der=0,ext=1)
    new_hc = interpolate.splev(tnew, hc_tck, der=0,ext=1)

    return new_ht, new_hp, new_hc    

def sim_data(fs,T_obs,psds,snr=1.0,dets=['H1'],Nnoise=25,size=1000,mdist='astro',beta=[0.75,0.95]):
    """
    Simulates all of the test, validation and training data timeseries
    """

    yval = []       # initialise the param output
    ts = []         # initialise the timeseries output
    par = []        # initialise the parameter output
    nclass = 1	    # the hardcoded number of classes
    npclass = int(size/float(nclass))    
    ndet = len(dets)               # the number of detectors
    #psds = [gen_psd(fs,T_obs,op='AdvDesign',det=d) for d in dets]

    # for the noise class
    #for x in xrange(npclass):

    #    ts_new = np.array([gen_noise(fs,T_obs,psd.data.data) for psd in psds]).reshape(ndet,-1)
    #ts.append(np.array([whiten_data(t,T_obs,fs,psd.data.data) for t,psd in zip(ts_new,psds)]).reshape(ndet,-1))
#	par.append(None)
#	yval.append(0)
#	if verb: print '{}: completed {}/{} noise samples'.format(time.asctime(),x+1,npclass)

    # for the signal class - loop over random masses
    cnt = 0
    if gw_tmp:
        size = size -1
    par_total = [] 
    while cnt<size:

        print '{}: making waveform {}/{}'.format(time.asctime(),cnt,size)
        # generate a single new timeseries and chirpmass
    
        # don't allow negative masses
        #if gan_post[cnt,0] < 0 or gan_post[cnt,1] < 0:
        #    cnt+=1
        #    continue
        par_new = gen_par(fs,T_obs,cnt,mdist=mdist,beta=beta,gw_tmp=False)
        ts_new,_,_ = gen_bbh(fs,T_obs,cnt,psds,snr=snr,dets=dets,beta=beta,par=par_new)

        #if cnt == 0:
        #    cnt = 0
        #    gan_par = gen_par(fs,T_obs,cnt,mdist=mdist,beta=beta,gw_tmp=False)
        #    gan_ts,_,_ = gen_bbh(fs,T_obs,cnt,psds,snr=snr,dets=dets,beta=beta,par=gan_par)
               
            
        #elif cnt == 1:
        #    cnt += 1
        #    continue
            #cnt = 1
            #lal_par = gen_par(fs,T_obs,cnt,mdist=mdist,beta=beta,gw_tmp=False)
            #lal_ts,_,_ = gen_bbh(fs,T_obs,cnt,psds,snr=snr,dets=dets,beta=beta,par=lal_par)

            # plot lalinf and gan waveforms
       #     plt.plot(gan_pe_waveforms[np.argsort(post_q,axis=0)][200][0],color='c',alpha=0.5)
       #     plt.plot(gan_ts[0][int(((T_obs/2)*fs)-fs/2):int(((T_obs/2)*fs)+fs/2)] * 817.98,color='g',alpha=0.5)
       #     plt.savefig('/home/hunter.gabbard/public_html/CBC/mahoGANy/gw150914_template/test.png')
       #     plt.close()

	# loop over noise realisations
        if Nnoise>0:
	    for j in xrange(Nnoise):
	        ts_noise = np.array([gen_noise(fs,T_obs,psd.data.data) for psd in psds]).reshape(ndet,-1)
                ts.append(np.array([whiten_data(t,T_obs,fs,psd.data.data) for t,psd in zip(ts_noise+ts_new,psds)]).reshape(ndet,-1))
                par.append(par_new)
                yval.append(1)
                cnt += 1
	    if verb: print '{}: completed {}/{} signal samples'.format(time.asctime(),cnt-npclass,int(size/2))        
	else:
            # just generate noise free signal
	    ts.append(np.array([t[int(((T_obs/2)*fs)-fs/2):int(((T_obs/2)*fs)+fs/2)] for t in ts_new]).reshape(ndet,-1))
            #ts.append(np.array([whiten_data(t,T_obs,fs,psds)[int(((T_obs/2)*fs)-fs/2):int(((T_obs/2)*fs)+fs/2)] for t in ts_new]).reshape(ndet,-1))
	    par.append(par_new)
	    yval.append(1)
            cnt += 1
            if verb: print '{}: completed {}/{} signal samples'.format(time.asctime(),cnt-npclass,int(size/2))

        # only generate waveforms up to batch size
        if len(ts) == batch_size - 1:
            size = batch_size - 1 
            break
        #tdelay = ((1126259462.0 - gw150914_posteriors['time'][cnt]))
        #par_total.append(gw150914_posteriors['time'][cnt])


    # trim the data down to desired length
    ts = np.array(ts)[:size]
    yval = np.array(yval)[:size]
    par = par[:size]

    # return randomised the data
    idx = np.random.permutation(size)
    temp = [par[i] for i in idx]
    ts, yval = ts[idx], yval[idx]

    # add gw150914 template at end of set
    if gw_tmp:
        print '{}: making waveform {}/{}'.format(time.asctime(),size+1,size+1)
        # generate a single new timeseries and chirpmass
        par_new = gen_par(fs,T_obs,cnt,mdist=mdist,beta=beta,gw_tmp=gw_tmp)
        ts_new,_,_ = gen_bbh(fs,T_obs,cnt,psds,snr=snr,dets=dets,beta=beta,par=par_new,gw_tmp=gw_tmp)

        # just generate noise free signal
        ts = np.concatenate((ts,np.array([t[int(((T_obs/2)*fs)-fs/2):int(((T_obs/2)*fs)+fs/2)] for t in ts_new]).reshape(ndet,-1).reshape(1,1,fs)))
        #ts = np.concatenate((ts,np.array([whiten_data(t,T_obs,fs,psds)[int(((T_obs/2)*fs)-fs/2):int(((T_obs/2)*fs)+fs/2)] for t in ts_new]).reshape(ndet,-1).reshape(1,1,fs)))
        temp.append(par_new)
        yval = np.append(yval,1)

        # par = bbhparams(mc,M,eta,m12[0],m12[1],ra,dec,iota,phi,psi,idx,fmin,None,None)
        
        #plt.hist(par_total, bins=100)
        #plt.title('time')
        #plt.axvline(x=par_new.iota, color='k', linestyle='--')
        #plt.savefig('/home/hunter.gabbard/public_html/CBC/mahoGANy/gw150914_template/gan_pe_hist.png')
        #plt.close()
        

    return [ts, yval], temp

# the main part of the code
def main():
    """
    The main code - generates the training, validation and test samples
    """
    snr_mn = 0.0
    snr_cnt = 0
    lalinf_out_loc = '/home/hunter.gabbard/Burst/GenNet/BBH_version/data/lalinf_nest_gw150914-like_allbutmassfixed'

    # get the command line args
    args = parser()
    if args.seed>0:
        np.random.seed(args.seed)
    safeTobs = safe*args.Tobs

    # load gw150914 time series and unwhitened psd
    unwht_wvf_file = np.loadtxt('%s/lalinferencenest-0-H1-1126259462.0-0.hdf5H1-freqData.dat' % lalinf_out_loc)[:,1:]
    sig_unwht_wvf_file = np.loadtxt('%s/lalinferencenest-0-H1-1126259462.0-0.hdf5H1-freqDataWithInjection.dat' % lalinf_out_loc)[:,1:]   
    unwht_wvf_file = np.add(unwht_wvf_file[:,0],1j*unwht_wvf_file[:,1])
    sig_unwht_wvf_file = np.add(sig_unwht_wvf_file[:,0],1j*sig_unwht_wvf_file[:,1]) 

    # set all NaN values in frequency series to zero
    sig_unwht_wvf_file[np.isnan(sig_unwht_wvf_file) == True] = 0+0*1j
    unwht_wvf_file[np.isnan(unwht_wvf_file) == True] = 0+0*1j

    # get gw150914 template
    h_t = sig_unwht_wvf_file - unwht_wvf_file
    wvf_psd_file = np.loadtxt('%s/lalinferencenest-0-H1-1126259462.0-0.hdf5H1-PSD.dat' % lalinf_out_loc)

    unwht_wvf_file = sig_unwht_wvf_file

    # whiten gw150914
    wht_wvf = whiten_data(unwht_wvf_file,4,args.fsample,wvf_psd_file[:,1],flag='fd')
    wht_wvf = np.fft.irfft(wht_wvf,4096)
    h_t = whiten_data(h_t,4,args.fsample,wvf_psd_file[:,1],'fd')
    h_t = np.fft.irfft(h_t,4096)

    wht_wvf = wht_wvf[int(((safeTobs/2)*args.fsample)-args.fsample/2.0):int(((safeTobs/2)*args.fsample)+args.fsample/2.0)]
    h_t = h_t[int(((safeTobs/2)*args.fsample)-args.fsample/2.0):int(((safeTobs/2)*args.fsample)+args.fsample/2.0)]

    # set psd to the same used in lalinference for gw150914
    psd = wvf_psd_file[:,1]

    # break up the generation into blocks of args.Nblock training samples
    nblock = int(np.ceil(float(args.Nsamp)/float(args.Nblock)))
    for i in xrange(nblock):

    	# simulate the dataset and randomise it
        # only use Nnoise for the training data NOT the validation and test
    	print '{}: starting to generate data'.format(time.asctime())
    	ts, par = sim_data(args.fsample,safeTobs,psd,args.snr,args.detectors,args.Nnoise,size=args.Nblock,mdist=args.mdist,beta=[0.4989,0.4989])
    	print '{}: completed generating data {}/{}'.format(time.asctime(),i+1,nblock)

        # plot waveforms
        
        # plot actual waveforms
        for n in range(ts[0].shape[0]):
            ts[0][n,0,:] *= 817.98
            #plt.plot(ts[0][n,0,:], alpha=0.5, color='green')

        # compute percentile curves
        perc_90 = []
        perc_75 = []
        perc_25 = []
        perc_5 = []
        for n in range(ts[0].shape[2]):
            perc_ts = np.array(ts[0][:,0,n])
            perc_90.append(np.percentile(perc_ts, 90))
            perc_75.append(np.percentile(perc_ts, 75))
            perc_25.append(np.percentile(perc_ts, 25))
            perc_5.append(np.percentile(perc_ts, 5))

        #plt.plot(ts[0][-1,0,:], alpha=0.5, color='green', linewidth=0.5)
        plt.plot(h_t * 817.98, alpha=0.5, color='cyan', linewidth=0.5) #1079.22
        plt.fill_between(np.linspace(0,len(perc_90),num=len(perc_90)),perc_90, perc_5, lw=0,facecolor='#d5d8dc')
        plt.fill_between(np.linspace(0,len(perc_75),num=len(perc_75)),perc_75, perc_25, lw=0,facecolor='#808b96')
        plt.title('gen + sig + (sig+noise)')
        plt.savefig('%s/gan_pe_waveforms.png' % (out_path), dpi=500)
        plt.close()
        # load keras CNN model
        #signal_pe = keras.models.load_model('best_models/signal_pe.h5')    
        # predict pe samples
        ts = np.reshape(ts[0],(ts[0].shape[0],ts[0].shape[2]))
        #pe_samples = signal_pe.predict(ts)
        #pe_samples[:,0] *= 50.0

    	# pickle the results
    	# save the timeseries data to file
    	f = open('data/cnn_sanity_check_ts.sav', 'wb')
    	cPickle.dump(ts, f, protocol=cPickle.HIGHEST_PROTOCOL)
    	f.close()
    	print '{}: saved timeseries data to file'.format(time.asctime())

    	# save the sample parameters to file
    	#f = open(args.basename + '_params_' + str(i) + '_8000Samp'+ '.sav', 'wb')
    	#cPickle.dump(par, f, protocol=cPickle.HIGHEST_PROTOCOL)
    	#f.close()
    	#print '{}: saved parameter data to file'.format(time.asctime())

        # save gw150914 whitened waveform
        #f = open('data/' + 'gw150914' + str(i) + '.sav', 'wb')
        #cPickle.dump(wht_wvf, f, protocol=cPickle.HIGHEST_PROTOCOL)
        #f.close()
        #print '{}: saved gw150914 timeseries data to file'.format(time.asctime())

    print '{}: success'.format(time.asctime())

if __name__ == "__main__":
    exit(main())
