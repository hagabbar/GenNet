#!/usr/local/bin/python

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
This is a script which produces template waveforms to be used during the training 
of both the GAN and the CNN. This script assumes that you are rusing Python 2.7
'''

from __future__ import division
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
import pandas as pd

# define global variables
safe = 2                       # define the safe multiplication scale for the desired time length
verb = False                   # if True: prints out more information during script
gw_tmp = True                  # add your own gw150914-like template at the end of array
do_time_grid = False           # multiple merger time realizations for each signal
oversamp = True                # choose whether or not to oversample mass region of interest
N_time_grid = 25               # number of time realizations per mass sample if do_time_grid true
sample_num = 50000             # number of mass samples to make
event_name = 'gw150914'        # name of gw event
event_time = '1126259462'      # event time
tag = '_srate-1024hz_oversamp' # special identifier for input files
# location of lalinference output files
lalinf_out_loc = '/home/hunter.gabbard/parameter_estimation/john_bayesian_tutorial/injection_run_mass-time-varry_%s_srate-1024/lalinferencenest/engine' % event_name
outdir = '/home/hunter.gabbard/public_html/CBC/mahoGANy/gw150914_template/' # location of output directory for files


class bbhparams:
    """ class used to define bbh template parameters.
    """
    def __init__(self,mc,M,eta,m1,m2,ra,dec,iota,phi,psi,idx,snr,SNR):
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
        self.snr = snr
        self.SNR = SNR

def tukey(M,alpha=0.5):
    """ Tukey window code copied from scipy.

    Parameters
    ----------
    M:
        Number of points in the output window.
    alpha:
        The fraction of the window inside the cosine tapered region.

    Returns
    -------
    w:
        The window
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
    parser.add_argument('-N', '--Nsamp', type=int, default=sample_num, help='the number of samples')
    parser.add_argument('-Nn', '--Nnoise', type=int, default=0, help='the number of noise realisations per signal, if 0 then signal only')
    parser.add_argument('-Nb', '--Nblock', type=int, default=sample_num, help='the number of training samples per output file')
    parser.add_argument('-f', '--fsample', type=int, default=1024, help='the sampling frequency (Hz)')
    parser.add_argument('-T', '--Tobs', type=int, default=2, help='the observation duration (sec)')
    parser.add_argument('-b', '--basename', type=str,default='templates/', help='output file path and basename')
    parser.add_argument('-m', '--mdist', type=str, default='astro', help='mass distribution for training (astro,gh,metric)')
    parser.add_argument('-z', '--seed', type=int, default=1, help='the random seed')

    return parser.parse_args()


def convert_beta(beta,fs,T_obs):
    """ Converts beta values (fractions defining a desired period of time in
    central output window) into indices for the full safe time window

    Parameters
    ----------
    beta:
        fractional range for placement of signal in time series
    fs:
        sampling frequency
    T_obs:
        observation time window (s).

    Returns
    -------
    low_idx:
        lower index
    high_idx:
        higher index
    """
    # pick new random max amplitude sample location - within beta fractions
    # and slide waveform to that location
    newbeta = np.array([(beta[0] + 0.5*safe - 0.5),(beta[1] + 0.5*safe - 0.5)])/safe
    low_idx = int(T_obs*fs*newbeta[0])
    high_idx = int(T_obs*fs*newbeta[1])

    return low_idx,high_idx

def gen_noise(fs,T_obs,psd):
    """ Generates noise from a psd

    Parameters
    ----------
    fs:
        sampling frequency
    T_obs:
        observation time window
    psd:
        noise power spectral density

    Returns
    -------
    x:
        noise time series
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
    """ generates noise for a variety of different detectors

    Parameters
    ----------
    fs:
        sampling frequency
    T_obs:
        observation time window
    op:
        type of noise curve
    det:
        detector

    Returns
    -------
    psd:
        noise power spectral density
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

def whiten_data(data,duration,sample_rate,psd,flag='td'):
    """ Takes an input timeseries and whitens it according to a psd

    Parameters
    ----------
    data:
        data to be whitened
    duration:
        length of time series in seconds
    sample_rate:
        sampling frequency of time series
    psd:
        power spectral density to be used
    flag:
        if 'td': then do in time domain. if not: then do in frequency domain

    Returns
    -------
    xf:
        whitened signal 
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
    """ function returns a pair of masses drawn from the appropriate distribution
   
    Parameters
    ----------
    m_min:
        minimum component mass
    M_max:
        maximum total mass
    mdist:
        mass distribution to use when generating templates

    Returns
    -------
    m12: list
        both component mass parameters
    eta:
        eta parameter
    mc:
        chirp mass parameter
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

    # this distribution will constrain the chirp mass and inverse mass ratio allowed values
    # you can choose those values in the flag statement below
    elif mdist=='hunt_constrain':
        if verb: print '{}: using astrophysical logarithmic mass distribution'.format(time.asctime())
        new_m_min = m_min
        new_M_max = M_max
        log_m_max = np.log(new_M_max - new_m_min)
        while not flag:
            m12 = np.exp(np.log(new_m_min) + np.random.uniform(0,1,2)*(log_m_max-np.log(new_m_min)))
            eta = m12[0]*m12[1]/(m12[0]+m12[1])**2
            mc = np.sum(m12)*eta**(3.0/5.0)
            flag = True if (np.sum(m12)<new_M_max) and (np.all(m12>new_m_min)) and (m12[0]>=m12[1]) and (m12[1]/m12[0] >= 0.5) and (mc >= 20.0) and (mc <= 35.0) else False
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

def gen_par(fs,T_obs,mdist='astro',beta=[0.75,0.95],gw_tmp=False):
    """ Generates a random set of parameters
    
    Parameters
    ----------
    fs:
        sampling frequency (Hz)
    T_obs:
        observation time window (seconds)
    mdist:
        distribution of masses to use
    beta:
        fractional allowed window to place time series
    gw_tmp:
        if True: generate an event-like template

    Returns
    -------
    par: class object
        class containing parameters of waveform
    """
    # define distribution params
    m_min = 5.0         # 5 rest frame component masses
    M_max = 100.0       # 100 rest frame total mass
    log_m_max = np.log(M_max - m_min)

    m12, mc, eta = gen_masses(m_min,M_max,mdist=mdist)
    M = np.sum(m12)
    if verb: print '{}: selected bbh masses = {},{} (chirp mass = {})'.format(time.asctime(),m12[0],m12[1],mc)

    # generate iota
    iota = np.arccos(-1.0 + 2.0*np.random.rand())
    if verb: print '{}: selected bbh cos(inclination) = {}'.format(time.asctime(),np.cos(iota))

    # generate polarisation angle
    psi = 2.0*np.pi*np.random.rand()
    if verb: print '{}: selected bbh polarisation = {}'.format(time.asctime(),psi)

    # generate reference phase
    phi = 2.0*np.pi*np.random.rand()
    if verb: print '{}: selected bbh reference phase = {}'.format(time.asctime(),phi)

    # pick sky position - uniform on the 2-sphere
    ra = 2.0*np.pi*np.random.rand()
    dec = np.arcsin(-1.0 + 2.0*np.random.rand())
    if verb: print '{}: selected bbh sky position = {},{}'.format(time.asctime(),ra,dec)

    # pick new random max amplitude sample location - within beta fractions
    # and slide waveform to that location
    if gw_tmp: beta = [0.5,0.5]
    low_idx,high_idx = convert_beta(beta,fs,T_obs)
    if low_idx==high_idx:
        idx = low_idx
    else:
        idx = int(np.random.randint(low_idx,high_idx,1)[0])
    if verb: print '{}: selected bbh peak amplitude time = {}'.format(time.asctime(),idx/fs)

    # the start index of the central region
    sidx = int(0.5*fs*T_obs*(safe-1.0)/safe)

    # currently fixing all parameters other than mass, and time
    ra = 2.21535724066
    dec = -1.23649695537
    iota = 2.5#-0.8011436155469337
    phi = 1.5
    psi = 1.75

    # store params
    par = bbhparams(mc,M,eta,m12[0],m12[1],ra,dec,iota,phi,psi,idx,None,None)

    """
    Only if you want to gen template like gw150914. Will be first template in 
    set.
    """
    if gw_tmp:
        m1, m2 = 36.0, 29.0
        eta = m1*m2/(m1+m2)**2
        M = m1 + m2
        mc = M*eta**(3.0/5.0)

        # fixing all parameters other than mass and time
        ra=2.21535724066
        dec=-1.23649695537
        iota=2.5
        phi=1.5
        psi=1.75
        par = bbhparams(mc,M,eta,m1,m2,ra,dec,iota,phi,psi,idx,None,None)

    return par

def gen_bbh(fs,T_obs,psds,dets=['H1'],beta=[0.75,0.95],par=None,gw_tmp=False):
    """ generates a BBH timedomain signal

    Parameters
    ----------
    fs:
        sampling frequency
    T_obs:
        observation time window
    psds:
        power spectral desnity to use
    dets:
        detector
    beta:
        fractional range of time series to place peak signals
    par:
        class containing parameters of waveform to generate
    gw_tmp:
        if True: make a template exactly like event to do PE on
    
    Returns
    -------
    ts:
        h-plus and h-cross combined GW time series waveform
    hp:
        h-plus GW time series waveform
    hc:
        h-cross GW time series waveform
    ts:
        this is redundant. need to remove    
    """
    N = T_obs * fs                                   # the total number of time samples
    dt = 1 / fs                                      # time resolution (sec)
    f_low = 40                                       # lowest frequency of waveform (Hz)
    amplitude_order = 0                              # amplitude order 
    phase_order = 7                                  # phase order
    f_max = fs/2                                     # maximum allowed frequency for FD waveforms
    approximant = lalsimulation.IMRPhenomPv2         # waveform approximant
    dist = 410e6*lal.PC_SI                           # waveform distance

    # if making event-like template, then fix distance
    if gw_tmp:
        dist = 410e6*lal.PC_SI

    # make waveform
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

    whiten_hp = whiten_data(hp.data.data,T_obs,fs,psds,flag='fd')
    whiten_hc = whiten_data(hc.data.data,T_obs,fs,psds,flag='fd')

    orig_hp = np.roll(np.fft.irfft(whiten_hp,T_obs*fs),int(-fs))
    orig_hc = np.roll(np.fft.irfft(whiten_hc,T_obs*fs),int(-fs))


    # transform back into time domain in upcoming sections

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
    	ht_shift, hp_shift, hc_shift = make_bbh(orig_hp,orig_hc,fs,par.ra,par.dec,par.psi,det)

    	# place signal into timeseries - including shift
    	ht_temp = ht_shift[int(ref_idx-par.idx-11):] # use 21 if sampling at 2kHz 
    	hp_temp = hp_shift[int(ref_idx-par.idx-11):] # use 21 if sampling at 2kHz
    	hc_temp = hc_shift[int(ref_idx-par.idx-11):] # use 21 if sampling at 2kHz

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

    return ts, hp, hc, ts

def make_bbh(hp,hc,fs,ra,dec,psi,det):
    """ Turns hplus and hcross into a detector output
    applies antenna response and
    and applies correct time delays to each detector

    Parameters
    ----------
    hp:
        h-plus version of GW waveform
    hc:
        h-cross version of GW waveform
    fs:
        sampling frequency
    ra:
        right ascension
    dec:
        declination
    psi:
        polarization angle        
    det:
        detector

    Returns
    -------
    ht:
        combined h-plus and h-cross version of waveform
    hp:
        h-plus version of GW waveform 
    hc:
        h-cross version of GW waveform
    """
    # make basic time vector
    tvec = np.arange(len(hp))/float(fs)

    # compute antenna response and apply
    Fp,Fc,_,_ = antenna.response(float(event_time), ra, dec, 0, psi, 'radians', det )
    ht = hp*Fp + hc*Fc     # overwrite the timeseries vector to reuse it

    # compute time delays relative to Earth centre
    frDetector =  lalsimulation.DetectorPrefixToLALDetector(det)
    tdelay = lal.TimeDelayFromEarthCenter(frDetector.location,ra,dec,float(event_time))
    if verb: print '{}: computed {} Earth centre time delay = {}'.format(time.asctime(),det,tdelay)

    # interpolate to get time shifted signal
    ht_tck = interpolate.splrep(tvec, ht, s=0)
    hp_tck = interpolate.splrep(tvec, hp, s=0)
    hc_tck = interpolate.splrep(tvec, hc, s=0)
    if gw_tmp: tnew = tvec - tdelay
    else: tnew = tvec - tdelay# + (np.random.uniform(low=-0.037370920181274414,high=0.0055866241455078125))
    new_ht = interpolate.splev(tnew, ht_tck, der=0,ext=1)
    new_hp = interpolate.splev(tnew, hp_tck, der=0,ext=1)
    new_hc = interpolate.splev(tnew, hc_tck, der=0,ext=1)

    return ht, hp, hc

def sim_data(fs,T_obs,psds,dets=['H1'],Nnoise=25,size=1000,mdist='astro',beta=[0.75,0.95]):
    """ Simulates all of the test, validation and training data timeseries

    Parameters
    ----------
    fs:
        sampling frequency (Hz)
    T_obs:
        observation time window (seconds)
    psds:
        power spectral density to be used (default aLIGO PSD)
    dets:
        detectors
    Nnoise:
        number of noise realizations per GW template
    size:
        number of templates to generate
    mdist:
        mass distribution of templates
    beta:
        fractional range of observation time window to place peak of template in

    Returns
    -------
    [ts, yval]:
        ts is an array containing all the template waveform time sereis. 
        yval is redundant code left over from a previous version of this code. It currently
        just contains an array full of 1s
    temp:
        array containing the parameter values to be estimated for each GW template 
        timeseries.
    """

    yval = []                         # initialise the param output
    ts = []                           # initialise the timeseries output
    par = []                          # initialise the parameter output
    nclass = 1	                      # the hardcoded number of classes
    npclass = int(size/float(nclass)) # number of noise signals to make     
    ndet = len(dets)                  # the number of detectors

    # for the signal class - loop over random masses
    cnt = 0
    if gw_tmp:
        size = size -1
    while cnt<size:

        print '{}: making waveform {}/{}'.format(time.asctime(),cnt,size)
        # generate a single new timeseries and chirpmass
        par_new = gen_par(fs,T_obs,mdist=mdist,beta=beta,gw_tmp=False)
        _,_,_,ts_new = gen_bbh(fs,T_obs,psds,dets=dets,beta=beta,par=par_new)

	# loop over noise realisations
        # not typically used. Can be removed later
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
            par.append(par_new)
            yval.append(1)
            cnt += 1
            if verb: print '{}: completed {}/{} signal samples'.format(time.asctime(),cnt-npclass,int(size/2))

            # add extra time variations for each sample
            if do_time_grid:
                # pick new random max amplitude sample location - within beta fractions
                # and slide waveform to that location
                for m in range(N_time_grid):
                    low_idx,high_idx = convert_beta(beta,fs,T_obs)
                    if low_idx==high_idx:
                        idx = low_idx
                    else:
                        idx = int(np.random.randint(low_idx,high_idx,1)[0])
                    par_new.idx=idx
                    _,_,_,ts_new = gen_bbh(fs,T_obs,psds,dets=dets,beta=beta,par=par_new)
                    ts.append(np.array([t[int(((T_obs/2)*fs)-fs/2):int(((T_obs/2)*fs)+fs/2)] for t in ts_new]).reshape(ndet,-1))
                    par.append(par_new)
                    yval.append(1)

    # trim the data down to desired length
    size = N_time_grid * sample_num 
    ts = np.array(ts)[:size]
    yval = np.array(yval)[:size]
    par = par[:size]
    size = len(ts)

    # return randomised the data
    idx = np.random.permutation(size)
    temp = [par[i] for i in idx]
    ts, yval = ts[idx], yval[idx]

    # add gw150914 template at end of set
    if gw_tmp:
        print '{}: making waveform {}/{}'.format(time.asctime(),size+1,size+1)
        # generate a single new timeseries and chirpmass
        par_new = gen_par(fs,T_obs,mdist=mdist,beta=beta,gw_tmp=gw_tmp)
        _,_,_,ts_new = gen_bbh(fs,T_obs,psds,dets=dets,beta=beta,par=par_new,gw_tmp=gw_tmp)

        # just generate noise free signal
        ts = np.concatenate((ts,np.array([t[int(((T_obs/2)*fs)-fs/2):int(((T_obs/2)*fs)+fs/2)] for t in ts_new]).reshape(ndet,-1).reshape(1,1,fs)))
        temp.append(par_new)
        yval = np.append(yval,1)
    return [ts, yval], temp

# the main part of the code
def main():
    """ The main code - generates the training samples """

    # get the command line args
    args = parser()
    if args.seed>0:
        np.random.seed(args.seed)
    safeTobs = safe*args.Tobs

    # load event in noise frequency series, noise alone frequency sereis, and unwhitened psd
    unwht_wvf_file = np.loadtxt('%s/lalinferencenest-0-H1-%s.0-0.hdf5H1-freqData.dat' % (lalinf_out_loc,event_time))[:,1:]
    sig_unwht_wvf_file = np.loadtxt('%s/lalinferencenest-0-H1-%s.0-0.hdf5H1-freqDataWithInjection.dat' % (lalinf_out_loc,event_time))[:,1:]
    unwht_wvf_file = np.add(unwht_wvf_file[:,0],1j*unwht_wvf_file[:,1])
    sig_unwht_wvf_file = np.add(sig_unwht_wvf_file[:,0],1j*sig_unwht_wvf_file[:,1])
    plt.plot(unwht_wvf_file)
    plt.savefig('%sinput_waveform_unwht.png' % outdir)
    plt.close()

    # set all NaN values in frequency series to zero
    sig_unwht_wvf_file[np.isnan(sig_unwht_wvf_file) == True] = 0+0*1j
    unwht_wvf_file[np.isnan(unwht_wvf_file) == True] = 0+0*1j

    # get lalinf event noise-free template
    h_t = sig_unwht_wvf_file - unwht_wvf_file
    wvf_psd_file = np.loadtxt('%s/lalinferencenest-0-H1-%s.0-0.hdf5H1-PSD.dat' % (lalinf_out_loc,event_time))

    # redifine name of variable denoting lalinf signal burried in noise
    unwht_wvf_file = sig_unwht_wvf_file

    # whiten event in noise and noise-free signal from lalinf
    # wht_wvf is signal in noise and h_t is noise-free signal
    wht_wvf = whiten_data(unwht_wvf_file,safeTobs,args.fsample,wvf_psd_file[:,1],'fd')
    wht_wvf = np.fft.irfft(wht_wvf,args.fsample*safeTobs)
    h_t = whiten_data(h_t,safeTobs,args.fsample,wvf_psd_file[:,1],'fd')
    h_t = np.fft.irfft(h_t,args.fsample*safeTobs)

    # normalization constant applied to all waveforms used in training
    # this is applied in order to ensure that the standard deviation of the 
    # noise is equal to 1. An assumption which is made by the GAN
    gw_norm_constant = 1.0/np.std(wht_wvf)
    print(np.var(wht_wvf),1.0/np.var(wht_wvf))
    print(np.std(wht_wvf),1.0/np.std(wht_wvf))

    # set psd to the same used in lalinference for gw150914
    psd = wvf_psd_file[:,1]

    # extract central 1s of lalinference waveform for both waveform burried in noise and noise-free waveform
    wht_wvf = wht_wvf[int(((safeTobs/2)*args.fsample)-args.fsample/2.0):int(((safeTobs/2)*args.fsample)+args.fsample/2.0)]
    h_t = h_t[int(((safeTobs/2)*args.fsample)-args.fsample/2.0):int(((safeTobs/2)*args.fsample)+args.fsample/2.0)]
    plt.plot(wht_wvf)
    plt.plot(h_t)
    plt.savefig('%sinput_waveform.png' % outdir)
    plt.close()

    # break up the generation into blocks of args.Nblock training samples
    nblock = int(np.ceil(float(args.Nsamp)/float(args.Nblock)))
    for i in xrange(nblock):
        
    	# simulate the dataset and randomise it
    	print '{}: starting to generate data'.format(time.asctime())

        # apply oversampling of a portion of the mass parameter space if True.
        if oversamp == True:
            ts, par = sim_data(args.fsample,safeTobs,psd,args.Nnoise,size=args.Nblock,mdist='hunt_constrain',beta=[0.45,0.55])
        else:
    	    ts, par = sim_data(args.fsample,safeTobs,psd,args.Nnoise,size=args.Nblock,mdist=args.mdist,beta=[0.45,0.55])
        

        # apply normalization constant to all GW template waveforms
        # same normalization which has been applied to GW event to do PE on
        for n in range(ts[0].shape[0]):
            ts[0][n,0,:] *= gw_norm_constant
            #plt.plot(ts[0][n,0,:], alpha=0.5, color='green')

        if i != (nblock-1): 
            ts[0] = ts[0][:-1]
            par = par[:-1]

        # plot some GW template waveforms superimposed on lalinf noise-free waveform
        plt.plot((h_t * gw_norm_constant), alpha=0.5, linewidth=0.5)#/np.max(h_t)) 
        plt.plot(ts[0][-1,0,:], alpha=0.5, color='green', linewidth=0.5)
        plt.xlim([412,612]) # zoom-in on waveform
        plt.savefig('%swhitened_geneated_template.png' % outdir, dpi=700)
        plt.close()

        # store GW template parameters in an array
        signal_train_pars = []
        for k in par:
            signal_train_pars.append([k.mc,(k.m2/k.m1)])
        signal_train_pars = np.array(signal_train_pars)

        # plot GW template parameters in a scatter plot to see distribution
        plt.scatter(signal_train_pars[:,0],signal_train_pars[:,1],s=0.5)
        plt.savefig('%swhitened_geneated_template.png' % outdir, dpi=700)
        plt.close()
    	print '{}: completed generating data {}/{}'.format(time.asctime(),i+1,nblock)

    	# pickle the results
    	# save the GW template timeseries data to file
    	f = open(args.basename + event_name + '_ts_' + str(i) + '_' + str(sample_num) + 'Samp' + tag + '.sav', 'wb')
    	cPickle.dump(ts, f, protocol=cPickle.HIGHEST_PROTOCOL)
    	f.close()
    	print '{}: saved timeseries data to file'.format(time.asctime())

    	# save the GW template parameters to file
    	f = open(args.basename + event_name + '_params_' + str(i) + '_' + str(sample_num) + 'Samp' + tag +'.sav', 'wb')
    	cPickle.dump(par, f, protocol=cPickle.HIGHEST_PROTOCOL)
    	f.close()
    	print '{}: saved parameter data to file'.format(time.asctime())
        
        # save GW event to do PE on whitened noisy waveform
        f = open('data/' + event_name + str(i) + tag + '.sav', 'wb')
        cPickle.dump(wht_wvf, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '{}: saved {} timeseries data to file'.format(time.asctime(),event_name)

        # save GW event to do PE on whitened noise-free template
        f = open('data/' + event_name + '_data' + tag + '.pkl', 'wb')
        cPickle.dump(h_t, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '{}: saved {} timeseries data to file'.format(time.asctime(),event_name)

    print '{}: success'.format(time.asctime())

if __name__ == "__main__":
    exit(main())
