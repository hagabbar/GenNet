"""
script to make whitened bbh tempaltes
"""
from __future__ import division
import h5py
import pickle
from gwpy.table import EventTable
import numpy as np
from scipy import integrate, interpolate
import random

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import lal
import lalsimulation
from pylal import antenna, cosmography
import argparse
import time
from scipy.signal import filtfilt, butter
from scipy.stats import norm, chi
from scipy.optimize import brentq
import os

import sys

#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import norm


from sys import exit

safe = 2    # define the safe multiplication scale for the desired time length

def gen_psd(fs, T_obs, op='AdvDesign', det='H1'):
    """
    generates noise for a variety of different detectors
    """
    N = T_obs * fs  # the total number of time samples
    dt = 1 / fs  # the sampling time (sec)
    df = 1 / T_obs  # the frequency resolution
    psd = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0, df, lal.HertzUnit, N // 2 + 1)

    if det == 'H1' or det == 'L1':
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

def chris_whiten_data(data,duration,sample_rate,psd,flag='td'):
    """
    Takes an input timeseries and whitens it according to a psd
    """

    if flag=='td':
        # FT the input timeseries - window first
        win = tukey((duration)*int(sample_rate),alpha=1.0/8.0)
        xf = np.fft.rfft(win*data)
    else:
        xf = data


    # deal with undefined PDS bins and normalise
    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300
    xf /= (np.sqrt(0.5*psd*sample_rate))

    # Detrend the data: no DC component.
    xf[0] = 0.0

    x = np.fft.irfft(xf)
    win = tukey(duration*sample_rate,alpha=1.0/8.0)
    x *= win
    return x

    #if flag=='td':
        # Return to time domain.
    #    x = np.fft.irfft(xf)
    #    return x
    #else
    #    return xf

def get_fmin(M,eta,dt):
    """
    Compute the instantaneous frequency given a time till merger
    """
    M_SI = M*lal.MSUN_SI

    def dtchirp(f):
        """
        The chirp time to 2nd PN order
        """
        v = ((lal.G_SI/lal.C_SI**3)*M_SI*np.pi*f)**(1.0/3.0)
        temp = (v**(-8.0) + ((743.0/252.0) + 11.0*eta/3.0)*v**(-6.0) -
                (32*np.pi/5.0)*v**(-5.0) + ((3058673.0/508032.0) + 5429*eta/504.0 +
                (617.0/72.0)*eta**2)*v**(-4.0))
        return (5.0/(256.0*eta))*(lal.G_SI/lal.C_SI**3)*M_SI*temp - dt

    # solve for the frequency between limits
    fmin = brentq(dtchirp, 1.0, 2000.0, xtol=1e-6)
    print '{}: signal enters segment at {} Hz'.format(time.asctime(),fmin)

    return fmin

def make_waveforms(template,dt,dist,fs,approximant,N,ndet,dets,psds,T_obs,f_low=12.0):
    """ make waveform"""


    # define variables
    template = list(template)
    m12 = [template[0],template[1]]
    eta = template[2]
    mc = template[3]
    N = T_obs * fs      # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)
    approximant = lalsimulation.IMRPhenomD
    f_high = fs/2.0          #/2.0 why was f_high divided by 2?????
    df = 1.0/T_obs
    f_low = df*int(get_fmin(mc,eta,1.0)/df)
    f_ref = f_low    
    dist = 1e6*lal.PC_SI  # put it as 1 MPc

    # generate iota
    iota = np.arccos(-1.0 + 2.0*np.random.rand())
    print '{}: selected bbh cos(inclination) = {}'.format(time.asctime(),np.cos(iota))

    # generate polarisation angle
    psi = 2.0*np.pi*np.random.rand()
    print '{}: selected bbh polarisation = {}'.format(time.asctime(),psi)

    # print parameters
    print '{}: selected bbh mass 1 = {}'.format(time.asctime(),m12[0])
    print '{}: selected bbh mass 2 = {}'.format(time.asctime(),m12[1])
    print '{}: selected bbh eta = {}'.format(time.asctime(),eta)

    # make waveform
    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
                    m12[0] * lal.MSUN_SI, m12[1] * lal.MSUN_SI,
                    0, 0, 0, 0, 0, 0,
                    dist,
                    iota, 
                    0, 0, 0, 0,
                    df,
                    f_low,f_high,
                    f_ref,
                    lal.CreateDict(),
                    approximant)



    hp = hp.data.data
    hc = hc.data.data

    # compute reference idx
    ref_idx = np.argmax(hp**2 + hc**2)

    # get peak time varrying range
    # pick new random max amplitude sample location - within beta fractions
    # and slide waveform to that location
    beta=[0.75,0.95]
    low_idx,high_idx = convert_beta(beta,fs,T_obs)
    if low_idx==high_idx:
        idx = low_idx
    else:
        idx = int(np.random.randint(low_idx,high_idx,1)[0])

    j = 0
    hp_temp = hp[int(ref_idx-idx):]
    hc_temp = hc[int(ref_idx-idx):]
    #if len(ht_temp)<N:
    #    hp[j,:len(ht_temp)] = hp_temp
    #    hc[j,:len(ht_temp)] = hc_temp
    #else:
    #    hp[j,:] = hp_temp[:N]
    #    hc[j,:] = hc_temp[:N]

    for psd in psds:
        hp_1_wht = chris_whiten_data(hp_temp, T_obs, fs, psd.data.data, flag='fd')
        hc_1_wht = chris_whiten_data(hc_temp, T_obs, fs, psd.data.data, flag='fd')


    return hp_1_wht,hc_1_wht,get_fmin(mc,eta,1)

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

def main():
    Tobs = 1       # observation time window
    fs = 1024      # sampling frequency
    ndet = 1       # number of detectors
    N = Tobs * fs  # the total number of time samples
    n = N // 2 + 1 # the number of frequency bins    
    f_low = 20.0     # the lower frequency cutoff
    dt = 1 / fs    # the sampling time (sec)
    dets = ['H1']

    # get psds
    psds = [gen_psd(fs, Tobs, op='AdvDesign', det='H1')] #for d in args.detectors]
    wpsds = (2.0 / fs) * np.ones((ndet, n)) # define effective PSD for whited data

    # get template bank parameters
    tmp_bank = 'data/f_low20-threePointFivePN-f_upper2048NonSpin_15-50m1m2.xml'
    # load template bank
    # format=ligolw.sngl_inspiral
    tmp_bank = np.array(EventTable.read(tmp_bank,
    format='ligolw', tablename='sngl_inspiral', columns=['mass1','mass2','eta','mchirp']))

    approximant = lalsimulation.IMRPhenomD
    dist = 1e6 * lal.PC_SI

    outdir = 'templates/'

    # loop over template bank params
    for idx,w in enumerate(tmp_bank):
        if idx == 0:
            hp,hc,fmin = make_waveforms(w,dt,dist,fs,approximant,N,ndet,dets,psds,Tobs,f_low)
            hp_bank = {idx:hp}
            hc_bank = {idx:hc}
            fmin_bank = {idx:fmin}
        #if idx == 10:
        #    break
        else:
            hp_new,hc_new,fmin_new = make_waveforms(w,dt,dist,fs,approximant,N,ndet,dets,psds,Tobs,f_low)
            hp_bank.update({idx:hp_new})
            hc_bank.update({idx:hc_new})
            fmin_bank.update({idx:fmin_new})


    # dump contents of hp and hc banks to pickle file
    pickle_hp = open("%shp.pkl" % outdir,"wb")
    pickle.dump(hp_bank, pickle_hp)
    pickle_hp.close()
    pickle_hc = open("%shc.pkl" % outdir,"wb")
    pickle.dump(hc_bank, pickle_hc)
    pickle_hc.close()
    pickle_fmin = open("%sfmin.pkl" % outdir,"wb")
    pickle.dump(fmin_bank, pickle_fmin)
    pickle_fmin.close()

    hp = hp_bank
    hc = hc_bank

if __name__ == "__main__":
    main()
