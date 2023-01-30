#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Piet Wagner (teip12@web.de)
# Created Date: 24.01.2022
# Updated:      17.06 by Piet
# =============================================================================
"""The Module has been build as a helper fucntion for the PEX_Recon and is necessary for it run."""
# =============================================================================
# Imports
# =============================================================================
from __future__ import print_function  # for python 2.7 compatibility
import datetime
import os
import pickle
import re
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import mapvbvd
# =============================================================================

# Functions
# =============================================================================

def log(*msg):
    """
    fancier logging
    :param msg: message to be displayed
    :return: nothing
    """
    print(f"[{datetime.datetime.now()}]", end=" ")
    for m in msg:
        print(m, end=" ")
    print("", flush=True)
def pload(filepath):
    try:
        with open(filepath, "rb") as f:
            dat = pickle.load(f)
            if len(dat) == 1:
                return dat[0]
            else:
                return dat
    except:
        try:
            # print("Try Python2 compatibility mode...")
            with open(filepath, "rb") as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                dat = u.load()
                if len(dat) == 1:
                    return dat[0]
                else:
                    return dat
        except:
            raise ValueError(f"Error in opening {filepath}")


def pdump(filepath, *data):
    filepath = Path(str(filepath))  # so we can use superpath as well :D
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=-1)

def loadParam(spacedata, header,max_Voltage_steps ,startV=25, stopV=300 ,T1= 0.6,
              stepV=1,):
    RX = spacedata.shape[2]
    TX = spacedata.shape[4]

    print(TX,"TX",RX,"RX")
    nV = int(header.search_header_for_val("MeasYaps",("sWiPMemBlock","alFree","2"))[0]) # 0V is not in the  list

    drop=0
    if nV > max_Voltage_steps:
        drop=nV-max_Voltage_steps
        nV-=drop
    dT = header.search_header_for_val("MeasYaps",("sWiPMemBlock","alFree","7"))[0] * 1e-6
    beta = header.search_header_for_val("MeasYaps",("adFlipAngleDegree"))[0] * np.pi / 180

    TR = header.search_header_for_val("MeasYaps",("alTR","0"))[0] * 1e-6
    maxV = header.search_header_for_val("MeasYaps",("sWiPMemBlock","alFree","1"))[0]
    prep_pulse_duration =header.search_header_for_val("MeasYaps",("sWiPMemBlock","alFree","10"))[0]*1e-2
    print(f'Prep Pulse duration: {prep_pulse_duration}')
    minf = 0.05  # minimum function value for model-data
    a = np.exp(-dT / T1)
    b = 1 - np.exp(-(TR - dT) / T1)

    alphainit = np.arccos((a - 1) / (a * b)) / np.arange(startV, stopV,
                                                         stepV)  # start value array for nelder mead
    maxiter = len(alphainit)  # iterations for nelder mead
    return RX, TX, dT, beta, TR, maxV, minf, maxiter, alphainit,nV,drop,prep_pulse_duration


def loadsingleTX(spacedata, TX):  #
    TXdata = spacedata[:, :, :, :, TX]
    return TXdata


def getmaskchall(spacedata, maskthresh=0.01, ch1=None,
                 TX=8):


    # singel channels
    sumspacedata_ch = np.sum(np.abs(np.sqrt(np.sum(np.abs(spacedata), axis=2))), axis=2)


    relsumspacedata_ch = sumspacedata_ch / np.max(sumspacedata_ch,axis=(0,1))
    maskallch_ch = np.copy(relsumspacedata_ch)
    maskallch_ch[maskallch_ch < maskthresh*2.98] = 0
    maskallch_ch[maskallch_ch >= maskthresh*2.98] = 1

    return maskallch_ch

def getspacedata(FilePath,max_Voltage_steps):
    '''
    Loads the spacedata out of the nifti files and transforms it into the right format for the FFT to get the spacedata

    :param FilePath:
    :return:
    '''
    #Open .dat files for the header information
    filepath = FilePath+r".dat"
    filepath.replace('\\', '/')
    #twixprot is a libery to load daten from siemens scanner
    twixObj = mapvbvd.mapVBVD(str(filepath))

    drop =0
    # nV voltage steps
    assert twixObj.search_header_for_val("MeasYaps",("sWiPMemBlock","alFree","2"))[0],AssertionError("check if you have the right PEX file(Missing voltage "
                                                              "steps in the Header)")

    nv = int(twixObj.search_header_for_val("MeasYaps",("sWiPMemBlock","alFree","2"))[0])
    if max_Voltage_steps < nv:
        drop=1
    #Load Imag data from nifti images!!! not needed anymore !!!
    #filepathnii = FilePath + r'.nii'
    #imagnii = nib.load(str(filepathnii))

    twixObj.image.flagRemoveOS =True # Remove Oversampling turned off
    kdata = np.moveaxis( np.squeeze(twixObj.image[""]),2, 1) #removes empty channels and Changes format to (H,W,RX,TX*n Voltage steps)


    # FFT shift because scanner 0 is in the middel and python starts in top left, more infos on Slide 11 in Powerpoint
    kspacedata_shiftnii = np.fft.fftshift( kdata, axes=(0, 1))  # shape=(H,W, RX,TX* n Voltage steps)
    # FFT over H and W parameter
    spacedata_resort_shiftnii = np.fft.fftn(kspacedata_shiftnii,axes=(0,1))
    # FFT shifted back because scanner 0 is in the middle and python starts in top left
    spacedata_resort_shift2nii = np.fft.fftshift(spacedata_resort_shiftnii, axes=(0, 1))
    # shape=shape=(H/2,W,RX), 1)
    spacedata = spacedata_resort_shift2nii.reshape(
        [np.int( kdata.shape[0] ),  kdata.shape[1],  kdata.shape[2], nv,
         int( kdata.shape[3] / nv)])[:, :, :, 1:nv - drop, :]  ##


    print('calculated spacedata and loaded Header')
    return kdata, spacedata, twixObj

def maskch1(spacedata, maskthresh,TX_channels):
        sumspacedatach = np.sum(np.abs(np.sqrt(np.abs(spacedata) ** 2))[:, :, 7, :], axis=2) #7 = master channel

        relsumspacedatach = sumspacedatach / np.max(sumspacedatach)
        relsumspacedatach[relsumspacedatach < maskthresh] = 0
        relsumspacedatach[relsumspacedatach >= maskthresh] = 1
        maskch = np.tile(relsumspacedatach, (1, 1, 8))
        print('calculated mask')
        return maskch
def Voltage_loss(U1, db):
    U2 = U1 * 10 ** (db / 20)
    return U2


