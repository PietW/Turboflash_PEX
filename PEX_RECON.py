#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Piet Wagner (teip12@web.de)
# Created Date: 24.01.2022
# Updated:       17.06 by Piet
# =============================================================================
"""The Module has been build as a Main function for the PEX Reconstruction.
You can modifie the parameters in the Head of PEX_RECON or start it as a Terminal command """
# =============================================================================
# Imports
# =============================================================================
import argparse
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path
import mapvbvd
import matplotlib.pyplot as plt

import PEX_helper as helper
import PEXplot as Plot
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm import tqdm
import errno
import os


# =============================================================================
# Hint you can use this function in the terminal
# python PEX_RECON.py --pathfolder "Z:\wagner12\data\PEX-TEST\data8ch" --jobs 8 --loss_in_db 0 --mask 0.04 --stopV 300 --startV 20 --stepV 2 --T1 0.6 --min_V 10 --max_V  800 --max_Voltage_steps 17
# =============================================================================
# Functions
# =============================================================================
def PEX_RECON(pathfolder,n_jobs,loss_in_db,mask_thresh,stopV ,startV,stepV,T1,Voltage_limits,max_Voltage_steps):
    '''
    main() of PEX Reconstruction
    needed Parameters
    pathfolder:      Path(r"Z:\wagner12\data\20220126_PEXText\RAW")
    n_jobs:          number of Jobs use for parallelization
    loss_in_db:      Loss from cables and Switchbox in dB, use if you want to compare with Simulation
    mask_thresh:     Threshold for the mask depends on the signal intensity
    T1:(0.6s)        T1 time of the medium

    Optional_parameters

    stopV:( 300V)    Parameter for the Fitting function as initial guesses
    startV:(20V)     Parameter for the Fitting function as initial guesses
    stepV:(2V)       Parameter for the Fitting function as initial guesses
    Combine_TXchannels:      Combines the TX Channels to one Channel
    Voltage_limits:([10, 800])  upper and lower limits for the fit function
    Return:         Safes a picked File of the results in Pathfolder location
                    List(kspacedatanii, spacedata, header, mask_ch, B1parray, U90array,Im_Ch_array, AlphaU_full_array)
                    KspaceData,Spacedata,header with information,Mask of the B1 Images,B1+ Image ,U90 Voltages,Normalized Spacedata,All Voltages
    '''
    #Needed_Parameters


    if not pathfolder:
        pathfolder = Path(r"/home/teip/Desktop/0_masterarbeit/data/PEX_7Tesla_Buch/PEX_Jan")
    if not n_jobs:
        n_jobs=14  # for debugging use 1 job
    if not loss_in_db:
        loss_in_db =0 # loss has to be negative
    if not mask_thresh:
        mask_thresh = 0.05   # <--------magical mask  threshold
    if not T1 :
        T1 = 1 #  T1 Time of the Phantom


    #Optional_parameter
    if not stopV :
        stopV = 350  # Init parameters that are needed for the Fitfunction
    if not startV :
        startV = 20  #  Init parameters that are needed for the Fitfunction
    if not stepV:
        stepV = 4  # Init parameters that are needed for the Fitfunction

    if not Voltage_limits[0] :
        Voltage_limits = [10, 800]  # limits for the fit function if the values make sense
    if not max_Voltage_steps:
        max_Voltage_steps=17 # the PEX sequenc has a limitation that only 16 Voltage steps will be sequenced every thing after the 16 step will be 0, therfore do max 17 steps and the 17 step is 0 V

    # =============================================================================
    #Find all .nii files in the PathFolder
    if type(pathfolder)==list:
        pathfolder=pathfolder[0]
    files_list = [f for f in listdir(pathfolder) if
                  isfile(join(pathfolder, f)) and splitext(join(pathfolder, f))[-1] == ".dat"]
    if pathfolder.is_file() :
        files_list =[pathfolder[0]]
    print("List of .dat Files:",files_list)

    # =============================================================================
    #Loop over every file in the list and reconstruct the Files
    # !!!!warning Files that have been already Reconstructed will be overwritten !!!


    for file in files_list:
        #Load Data and Header
        print(f"Start Proccesing of File {file},  {files_list.index(file)} of {len(files_list)}")

        kspacedatanii, spacedata, header = helper.getspacedata(join(pathfolder, file[:-4]),max_Voltage_steps)

        TX = spacedata.shape[4]
        RX = spacedata.shape[2]

        #header_dict.update()
        # make masks per channel
        mask_ch= helper.getmaskchall(spacedata, mask_thresh, TX=TX)#updatet to multi channelsensitiv maps

        B1parray, U90array, Im_Ch_array, AlphaU_full_array, Model_array = EvalTurboFlashAll(spacedata, header, mask_ch,
                                                                               loss_in_db=loss_in_db, startV=startV,
                                                                               stopV=stopV, stepV=stepV,
                                                                               Voltage_limits=Voltage_limits,
                                                                               n_jobs=n_jobs,T1=T1,max_Voltage_steps=max_Voltage_steps)
        # ==============================================================================================================
        #saves Data in pickled File for ploting

        savepath=join(pathfolder,str(splitext(file)[0])+str("_Results"))
        try:
            os.mkdir(join(pathfolder,str(splitext(file)[0])+str("_Results")))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        Count = -1
        savepath_count=Path(join(savepath,str("File.pickled")))

        while savepath_count.is_file():
            Count += 1
            savepath_count=Path(join(savepath,str(Count)+str("_File.pickled")))


        helper.pdump(join(savepath_count)  , kspacedatanii, spacedata, header.hdr, mask_ch, B1parray, U90array,
                  Im_Ch_array, AlphaU_full_array, Model_array)  # <-------- Save as pickedfile
        # ==============================================================================================================
        #Plot B1+ map to have a look how the data looks
        Plot.Plot_one(Filepath=savepath_count)
        print("Finished with PEX Recon, Data is in:",savepath_count)
    return  kspacedatanii, spacedata, header, mask_ch, B1parray, U90array,Im_Ch_array, AlphaU_full_array, Model_array




def EvalTurboFlashAll(spacedata, header, maskch,loss_in_db,startV,stopV,stepV, Voltage_limits,n_jobs,T1,max_Voltage_steps):
    '''
    Evaluation of the PEX/Turboflash sequenze

    :return: kspacedatanii, spacedata, header, mask_ch, B1parray, U90array,Im_Ch_array, AlphaU_full_array,t1_map
    '''
    #Load more Parameters from the Header

    RX, TX, dT, beta, TR, maxV, minf, maxiter, alphainit,nV,drop,prep_pulse_duration= helper.loadParam(spacedata, header,max_Voltage_steps,startV,stopV,stepV,T1)

    # create empty arrays to store data

    B1p_array = np.zeros((TX, spacedata.shape[0], spacedata.shape[1]))

    U90_array = np.zeros((TX, spacedata.shape[0], spacedata.shape[1]))

    AlphaU_full_array = np.zeros((TX, spacedata.shape[0], spacedata.shape[1]))
    Model_array=np.zeros((TX, spacedata.shape[0], spacedata.shape[1], nV - 1))

    # TX channel wise calculations

    RXset = []
    for countcoil in range(TX):
        #makes a list of all the Nr. of coils
        TXcount = (countcoil + 7) % 8
        RXset.append(TXcount)


    for countcoil in tqdm(range(TX)):
        # Loop over all the coils

        helper.log("Calculate TX Coil countcoil: " + str(countcoil))

        # reduce data to data of 1 TX channel
        spacedata_ch = helper.loadsingleTX(spacedata, countcoil)

        # sum up rx channels
        im_1ch = np.sqrt(np.sum(np.abs(spacedata_ch[:, :, RXset, :]) ** 2, axis=2))[:, :, ::-1]

        # create normalization array with first voltage nV times to normalize im_1ch for first entries
        S0 = np.abs(np.tile(im_1ch[:, :, 0], (nV, 1, 1)).transpose(1, 2, 0))[:,:,1:]

        # normalize data
        im_1ch = (im_1ch / S0)
        try:
            Im_Ch_array
        except NameError:
            Im_Ch_array= [im_1ch[:, :, :]]
        else:


            Im_Ch_array=np.append(Im_Ch_array,[im_1ch],axis=0)

        # list of Voltages
        Voltage = list(header.search_header_for_val("MeasYaps",("sWiPMemBlock","adFree")).copy() )
        # Voltage 0 is not listed in header therfore it needs to be added
        Voltage.append(float(0))

        if not len(Voltage)==nV:
                print(Voltage,nV)
                Voltage=Voltage[:nV-1]
                Voltage.append(float(0))
                print(Voltage)
                print("old PEX sequence with wrong Voltage steps")

        # get voltagesteps for measured points################including loss from the scanner only nessesary if you want to compare with Simulation
        Vdata = helper.Voltage_loss(np.array(Voltage)[::-1][:-1],loss_in_db)




        # Parallel Pixel wise calculations
        ist=np.nonzero(maskch[:,:, countcoil])
        print("Start parallel pixelwise processing on ",n_jobs,"Kernels")
        Data=Parallel(n_jobs=n_jobs,verbose=5)(delayed(Paralle_funtion)(ii, jj, im_1ch, minf, maxiter, alphainit, Vdata, dT, T1, beta, TR, TX, Voltage_limits,prep_pulse_duration) for ii,jj in zip(ist[0],ist[1]))

        for x in Data:
            B1p, U90, alpha_U, ydata, ii, jj,Model = x
            AlphaU_full_array[countcoil, ii, jj] = alpha_U
            U90_array[countcoil, ii, jj] = U90
            B1p_array[countcoil, ii, jj] = B1p

            for f in range(len(Model)):
                Model_array[countcoil, ii, jj,f] = Model[f]

    return B1p_array, U90_array, Im_Ch_array, AlphaU_full_array, Model_array

def Paralle_funtion(ii,jj,im_1ch,minf, maxiter, alphainit, Vdata, dT, T1, beta,TR ,TX,Voltage_limits,prep_pulse_duration):
    # only take data points inside the mask
    ydata = im_1ch[ii, jj, :]  # get the data for the pixel

    # Neldermead fit
    fval_arr, alpha_Uarr, succes_arr,Model_arr_min = Nealder_mead_fit(T1,minf, maxiter, alphainit, ydata, Vdata, dT, beta, TR)

    # find best fit inbetween different start values
    index_min_fval = fval_arr.index(min(fval_arr))

    alpha_U = alpha_Uarr[index_min_fval]

    Model=Model_arr_min[index_min_fval]
    # calculate U90
    # analytical solution of Fitfunction(Vdata)==0 which returns U90(Voltage for 90° Flipangle

    if succes_arr[index_min_fval] == True and alpha_U != 0:
        a = np.exp(-dT / T1)
        b = 1 - np.exp(-(TR - dT) / T1)
        U90 = np.abs((np.arccos((a - 1) / (a * b))) / alpha_U)  #solved for u90 with singal =0
        if U90 < Voltage_limits[0]:  # threshold if u90 less than 10V

            U90 = Voltage_limits[0]
        elif U90 > Voltage_limits[1]:  # threshold if u90 more than 800V
            U90 = Voltage_limits[1]

        # calculate B1+ for more information see Meeting_31.01_HOW_THE_PEX_WORKS.pptx part 4. additional Information )np.pi / (2 * 267.52218744e6 * prep_pulse_duration)*1e6
        scalefactor_B1p = np.pi / (2 * 42.577 * 2 * np.pi * prep_pulse_duration* 10 ** 3) * 1e6  # B1+ for 90° flip angle in muT
        scalefactor_Power = np.sqrt(1000 * 50 / TX)  #
        B1p = scalefactor_B1p * scalefactor_Power * 1 / U90


    else:
        U90 = float("nan")
        B1p = 0


    return [B1p, U90, alpha_U, ydata, ii, jj,Model]



def Nealder_mead_fit(T1,minf,maxiter ,alphainit, ydata, Vdata, dT, beta,TR):
    '''
    minimize function devModelDataiijj with start value alpha0U, if minimum is still larger than minf,
    repeat it for next alpha0U entry of alphainit

    '''

    # define start values for Nelder mead fit
    fval = 1
    iter = 0

    # define arrays for results of Nelder Mead fit with different start values
    fvalarr = []
    alphaUarr = []
    Model_arr_min=[]
    succesarr = []

    while fval > minf:
        alpha0U = alphainit[iter]
        Model_arr = []
        def ModelData_function(alphaU):
            '''
            This Function is here because the minimizer function need a Function that has a model and a measure of the
            discrepancy between the data and an estimation mode
            '''

            return devModelData(alphaU=alphaU, ydata=ydata, Vdata=Vdata, dT=dT, T1=T1, beta=beta,
                                TR=TR, Model_arr=Model_arr)
        #minimizer function to fit the model on the data
        NMresult = minimize(ModelData_function, np.array(alpha0U), method='Nelder-Mead')

        # run Minimizer
        fval = NMresult.fun
        fvalarr.append(fval)
        Model_arr_min.append(Model_arr[NMresult.nit])

        #save results
        alphaUarr.append(NMresult.x)

        succesarr.append(NMresult.success)
        iter += 1
        if iter > maxiter - 1:
            break

    return fvalarr, alphaUarr, succesarr,Model_arr_min

def devModelData(alphaU,ydata,Vdata,dT,T1,beta,TR,Model_arr):
    #Model Fitting

    model= FitFunction(alphaU,Vdata,dT,T1,beta,TR)
    dev = np.sum(np.abs(ydata-model)) #changed from summed squared deviations to abs because erros >1 will fit bad
    Model_arr.append(model)
    return dev

def FitFunction(alphaU, Vdata,dT,T1,beta,TR):
    #Model Equation

    #parameters computation
    a = np.exp(-dT / T1)
    b = 1 - np.exp(-(TR - dT) / T1)
    c = np.exp(-TR / T1) * np.cos(beta)

    # Computation of Signal Equation
    ##Signal Equation S = np.abs((1 - np.exp(-dT / T1) * (1 - np.cos(alphaU * Vdata) * (1 - np.exp(-(TR - dT) / T1)))) / (1 - np.exp(-TR / T1) * np.cos(beta) * np.cos(alphaU * Vdata)))

    S00 = (1-c)/(1-a+a*b)
    return np.abs(S00*(1-a*(1-np.cos(alphaU * Vdata)*b))/(1-c*np.cos(alphaU * Vdata))) # fragen max alha u beta?



if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Set parameters for PEX_RECON')

    parser.add_argument('--pathfolder',default=[None], metavar='P', type=Path, nargs='+',
                        help='as')
    parser.add_argument('--jobs', default=[None], metavar='J', type=int, nargs='+',
                        help='as')
    parser.add_argument('--loss_in_db',default=[None], metavar='L', type=float, nargs='+',
                        help='loss_in_db')
    parser.add_argument('--mask',default=[None], metavar='M', type=float, nargs='+',
                        help='mask')

    parser.add_argument('--stopV',default=[None], metavar='V1', type=int, nargs='+',
                        help='stopV')
    parser.add_argument('--startV',default=[None], metavar='V2', type=int, nargs='+',
                        help='startV')
    parser.add_argument('--stepV',default=[None], metavar='V3', type=int, nargs='+',
                        help='stepV')
    parser.add_argument('--T1',default=[None], metavar='T1', type=float, nargs='+',
                        help='T1')
    parser.add_argument('--min_V',default=[None], metavar='V4', type=int, nargs='+',
                        help='min_V')
    parser.add_argument('--max_V',default=[None], metavar='V5', type=int, nargs='+',
                        help='max_V')
    parser.add_argument('--max_Voltage_steps', default=[None], metavar='V6', type=int, nargs='+',
                        help='max_Voltage_steps')


    args = parser.parse_args()




    PEX_RECON(args.pathfolder[0],args.jobs[0],args.loss_in_db[0],args.mask[0],args.stopV[0],args.startV[0],args.stepV[0],args.T1[0],[args.min_V[0],args.max_V[0]],args.max_Voltage_steps[0])
# use this in the Terminal to run the function
#python PEX_RECON.py --pathfolder "Z:\allgemein\projects\User\PEX\raw" --jobs 2 --loss_in_db 0 --mask 0.04 --stopV 300 --startV 20 --stepV 2 --T1 0.6 --min_V 10 --max_V 800 --max_Voltage_steps 17
