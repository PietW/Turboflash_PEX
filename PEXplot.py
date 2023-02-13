import matplotlib.pyplot as plt
from skimage import feature
import numpy as np
import PEX_helper as helper
import errno
import random as rng
from scipy import interpolate
import pathlib
from os.path import isfile, join, splitext
from os import listdir

def Plot_one(Filepath=None, TX=1, cmap=0):  # rang=[start,end]



    File = np.array(helper.pload(Filepath),
                    dtype=object)  # data.format=[kspacedatanii, spacedata, header, maskch, B1parray, U90array, Im_Ch_array, AlphaU_full_array,Modelfit per pixel]
    TX = File[4].shape[0]



    #############################################################################
    # plot B1_plus map

    B1plus = np.array(np.abs(File[4]))

    B1plus[B1plus>70]=0

    for tx in range(TX):
        if TX == 1:
            fig, axs = plt.subplots(ncols=1, nrows=1)
            fig1 = axs.imshow(B1plus[tx],vmax=30, cmap='inferno')
        else:
            fig, axs = plt.subplots(ncols=1, nrows=1)
            fig1 = axs.imshow(B1plus[tx],vmax=30, cmap='inferno')
            axs.set_title("Channel:" + str(tx))

        cbar = fig.colorbar(fig1)
        cbar.set_label(r'$\frac{μT}{\sqrt{kW}} $')

        plt.savefig(join(splitext(Filepath)[0]+str("B1_plus_map")+str(tx)))
        plt.show()
def Plot_one_slice(Filepath=None, TX=1, cmap=0):  # rang=[start,end]

    File = np.array(helper.pload(Filepath), dtype=object)
    TX=File[4].shape[0]
    fig, axs = plt.subplots(ncols=TX, nrows=1, constrained_layout=True)


    B1plus = np.array(File[4][:][:][:])# data,0,x,y kspacedatanii, spacedata, header, maskch, B1parray, U90array, Im_Ch_array, AlphaU_full_array)

    #B1plus[B1plus > abs(np.median(B1plus)-np.var(B1plus))] = 0
    for tx in range(TX):
        if TX==1:

            fig1 = axs.imshow(B1plus[tx], cmap='inferno')
        else:

            fig1 = axs[tx].imshow( B1plus[tx],  cmap='inferno')
            axs[tx].set_title("Channel:"+str(tx))


    cbar = fig.colorbar(fig1)
    cbar.set_label(r'$\frac{μT}{\sqrt{kW}} (x)$')

    plt.show()
    plt.close()
    plt.plot(B1plus[tx], label="PEX scan ")
    plt.show()




def Plot_debuging_Images(Filepath=None, TX=1, cmap=0):


    File = np.array(helper.pload(Filepath), dtype=object)# data.format=[kspacedatanii, spacedata, header, maskch, B1parray, U90array, Im_Ch_array, AlphaU_full_array,Modelfit per pixel]
    header = File[2]

    nv=[]


    for voltage_steps in range(int(header[f"MeasYaps"]["sWiPMemBlock","alFree","2"])):
        try:
            nv.append(header[f"MeasYaps"]["sWiPMemBlock","adFree",str(voltage_steps)])
        except KeyError:
            pass





    nv.append(0)
    print("Voltage steps ",nv)
#############################################################################
    #plot B1_plus map

    B1plus = np.array(np.abs(File[4]))[0]
    fig, axs = plt.subplots(ncols=1, nrows=TX, constrained_layout=True)
    for tx in range(TX):
        if TX == 1:

            fig1 = axs.imshow(B1plus, cmap='inferno')
        else:

            fig1 = axs[tx].imshow(B1plus[tx], cmap='inferno')
            axs[tx].set_title("Channel:" + str(tx))

    cbar = fig.colorbar(fig1)
    cbar.set_label(r'$\frac{μT}{\sqrt{kW}} $')
    plt.show()

    X=int(input("Please enter your x-point of interest: "))
    Y = int(input("Please enter your y-point of interest: "))
####################################################################
    #plot Image mask where the Image Recon is done
    Image_mask = np.array(np.abs(File[3]))
    fig, axs = plt.subplots(ncols=1, nrows=TX, constrained_layout=True)
    for tx in range(TX):
        if TX == 1:

            fig1 = axs.imshow(Image_mask[:,:,0] , cmap='inferno')
        else:

            fig1 = axs[tx].imshow(Image_mask[:,:,TX] , cmap='inferno')
            axs[tx].set_title("Channel:" + str(tx))
    plt.show()


####################################################################

    # data.format=[kspacedatanii, spacedata, header, maskch, B1parray, U90array, Im_Ch_array, AlphaU_full_array,Modelfit per pixel]

    Model=np.array(File[8])[:, X, Y]
    Data_fit=np.array(File[6])[:, X, Y]
    fig, axs = plt.subplots(ncols=1, nrows=TX, constrained_layout=True)
    for tx in range(TX):
        if TX == 1:

            fig1 = axs.plot(Model[0],  label="Modelfit")# [y,x]y^|_>x
            fig1 = axs.plot(Data_fit[0], label="Data ")

        else:


            fig1 = axs[tx].plot(Model[ TX], label="Modelfit")
            fig1 = axs[tx].plot(Image_mask[ TX], label="Data points ")
            axs[tx].set_title("Channel:" + str(tx))

    plt.title("Model fitting function")
    plt.legend()

    plt.ylabel(r'$\frac{S}{S_{0}} $',rotation=0)
    plt.xlabel(r'$\frac{U_{TX}}{ V}$',rotation=0)
    plt.legend()
    plt.grid()
    plt.show()









def Plot_compare_two(Filepath=None,Filepath2=None, rangee=None, cmap=0):  # rang=[start,end]
    if not Filepath:
        Filepath= r"../OLD_.pickled"
        Filepath2 = r"meas_MID73_PEX_B1_HC_slow_FID85216.pickled"

    File = np.array(helper.pload(Filepath), dtype=object)
    File2 = np.array(helper.pload(Filepath2), dtype=object)
    fig, axs = plt.subplots(ncols=2, nrows=1, constrained_layout=True)


    B1plus = np.array(File[4][0][:][
                      :])  # data,0,x,y kspacedatanii, spacedata, header, maskch, B1parray, U90array, Im_Ch_array, AlphaU_full_array)
    #B1plus[B1plus < 1] = 0
    #B1plus[B1plus > 15] = 0
    B1plus2 = np.array(File2[4][0][:][
                      :])  # data,0,x,y kspacedatanii, spacedata, header, maskch, B1parray, U90array, Im_Ch_array, AlphaU_full_array)
    #B1plus2[B1plus2 < 1] = 0
    #B1plus2[B1plus2 > 15] = 0

    fig1 = axs[0].imshow(B1plus, cmap='inferno')
    fig2 = axs[1].imshow(B1plus2, vmin=5, vmax=9, cmap='inferno')
    cbar = fig.colorbar(fig1)
    cbar.set_label(r'$\frac{μT}{\sqrt{kW}} $',rotation=0)

    plt.show()
