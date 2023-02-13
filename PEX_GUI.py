
import os




import PyQt6 as qt
from PyQt6 import QtWidgets
import PEX_helper as help
from PyQt6.QtCore import QProcess
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from design import Ui_Dialog
import numpy as np
from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
class MyApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.pro2 = None
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # connect  the writable parameters from the design.py
        self.ui.lineEdit_17.setText \
            ("/home/teip/Desktop/0_masterarbeit/data/PEX_7Tesla_Buch/2022.01.26_Franksloop_fast_reallyfast/RAW")
        self.ui.lineEdit_12.setText("6")
        self.ui.lineEdit_16.setText("0")
        self.ui.lineEdit_14.setText("0.2")
        self.ui.lineEdit_13.setText("10")
        self.ui.lineEdit_2.setText("300")
        self.ui.lineEdit_3.setText("4")
        self.ui.lineEdit_15.setText("0.6")
        self.ui.lineEdit.setText("10")
        self.ui.lineEdit_11.setText("800")
        self.ui.lineEdit_10.setText("17")
        self.ui.lineEdit_18.setText("/home/teip/Desktop/0_masterarbeit/data/PEX_7Tesla_Buch/2022.06.23_1xHBC+2xHBC/1chanel/meas_MID153_PEX_B1_sd_1ch_fast_tBP_HBC_M8_2mmStyro_L_loop_phaseRL_0m__FID22552_Results/0_File.pickled")
        self.ui.lineEdit_19.setText("20")

        self.ui.label_3.setToolTip("Path to the PEX(*.dat) file/folder")
        self.ui.label_2.setToolTip("T1 time in s for the Phantom. For different media use average  ")
        self.ui.label_5.setToolTip("Number of CPU kernels for multiprocessing. Don't do 100%")
        self.ui.label_4.setToolTip("loss in dB, only necessary if you want to compare it with Simulation and you know all the losses in the Scanner and cabel")
        self.ui.label_7.setToolTip("mask threshold 0.3-0.05, try a higher value first")
        self.ui.label_11.setToolTip("lowest Voltage step in V Init parameter for model fitting")
        self.ui.label_6.setToolTip("highest Voltage step in V Init parameter for model fitting")
        self.ui.label_10.setToolTip("Voltage step size in V Init parameter for model fitting")
        self.ui.label_9.setToolTip("Minimum Voltage threshold, choose at least lowest Voltage level ")
        self.ui.label_8.setToolTip("Maximum Voltage threshold, choose at least double highest Voltage level ")
        self.ui.label_13.setToolTip("Some Scanner have a limit of how many Voltages can be measured after each other. B.U.F.Fs 7T scanner limit is 17 ")
        self.ui.label_17.setToolTip(
            "vmax value for a plot in matplotlib ")
        self.ui.toolButton.clicked.connect(self.getfile)
        self.ui.toolButton_2.clicked.connect(self.getfile2)


        # connect Push buttons from design.py

        self.ui.pushButton_2.pressed.connect(self.process2)
        self.ui.pushButton.pressed.connect(self.process)

        #connect kill button
        self.ui.pushButton_3.pressed.connect(self.kill)

        self.ui.plainTextEdit.setReadOnly(True)
        # Create a layout for the writable parameters
        # Get the number of kernels
        kernels = os.cpu_count()

        # Print the number of kernels
        print(f'Number of kernels: {kernels}')
        self.ui.label_16.setText(str(kernels))

        # Set the window properties
        self.setWindowTitle("PEX GUI")
        self.setGeometry(100, 130, 1300, 1000)


    def getfile2(self):# get files with .dat and write them in the textEdit
        dlg = qt.QtWidgets.QFileDialog(filter="pickled Files (*.pickled)")
        dlg.show()

        if dlg.exec():
            self.ui.lineEdit_18.setText(str(dlg.selectedFiles()[0]))
    def getfile(self):# get files with *.pickled and write them in the textEdit
        dlg = qt.QtWidgets.QFileDialog(filter="Data Files (*.dat)")
        dlg.show()

        if dlg.exec():
            self.ui.lineEdit_17.setText(str(dlg.selectedFiles()[0]))

    def message(self, s): # print message
        self.ui.plainTextEdit.appendPlainText(s)
    def kill(self): # kill process
        self.pro2.kill()

    def process(self): # run PEX debugger process

        data = np.array(help.pload(str(self.ui.lineEdit_18.text())),
                        dtype=object)  # data.format=[kspacedatanii, spacedata, header, maskch, B1parray, U90array, Im_Ch_array, AlphaU_full_array,Modelfit per pixel]
        header = data[2]
        nv = []
        for voltage_steps in range(int(header[f"MeasYaps"]["sWiPMemBlock", "alFree", "2"])):
            try:
                nv.append(header[f"MeasYaps"]["sWiPMemBlock", "adFree", str(voltage_steps)])
            except KeyError:
                pass

        nv.append(0)
        print("Voltage steps ", nv)

        # plot B1_plus map
        self.B1_plus = np.array(np.abs(data[4]))

        if self.B1_plus.shape[0] != 1:
            self.mask =np.transpose(data[3],(2,0,1))
            self.mask_parts=np.transpose(data[3],(2,0,1))

            self.B1_plus_parts= self.B1_plus* self.mask
            self.B1_plus = np.sum(np.abs(self.B1_plus*  self.mask), axis=0)
            self.mask = np.sum(np.abs(self.mask), axis=0)
            self.mask[self.mask > 0] = 1
            print("TX",self.B1_plus.shape[0])

        else:
            self.B1_plus = np.abs(self.B1_plus)[0]
            self.mask = data[3]
            self.B1_plus_parts = self.B1_plus * self.mask[:,:,0]

        # make figure for B1+ map

        sc= matplofigure(self, width=4, height=4, dpi=80)
        fig2=sc.axes.imshow(self.B1_plus ,vmax=float(self.ui.lineEdit_19.text()), cmap='inferno')
        cbar=sc.fig.colorbar(fig2)
        cbar.set_label(r'$\frac{μT}{\sqrt{kW}} $', rotation=0)
        sc.axes.set_title("B1+ map")

        # make figure for model/data fitting plots
        sc2 = matplofigure(self, width=4, height=3, dpi=80)

        # make figure for mask chennels plots
        sc3 = matplofigure(self, width=4, height=3, dpi=80)
        sc3.axes.imshow(self.mask)

        sc3.axes.set_title("Mask of the Image")


        # connect Layoutform with widget for  matplofigure
        self.ui.formLayout.addWidget(sc)
        self.ui.formLayout_2.addWidget(sc2)
        self.ui.formLayout_3.addWidget(sc3)
        def click(event):   # small porgramm that collects the x and y coordinates of the click


            sc2.axes.cla() # clear figure


            #Model/data fitting plots


            self.Model = np.array(data[8])
            self.Data_fit = np.array(data[6])


            # plotting only the model datta for the highest B1+ map point
            if self.Model.shape[0]!=1:

                index_array=np.argmax(self.B1_plus_parts,axis=0)
                index_array=np.expand_dims(index_array, axis=2)

                self.Data_fit=np.take_along_axis(self.Data_fit, np.expand_dims(index_array, axis=0), axis=0)
                self.Model = np.take_along_axis(self.Model, np.expand_dims(index_array, axis=0), axis=0)



            sc2.axes.plot(self.Model[0, int(event.ydata), int(event.xdata)], label="Modelfit")
            sc2.axes.plot(self.Data_fit[0, int(event.ydata), int(event.xdata)],'r+', label="Data")
            sc2.axes.set_title(r"Model fit of the data",)
            sc2.fig.legend()
            sc2.axes.set_ylabel(r'$\frac{S}{S_{0}} $',rotation=0)

            sc2.draw() # update figure
            sc3.draw()
            self.ui.label.setText("X-axis:  " + str(int(event.xdata)) + " Y-axis:" + str(
                int(event.ydata)))  # sets label widget with coordinates


        sc.mpl_connect("button_press_event", click) # click event for the figure collector





        plt.show(block=False)

        # remove matplofigure from layout for new  plots
        self.ui.formLayout.removeWidget(sc)
        self.ui.formLayout_2.removeWidget(sc2)
        self.ui.formLayout_3.removeWidget(sc3)




    def process2(self): # run PEX  process
        if self.pro2 is None:  # No process running.
            self.message("Executing process")
            self.pro2 = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.pro2.readyReadStandardOutput.connect(self.stdout)
            self.pro2.readyReadStandardError.connect(self.stderr)
            self.pro2.stateChanged.connect(self.state)
            self.pro2.finished.connect(self.process_finish)  # Clean up once complete.

            self.pro2.start("python", ["PEX_RECON.py" ,"--pathfolder", str(self.ui.lineEdit_17.text()) ,"--jobs"
                                    ,str(int(self.ui.lineEdit_12.text())), "--loss_in_db",
                                    str(float(self.ui.lineEdit_16.text())), "--mask",
                                    str(float(self.ui.lineEdit_14.text())), "--startV",
                                    str(int(self.ui.lineEdit_13.text())), "--stopV",
                                    str(int(self.ui.lineEdit_2.text())), "--stepV", str(int(self.ui.lineEdit_3.text())),
                                    "--T1", str(float(self.ui.lineEdit_15.text())), "--min_V",
                                    str(int(self.ui.lineEdit.text())), "--max_V", str(int(self.ui.lineEdit_11.text())),
                                    "--max_Voltage_steps", str(int(self.ui.lineEdit_10.text()))])

    def stderr(self): # display error messages
        data = self.pro2.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.message(stderr)

    def stdout(self): #
        data = self.pro2.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.message(stdout)

    def state(self, state): # display state
        states = {
            QProcess.ProcessState.NotRunning: 'Not running',
            QProcess.ProcessState.Starting: 'Starting',
            QProcess.ProcessState.Running: 'Running',
        }
        state_name = states[state]
        self.message(f"State changed: {state_name}")

    def process_finish(self): # clean up
        self.message("Process finished.")
        self.pro2 = None


class matplofigure(FigureCanvasQTAgg):
 #   Matplotlib figure canvas class for matplot figure in a separate Box
    def __init__(self, B1plus=None, vmax2=60, vmin2=0, width=5, height=4, dpi=100 ):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        super(matplofigure, self).__init__( self.fig )



if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()

    sys.exit(app.exec())

