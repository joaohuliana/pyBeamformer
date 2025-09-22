import numpy as np
from PyQt6.QtWidgets import *
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtCore import pyqtSignal, Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Rectangle, RectangleSelector

class ProgressBarWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()

        # Create a QProgressBar
        self.progress_bar = QProgressBar(self)
        vbox.addWidget(self.progress_bar)

        self.setLayout(vbox)

        self.setGeometry(300, 500, 300, 50)

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint & ~Qt.WindowType.WindowMaximizeButtonHint & ~Qt.WindowType.WindowMinimizeButtonHint)
    def center_on_screen(self):
        # Get the geometry of the screen
        primary_screen = QGuiApplication.primaryScreen()
        screen_rect = primary_screen.availableGeometry()

        # Center the window on the screen
        self.move((screen_rect.width() - self.width()) // 2, (screen_rect.height() - self.height()) // 2)

    def set_title(self, title):
        self.setWindowTitle(title)
    def set_progress_value(self, value):
        self.progress_bar.setValue(value)

    def close_window(self):
        self.close()


class Frames_Dialog(QDialog):
    frames_index = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frames Select")
        self.resize(300, 200)  # Adjust size
        self.center_on_screen()

        self.lbIni = QLabel("Initial Frame:", self)
        self.lbIni.setGeometry(20, 20, 100, 30)  # Adjust position and size

        self.sbIni = QSpinBox(self)
        self.sbIni.setGeometry(130, 20, 100, 30)  # Adjust position and size
        self.sbIni.setMinimum(0)  # Set minimum value
        self.sbIni.setMaximum(100)  # Set maximum value

        self.lbEnd = QLabel("Final Frame:", self)
        self.lbEnd.setGeometry(20, 70, 100, 30)  # Adjust position and size

        self.sbEnd = QSpinBox(self)
        self.sbEnd.setGeometry(130, 70, 100, 30)  # Adjust position and size
        self.sbEnd.setMinimum(0)  # Set minimum value
        self.sbEnd.setMaximum(100)  # Set maximum value

        self.pbOK = QPushButton("OK", self)
        self.pbOK.setGeometry(50, 120, 75, 30)  # Adjust position and size
        self.pbOK.clicked.connect(self.get_values)

        self.pbCancel = QPushButton("Cancel", self)
        self.pbCancel.setGeometry(175, 120, 75, 30)  # Adjust position and size
        self.pbCancel.clicked.connect(self.cancel_action)


    def center_on_screen(self):
        # Get the geometry of the screen
        primary_screen = QGuiApplication.primaryScreen()
        screen_rect = primary_screen.availableGeometry()

        # Center the window on the screen
        self.move((screen_rect.width() - self.width()) // 2, (screen_rect.height() - self.height()) // 2)

    def get_values(self):
        ini = self.sbIni.value()
        end = self.sbEnd.value()
        self.frames_index.emit(ini, end)
        self.accept()
    def cancel_action(self):
        self.frames_index.emit(-1, -1)
        self.accept()


class SaveDisp_Dialog(QDialog):
    saveas = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Save disp as:")
        self.resize(300, 120)  # Adjust size

        self.pbFigure = QPushButton("Figure", self)
        self.pbFigure.setGeometry(20, 30, 75, 30)
        self.pbFigure.clicked.connect(lambda: self.choice_clicked("Figure"))

        self.pbMovie = QPushButton("Movie", self)
        self.pbMovie.setGeometry(100, 30, 75, 30)
        self.pbMovie.clicked.connect(lambda: self.choice_clicked("Movie"))

        self.pbData = QPushButton("Data", self)
        self.pbData.setGeometry(180, 30, 75, 30)
        self.pbData.clicked.connect(lambda: self.choice_clicked("Data"))

        self.pbCancel = QPushButton("Cancel", self)
        self.pbCancel.setGeometry(180, 80, 75, 30)
        self.pbCancel.clicked.connect(lambda: self.choice_clicked("Cancel"))

    def choice_clicked(self, choice):
        self.saveas.emit(choice)
        self.accept()  # Close the dialog


class Disp_area(QDialog):

    def __init__(self, dispmap, prm, parent=None):
        super().__init__()
        self.initUI(dispmap,prm)

    def initUI(self, dispmap, prm):
        self.dispmap = dispmap
        [self.Z, self.X, self.frames] = (self.dispmap.shape)


        self.dx = prm['dx']
        self.dz = prm['dz']
        self.dt = prm['dt']
        self.index = prm['index']

        self.roixi = 0
        self.roizi = 0
        self.roixf = self.X * self.dx
        self.roizf = self.Z * self.dz

        self.setWindowTitle("Area Analyzer")
        self.resize(1000, 700)

        self.max = np.amax(self.dispmap)
        self.min = np.amin(self.dispmap)

        #Disp Axis
        self.dispAxis = QWidget(self)
        self.dispAxis.setObjectName("dispAxis")
        self.dispAxis.setGeometry(30, 10, 591, 611)
        layout = QVBoxLayout(self.dispAxis)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addStretch()
        self.dispfig = plt.Figure()
        self.dispfig.set_facecolor((0.94, 0.94, 0.94))
        self.dispax = self.dispfig.add_axes(rect=[0.17, 0.14, 0.75, 0.8])
        self.dispax.set_xlabel('Lateral [mm]', fontsize=9)
        self.dispax.set_ylabel('Axial [mm]', fontsize=9)
        self.dispax.tick_params(axis='x', labelsize=9)
        self.dispax.tick_params(axis='y', labelsize=9)
        self.dispcanvas = FigureCanvas(self.dispfig)
        layout.addWidget(self.dispcanvas)
        layout.addStretch()

        #Disp Slider
        self.slDisp = QSlider(self)
        self.slDisp.setGeometry(50, 620, 501, 16)
        self.slDisp.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slDisp.setObjectName("slDisp")
        self.slDisp.setEnabled(False)
        self.programmatic_change = True
        self.slDisp.setMinimum(1)  # Set the minimum value
        self.slDisp.setMaximum(self.frames)
        self.slDisp.setValue(self.index)
        self.programmatic_change = False
        self.slDisp.valueChanged.connect(self.slider_changed)

        self.lbDisp = QLabel(self)
        self.lbDisp.setGeometry(560, 620, 61, 20)
        self.lbDisp.setObjectName("lbDisp")
        self.lbDisp.setText(f'{self.index} / {self.frames} ')

        self.pbResetROI = QPushButton(self)
        self.pbResetROI.setGeometry(50, 640, 75, 23)
        self.pbResetROI.setObjectName("pbResetROI")
        self.pbResetROI.setText("Reset ROI")
        self.pbResetROI.clicked.connect(self.select_roi)

        # Signal Axis
        self.signalAxis = QWidget(self)
        self.signalAxis.setObjectName("Widget3")
        self.signalAxis.setGeometry(640, 10, 331, 311)
        layout = QVBoxLayout(self.signalAxis)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addStretch()
        self.signalfig = plt.Figure()
        self.signalfig.set_facecolor((0.94, 0.94, 0.94))
        self.signalax = self.signalfig.add_axes(rect=[0.22, 0.14, 0.75, 0.7])
        self.signalax.set_facecolor((0, 0, 0))
        self.signalax.set_xlabel('Time [ms]', fontsize=9)
        self.signalax.set_ylabel('Displacement [μm]', fontsize=9)
        self.signalax.tick_params(axis='x', labelsize=9)
        self.signalax.tick_params(axis='y', labelsize=9)
        self.signalcanvas = FigureCanvas(self.signalfig)
        layout.addWidget(self.signalcanvas)
        layout.addStretch()

        '''x = [0, 10, 20, 30, 40, 50]
        y = [0, -0.01, -0.02, -0.03, -0.04, -0.05]
        self.signalax.plot(x, y, 'k-')
        self.signalax.set_ylabel('Amplitude [μm]', fontsize=9)
        self.signalax.set_xlabel('Time [ms]', fontsize=9)
        self.signalax.tick_params(axis='x', labelsize=9)
        self.signalax.tick_params(axis='y', labelsize=9)
        self.signalax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
        self.signalcanvas.draw()'''

        #PSD Axis
        self.psAxis = QWidget(self)
        self.psAxis.setObjectName("psAxis")
        self.psAxis.setGeometry(640, 330, 331, 311)
        layout = QVBoxLayout(self.psAxis)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addStretch()
        self.psfig = plt.Figure()
        self.psfig.set_facecolor((0.94, 0.94, 0.94))
        self.psax = self.psfig.add_axes(rect=[0.22, 0.14, 0.75, 0.7])
        self.psax.set_facecolor((0, 0, 0))
        self.psax.set_xlabel('Frequency [Hz]', fontsize=9)
        self.psax.set_ylabel('Amplitude [a.u.]', fontsize=9)
        self.psax.tick_params(axis='x', labelsize=9)
        self.psax.tick_params(axis='y', labelsize=9)
        self.pscanvas = FigureCanvas(self.psfig)
        layout.addWidget(self.pscanvas)
        layout.addStretch()


        self.rect = None
        self.ROItext = None
        self.dispshow_image()
        self.select_roi()

    def dispshow_image(self):
        self.dispax.clear()
        extent = [0, self.X*self.dx, self.Z*self.dz, 0]
        image = self.dispax.imshow(self.dispmap[:, :, self.index - 1], cmap='jet',
                                   extent=extent, vmin = self.min, vmax = self.max, interpolation='none')


        self.dispax.set_xlabel('Lateral [mm]', fontsize=9)
        self.dispax.set_ylabel('Axial [mm]', fontsize=9)
        self.dispax.tick_params(axis='x', labelsize=9)
        self.dispax.tick_params(axis='y', labelsize=9)
        self.dispcanvas.draw()
        self.dispax.set_aspect('equal')
        self.generate_roi()

    def select_roi(self):
        self.PSD = True
        if self.rect is not None:
            self.rect.remove()
            self.rect = None
            self.dispcanvas.draw()
        if self.ROItext is not None:
            self.ROItext.remove()
            self.dispcanvas.draw()
        self.ROItext = self.dispax.text(0.5, 0.5, "Select a ROI \n Right-Click to finish",
                                    color='black', fontsize=12, ha='center', va='center', transform=self.dispax.transAxes, )
        self.dispcanvas.draw()
        self.rs = RectangleSelector(self.dispax, self.draw_roi, useblit=True,
                                    button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                    interactive=True,  props = dict(facecolor='black', edgecolor='black', alpha=0.3, fill=True))
        self.is_drawing = True
        self.connection_id = self.dispfig.canvas.mpl_connect('button_press_event', self.finish_roi)
        self.dispAxis.setCursor(Qt.CursorShape.CrossCursor)

    def draw_roi(self, eclick, erelease, *args, **kwargs):
        if self.is_drawing:
            if self.rect is not None:
                self.rect.remove()
                self.rect = None
                self.dispcanvas.draw()
            # Draw a red rectangle on the selected region
            self.rect = Rectangle((eclick.xdata, eclick.ydata),
                             erelease.xdata - eclick.xdata,
                             erelease.ydata - eclick.ydata,
                             linewidth=2, edgecolor='black', facecolor='none', linestyle = 'dashed')
            self.dispax.add_patch(self.rect)
            self.dispcanvas.draw()
            #

    def finish_roi(self,event):
        #if event.dblclick and event.button == 1:
        if event.button == 3:
            if self.rect is not None:
                self.rs.set_visible(False)
                self.rs.set_active(False)
                self.rs.disconnect_events()
                self.rs = None
                self.roixi = self.rect.get_x()
                self.roizi = self.rect.get_y()
                self.roixf = self.roixi + self.rect.get_width()
                self.roizf = self.roizi + self.rect.get_height()
                self.ROItext.remove()
                self.ROItext = None
                self.rect.remove()
                self.rect = None
                self.dispcanvas.draw()
                self.dispfig.canvas.mpl_disconnect(self.connection_id)
                self.is_drawing = False
                self.generate_roi()
                self.dispAxis.setCursor(Qt.CursorShape.ArrowCursor)
                self.dispshow_image()
                self.generate_signal()
                self.slDisp.setEnabled(True)
            else:
                return

    def generate_roi(self):
        self.rect = Rectangle((self.roixi, self.roizi),
        self.roixf - self.roixi, self.roizf - self.roizi,
                      linewidth=2, edgecolor='black', facecolor='none', linestyle='dashed')
        self.dispax.add_patch(self.rect)
        self.dispcanvas.draw()

    def generate_signal(self):
        self.signalMean = np.zeros((self.frames))
        self.timedim = np.zeros((self.frames))
        for f in range(0,self.frames):
            sum = 0
            count = 0
            for z in range(int(self.roizi/self.dz),int(self.roizf/self.dz)):
                for x in range(int(self.roixi/self.dx),int(self.roixf/self.dx)):
                    sum = sum + self.dispmap[z,x,f]
                    count = count + 1

            self.signalMean[f] = sum/count
            self.timedim[f] = f*self.dt*1000
        self.plotSignal()

    def plotSignal(self):
        self.signalax.clear()
        self.signalax.set_facecolor((1, 1, 1))
        self.signalax.plot(self.timedim, self.signalMean, 'k-')
        self.signalcanvas.draw()

        self.plotMarker = self.signalax.plot(self.timedim[self.index - 1], self.signalMean[self.index - 1], 'ro')
        self.signalax.set_ylabel('Amplitude [μm]', fontsize=9)
        self.signalax.set_xlabel('Time [ms]', fontsize=9)
        self.signalax.tick_params(axis='x', labelsize=9)
        self.signalax.tick_params(axis='y', labelsize=9)
        self.signalax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        self.signalax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
        self.signalax.set_title(f'{self.timedim[self.index - 1]: .3f} ms , {self.signalMean[self.index - 1]: .3f} μm')
        self.signalcanvas.draw()
        if self.PSD:
            self.plotPSD()

    def plotSignalupdate(self):
        self.signalax.lines[1].remove()
        self.plotMarker = self.signalax.plot(self.timedim[self.index - 1], self.signalMean[self.index - 1], 'ro')
        self.signalax.set_title(f'{self.timedim[self.index - 1]: .3f} ms , {self.signalMean[self.index - 1]: .3f} μm')
        self.signalcanvas.draw()

    def plotPSD(self):
        [ps, freq] = self.powerSpectrum(self.signalMean, 1/self.dt)
        self.psax.clear()
        self.psfig.set_facecolor((0.95, 0.95, 0.95))
        self.psax.set_facecolor((1, 1, 1))
        self.psax.plot(freq, ps, 'k-')
        self.psax.set_ylabel('Amplitude [a.u.]', fontsize=9)
        self.psax.set_xlabel('Frequency [Hz]', fontsize=9)
        self.psax.tick_params(axis='x', labelsize=9)
        self.psax.tick_params(axis='y', labelsize=9)
        self.psax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        self.psax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
        self.pscanvas.draw()
        self.PSD = False

    def update_frame(self):
        self.index = self.slDisp.value()
        self.lbDisp.setText(f'{self.index} / {self.frames} ')
        self.dispshow_image()
        #self.plotSignal()
        self.plotSignalupdate()

    def slider_changed(self):
        if self.programmatic_change:
            return
        else:
            self.update_frame()


    def powerSpectrum(self,signal, fs):
        # Compute the FFT of the signal
        interfft = np.fft.fft(signal, n=1024)

        # Compute the power spectrum (magnitude squared)
        interps = np.abs(interfft) ** 2

        # Frequencies corresponding to the FFT result
        # For real input, the FFT result is symmetric, and the positive frequencies are given by:
        frequencies = np.fft.fftfreq(len(interfft), d=1 / fs)
        positive_frequencies = frequencies[:len(frequencies) // 2]
        ps = interps[:len(interps) // 2]
        return ps, positive_frequencies


    def closeEvent(self, event):
        return
       # self.retranslateUi(MainWindow)


    '''start_time = time.time()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")'''

class Error_Dialog(QDialog):

    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(300, 120)  # Adjust size

        # Create the label with an initial empty text
        self.label = QLabel('', self)
        self.label.setGeometry(20, 30, 260, 60)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.label.setWordWrap(True)
        # Create the "Finish" button
        self.pbFinish = QPushButton("OK", self)
        self.pbFinish.setGeometry(115, 80, 75, 30)
        self.pbFinish.clicked.connect(self.finish_clicked)


    def set_text(self, text):
        self.label.setText(text)

    def finish_clicked(self):
        self.accept()  # Close the dialog
