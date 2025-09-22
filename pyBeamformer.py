import sys, ctypes, os, math, time
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout,QApplication, QMainWindow, QSizePolicy
from PyQt6.QtGui import QIcon
from tkinter import filedialog as fd

import numpy as np
import scipy
from scipy.io import savemat, whosmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Rectangle, RectangleSelector
import matplotlib.ticker as ticker

from pyUStools.pyUS_processing import bmode
from pyUStools.ui_utilities import ProgressBarWindow, Error_Dialog

class pyBeamformer(QtWidgets.QMainWindow):
# ----------------------------------------General UI--------------------------------------------------------------------
    def __init__(self):
        super().__init__()

        # Get the path to the bundled UI file
        ui_file_path = self.resource_path("gui/pybf_gui.ui")
        ui_icon_path = self.resource_path("gui/Icon32.ico")
        self.setWindowIcon(QIcon(ui_icon_path))
        # Load the UI file
        uic.loadUi(ui_file_path, self)
        app_icon = QIcon(ui_icon_path)
        app.setWindowIcon(app_icon)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, False)
        self.setFixedSize(1011, 700)
        #pyinstaller --onefile --add-data ".\gui\pybf_gui.ui:gui" --add-binary 'D:\Programs\C++\US_tools_cpp\us_tools.dll;.' --icon=D:\Programs\C++\pyBeamforming\gui\Icon.ico --add-data="D:\Programs\C++\pyBeamforming\gui\Icon32.ico;." --noconsole pyBeamformer.py
        #pip freeze > requirements.txt
        #pip install -r requirements.txt

        #center mainwindow
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        self.initUI()

    def resource_path(self, relative_path):
        # Function to get the path to a bundled resource
        if getattr(sys, 'frozen', False):
            # Running in a bundle (PyInstaller)
            return os.path.join(sys._MEIPASS, relative_path)
        else:
            # Running in a normal Python environment
            return os.path.join(os.path.abspath("."), relative_path)

    def initUI(self):
        self.menuExit.triggered.connect(self.close)

        #Rawaxis
        layout = QVBoxLayout(self.rawAxis)
        layout.setContentsMargins(1,1,1,1)
        layout.addStretch()
        self.rawfig = plt.Figure()
        self.rawfig.set_facecolor((0.0, 0.0, 0.0))
        self.rawax = self.rawfig.add_axes(rect=[0.17, 0.14, 0.75, 0.8])
        self.rawax.set_facecolor((0, 0, 0))
        self.rawax.set_xlabel('Lateral [mm]', fontsize=9)
        self.rawax.set_ylabel('Axial [mm]', fontsize=9)
        self.rawax.tick_params(axis='x', labelsize=9)
        self.rawax.tick_params(axis='y', labelsize=9)
        self.rawcanvas = FigureCanvas(self.rawfig)
        layout.addWidget(self.rawcanvas)
        layout.addStretch()

        # BFaxis
        layout = QVBoxLayout(self.bfAxis)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addStretch()
        self.bffig = plt.Figure()
        self.bffig.set_facecolor((0.0, 0.0, 0.0))
        self.bfax = self.bffig.add_axes(rect=[0.17, 0.14, 0.75, 0.8])
        self.bfax.set_facecolor((0, 0, 0))
        self.bfax.set_xlabel('Lateral [mm]', fontsize=9)
        self.bfax.set_ylabel('Axial [mm]', fontsize=9)
        self.bfax.tick_params(axis='x', labelsize=9)
        self.bfax.tick_params(axis='y', labelsize=9)
        self.bfcanvas = FigureCanvas(self.bffig)
        layout.addWidget(self.bfcanvas)
        layout.addStretch()

        self.rawloaded = False
        self.beamformed = False


        # parameters
        self.c = self.sbC.value()
        self.fs = (self.sbSf.value() * 1000000)
        if self.cbMode.currentText() == 'Conventional Ultrasound':
            self.dz = (self.c / (2 * self.fs)) * 1000
        elif self.cbMode.currentText() == 'Photoacoustic':
            self.dz = (self.c / (self.fs)) * 1000

        self.dx = self.sbdx.value()

        self.ctag = self.sbCtag.value()
        self.aperture = self.sbAp.value()

        self.menuLoadmat.triggered.connect(self.load_mat)
        self.menuClosescene.triggered.connect(self.close_mat)
        self.slraw.valueChanged.connect(self.rawslider_changed)
        self.cbCompression.activated.connect(self.compression_type)

        # US Parameter Panel
        self.sbSf.valueChanged.connect(self.sf_changed)
        self.sbdx.valueChanged.connect(self.dx_changed)
        self.sbC.valueChanged.connect(self.c_changed)
        self.sbAp.valueChanged.connect(self.ap_changed)
        self.sbCtag.valueChanged.connect(self.ctag_changed)
        self.cbMode.activated.connect(self.modality)

        self.pbRun.clicked.connect(self.run_bf)
        self.pbSave.clicked.connect(self.save_bf)
        self.pbPreview.clicked.connect(self.preview_bf)


# ------------------------------------------Load Raw--------------------------------------------------------------------
    def load_mat(self):
        if self.rawloaded == True:
            self.close_mat()

        filetypes = (
            ('Mat files', '*.mat'),
        )
        matfilename = fd.askopenfilename(
            title='Select a .mat file',
            initialdir='/',
            filetypes=filetypes)
        if matfilename:
            #load
            auxnamelist = whosmat(matfilename)
            auxname = auxnamelist[0]
            auxload = scipy.io.loadmat(matfilename)
            self.rfdata = auxload[auxname[0]]
            if self.rfdata.shape == (1,1):
                dialog = Error_Dialog("Load Error")
                # Set the label text
                dialog.set_text("Load error! The .mat file must contain only a 3D (or 2D) numeric matrix")
                # Show the dialog
                dialog.exec()
                return

            filename_only = matfilename.split("/")[-1]
            filename_only =filename_only.replace(".mat", "")
            filename_only = filename_only.replace("_", " ")

            self.lbFileName.setText(str(filename_only))
            self.lbFileName.setAlignment(Qt.AlignmentFlag.AlignHCenter)  # Center the text horizontally
            self.lbFileName.setStyleSheet("font-weight: bold; font-size: 12pt;")

            #dimensions
            if self.rfdata.ndim == 3:
                [self.Z, self.X, self.frames] = (self.rfdata.shape)
            else:
                [self.Z, self.X] = (self.rfdata.shape)
                self.frames = 1
            self.programmatic_change = True
            self.rfdata_frames_update()
            self.programmatic_change = False

            # parameters
            self.c = self.sbC.value()
            self.fs = (self.sbSf.value() * 1000000)
            if self.cbMode.currentText() == 'Conventional Ultrasound':
                self.dz = (self.c / (2 * self.fs)) * 1000
            elif self.cbMode.currentText() == 'Photoacoustic':
                self.dz = (self.c / (self.fs)) * 1000

            self.dx = self.sbdx.value()

            self.ctag = self.sbCtag.value()
            self.aperture = self.sbAp.value()

            self.rawloaded = True

            self.activate_guiRaw(True)
            self.update_frame()
        else:
            return

    def rfdata_frames_update(self):
        self.slraw.setMinimum(1)  # Set the minimum value
        self.slraw.setMaximum(self.frames)

    def close_mat(self):
        self.activate_guiRaw(False)
        self.rawax.clear()
        self.rawfig.set_facecolor((0, 0, 0))
        self.rawax.set_facecolor((0, 0, 0))
        self.rawcanvas.draw()
        self.bfax.clear()
        self.bffig.set_facecolor((0, 0, 0))
        self.bfax.set_facecolor((0, 0, 0))
        self.bfcanvas.draw()
        self.rawloaded = False
        self.beamformed = False
        self.pbSave.setEnabled(False)

# ------------------------------------------Save BF------------------------------------------------------------------
    def save_bf(self):
        filetypes = (
            ('Mat files', '*.mat'),
        )

        # Open the file dialog for saving
        filename = fd.asksaveasfilename(
            title='Save As',
            filetypes=filetypes,
            defaultextension='.mat'  # Optional: specify default extension
        )

        # Check if a filename was selected
        if filename:
            bf_data = self.bfdata.astype(np.int16)
        # Create a dictionary with key-value pairs for each variable
            variables_dict = {
                'bf_data': bf_data,
            }

            # Save the variables as a .mat file
            savemat(filename, variables_dict)
        else:
            return

# ------------------------------------------UI control------------------------------------------------------------------
    def activate_guiRaw(self, value):

        self.menuClosescene.setEnabled(value)

        #Raw axis
        self.rawfig.set_facecolor((0.94, 0.94, 0.94))
        self.slraw.setEnabled(value)
        self.lbraw.setEnabled(value)

        self.cbCompression.setEnabled(value)
        self.slraw.setEnabled(value)
        self.pbRun.setEnabled(value)
        self.pbPreview.setEnabled(value)
        self.cbBftype.setEnabled(value)
        #self.pbSaveBM.setVisible(value)

    def activate_guiBf(self, value):
        #Bf axis
        self.bffig.set_facecolor((0.94, 0.94, 0.94))

# ----------------------------------------Display Images--------------------------------------------------------------

    def show_image(self):
        self.rawax.clear()
        compression = self.cbCompression.currentText()
        b_mode = bmode(self.rfdata, self.current_frame, compression);

        extent = [0, self.X * self.dx, self.Z * self.dz,0]
        if self.cbMode.currentText() == 'Conventional Ultrasound':
            self.rawax.imshow(b_mode, cmap='gray',
                             extent=extent)
        elif self.cbMode.currentText() == 'Photoacoustic':
            self.rawax.imshow(b_mode, cmap='hot',
                             extent=extent)

        self.rawax.set_title('Raw Data', fontsize=12)
        self.rawax.set_xlabel('Lateral [mm]', fontsize=9)
        self.rawax.set_ylabel('Axial [mm]', fontsize=9)
        self.rawax.tick_params(axis='x', labelsize=9)
        self.rawax.tick_params(axis='y', labelsize=9)
        self.rawcanvas.draw()

    def update_frame(self):
        self.current_frame = self.slraw.value()
        self.lbraw.setText(f'{self.current_frame} / {self.frames} ')
        self.show_image()
        self.bfshow_image()

    def rawslider_changed(self):
        if self.programmatic_change:
            return
        else:
            self.update_frame()

    def compression_type(self):
        if self.rawloaded == True:
            self.update_frame()

    def modality(self):
        if self.cbMode.currentText() == 'Conventional Ultrasound':
            self.dz = (self.c / (2 * self.fs)) * 1000
        elif self.cbMode.currentText() == 'Photoacoustic':
            self.dz = (self.c / (self.fs)) * 1000
        if self.rawloaded == True:
            self.update_frame()

# ---------------------------------------Display BF Images--------------------------------------------------------------

    def bfshow_image(self):
        if self.beamformed:
            self.bfax.clear()
            compression = self.cbCompression.currentText()
            b_mode = bmode(self.bfdata, self.current_frame, compression);

            extent = [0, self.X * self.dx, self.Z * self.dz,0]
            if self.cbMode.currentText() == 'Conventional Ultrasound':
                self.bfax.imshow(b_mode, cmap='gray',
                        extent=extent)
            elif self.cbMode.currentText() == 'Photoacoustic':
                self.bfax.imshow(b_mode, cmap='hot',
                                 extent=extent)

            self.bfax.set_title('Beamformed Data', fontsize=12)
            self.bfax.set_xlabel('Lateral [mm]', fontsize=9)
            self.bfax.set_ylabel('Axial [mm]', fontsize=9)
            self.bfax.tick_params(axis='x', labelsize=9)
            self.bfax.tick_params(axis='y', labelsize=9)
            self.bfcanvas.draw()
        else:
            return

    def bfshow_preview(self):
        self.bfax.clear()
        compression = self.cbCompression.currentText()
        b_mode = bmode(self.bfprev, self.current_frame, compression);

        extent = [0, self.X * self.dx, self.Z * self.dz,0]
        if self.cbMode.currentText() == 'Conventional Ultrasound':
            self.bfax.imshow(b_mode, cmap='gray',
                             extent=extent)
        elif self.cbMode.currentText() == 'Photoacoustic':
            self.bfax.imshow(b_mode, cmap='hot',
                             extent=extent)

        self.bfax.set_title('Preview of Beamformed Data', fontsize=12)
        self.bfax.set_xlabel('Lateral [mm]', fontsize=9)
        self.bfax.set_ylabel('Axial [mm]', fontsize=9)
        self.bfax.tick_params(axis='x', labelsize=9)
        self.bfax.tick_params(axis='y', labelsize=9)
        self.bfcanvas.draw()


# ---------------------------------------US Parameters Panel----------------------------------------------------------
    def sf_changed(self):
        self.fs = (self.sbSf.value() * 1000000)
        if self.cbMode.currentText() == 'Conventional Ultrasound':
            self.dz = (self.c / (2 * self.fs)) * 1000
        elif self.cbMode.currentText() == 'Photoacoustic':
            self.dz = (self.c / (self.fs)) * 1000
        if self.rawloaded == True:
            self.update_frame()

    def dx_changed(self):
        self.dx = self.sbdx.value()
        if self.rawloaded == True:
            self.update_frame()

    def c_changed(self):
        self.c = self.sbC.value()
        if self.cbMode.currentText() == 'Conventional Ultrasound':
            self.dz = (self.c / (2 * self.fs)) * 1000
        elif self.cbMode.currentText() == 'Photoacoustic':
            self.dz = (self.c / (self.fs)) * 1000
        if self.rawloaded == True:
            self.update_frame()

    def ap_changed(self):
        self.aperture = self.sbAp.value()

    def ctag_changed(self):
        self.ctag = self.sbCtag.value()

# --------------------------------------------Beamforming---------------------------------------------------------------
    def run_bf(self):

        # dll load
        dll_path =  self.resource_path("US_tools_cpp/us_tools.dll")
        bfdll = ctypes.CDLL(dll_path)
        bfdll.das.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.int32),  # rawData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        bfdll.dmas.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.int32),  # rawData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        bfdll.fdmas.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.int32),  # rawData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        bfdll.delaymap.restype = None
        bfdll.delaymap.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        bfdll.delaymapPA.restype = None
        bfdll.delaymapPA.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        delays = np.zeros((self.Z, 2 * self.X))
        dasprm = np.zeros((6))
        dprm = np.zeros((5))

        dasprm[0] = self.Z
        dasprm[1] = self.X
        dasprm[2] = self.dz
        dasprm[3] = self.dx
        dasprm[4] = self.aperture
        dasprm[5] = self.ctag
        dprm[0] = dasprm[0]
        dprm[1] = dasprm[1]
        dprm[2] = dasprm[3]
        dprm[3] = self.c
        dprm[4] = self.fs

        if self.cbMode.currentText() == 'Conventional Ultrasound':
            bfdll.delaymap(delays, dprm)
        elif self.cbMode.currentText() == 'Photoacoustic':
            bfdll.delaymapPA(delays, dprm)


        if self.rfdata.ndim == 3:
            app = QApplication.instance()
            progress_bar_window = ProgressBarWindow()
            progress_bar_window.center_on_screen()
            progress_bar_window.set_title("Beamforming")
            progress_bar_window.show()
            intermedraw = np.zeros((self.Z, self.X))
            interbf = np.zeros((self.Z, self.X))
            self.bfdata = np.zeros((self.Z, self.X, self.frames))
            for i in range (0, self.frames):
                intermedraw = self.rfdata[:,:,i]
                if self.cbBftype.currentText() == 'Delay and Sum':
                    bfdll.das(interbf, np.int32(intermedraw.flatten()), np.int32(delays.flatten()), dasprm)
                elif self.cbBftype.currentText() == 'Delay Multiply and Sum':
                    bfdll.dmas(interbf, np.int32(intermedraw.flatten()), np.int32(delays.flatten()), dasprm)
                elif self.cbBftype.currentText() == 'Filtered Delay Multiply and Sum':
                    bfdll.fdmas(interbf, np.int32(intermedraw.flatten()), np.int32(delays.flatten()), dasprm)
                self.bfdata[:, :, i] = interbf
                progress_bar_window.set_progress_value(int((i / (self.frames)) * 100))
                progress_bar_window.repaint()
                app.processEvents()
        else:
            self.bfdata = np.zeros((self.Z, self.X))
            if self.cbBftype.currentText() == 'Delay and Sum':
                bfdll.das(self.bfdata[:, :], np.int32(self.rfdata.flatten()), np.int32(delays.flatten()), dasprm)
            elif self.cbBftype.currentText() == 'Delay Multiply and Sum':
                bfdll.dmas(self.bfdata[:, :], np.int32(self.rfdata.flatten()), np.int32(delays.flatten()), dasprm)
            elif self.cbBftype.currentText() == 'Filtered Delay Multiply and Sum':
                bfdll.fdmas(self.bfdata[:, :], np.int32(self.rfdata.flatten()), np.int32(delays.flatten()), dasprm)
        self.beamformed = True
        self.activate_guiBf(True)
        self.pbSave.setEnabled(True)
        self.update_frame()

    def preview_bf(self):

        # dll load
        dll_path =  self.resource_path("US_tools_cpp/us_tools.dll")
        bfdll = ctypes.CDLL(dll_path)
        bfdll.das.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.int32),  # rawData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        bfdll.dmas.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.int32),  # rawData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        bfdll.fdmas.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.int32),  # rawData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        bfdll.delaymap.restype = None
        bfdll.delaymap.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        bfdll.delaymapPA.restype = None
        bfdll.delaymapPA.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),  # bfData
            np.ctypeslib.ndpointer(dtype=np.float64)  # prm
        ]

        delays = np.zeros((self.Z, 2 * self.X))
        dasprm = np.zeros((6))
        dprm = np.zeros((5))

        dasprm[0] = self.Z
        dasprm[1] = self.X
        dasprm[2] = self.dz
        dasprm[3] = self.dx
        dasprm[4] = self.aperture
        dasprm[5] = self.ctag

        dprm[0] = dasprm[0]
        dprm[1] = dasprm[1]
        dprm[2] = dasprm[3]
        dprm[3] = self.c
        dprm[4] = self.fs

        if self.cbMode.currentText() == 'Conventional Ultrasound':
            bfdll.delaymap(delays, dprm)
        elif self.cbMode.currentText() == 'Photoacoustic':
            bfdll.delaymapPA(delays, dprm)

        interraw = np.zeros((self.Z, self.X))
        if self.rfdata.ndim == 3:
            interraw = self.rfdata[:,:,self.current_frame-1]
        else:
            interraw = self.rfdata[:, :]
        self.bfprev = np.zeros((self.Z, self.X))

        if self.cbBftype.currentText() == 'Delay and Sum':
            bfdll.das(self.bfprev[:, :], np.int32(interraw.flatten()), np.int32(delays.flatten()), dasprm)
        elif self.cbBftype.currentText() == 'Delay Multiply and Sum':
            bfdll.dmas(self.bfprev[:, :], np.int32(interraw.flatten()), np.int32(delays.flatten()), dasprm)
        elif self.cbBftype.currentText() == 'Filtered Delay Multiply and Sum':
            bfdll.fdmas(self.bfprev[:, :], np.int32(interraw.flatten()), np.int32(delays.flatten()), dasprm)

        self.activate_guiBf(True)
        self.bfshow_preview()

app = QtWidgets.QApplication(sys.argv)
window = pyBeamformer()
window.show()
app.exec()