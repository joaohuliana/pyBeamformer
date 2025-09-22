import math
import statistics
import numpy as np

from scipy.io import savemat, whosmat
from scipy.signal import hilbert2, butter, lfilter, detrend, medfilt
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

from PyQt6.QtWidgets import QApplication
from tkinter import filedialog as fd

from pyUStools.ui_utilities import *


#------------------------------------RF Data--------------------------------------------------
def bmode(rfdata, currentframe, compression):
    #raw = np.real(rfdata[:, :, currentframe - 1])
    if rfdata.ndim == 3:
        raw = rfdata[:, :, currentframe - 1]
    else:
        raw = rfdata[:, :]

    if compression == 'Square Root (a.u.)':
        b_mode = np.sqrt(abs(hilbert2(raw)))
    elif compression == 'Logarithm (dB)':
        aux = abs(hilbert2(raw))
        aux = aux/np.amax(aux)
        b_mode = 20*np.log10(aux)
    elif compression == 'Mod Logarithm (dB)':
        aux = abs(hilbert2(raw))
        aux = aux / np.amax(aux)
        b_mode = 20 * np.log10(0.01+aux)
    else:
        b_mode = abs(hilbert2(raw))
    return b_mode

def cropRfdata(rfdata,prm,roi):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Croping RF Data")
    progress_bar_window.show()

    fs = prm['fs']
    dt = 1 / fs
    fc = prm['fc']
    df = prm['df']
    ini = prm['ini']
    end = prm['end']

    xi = roi['xi']
    xf = roi['xf']
    zi = roi['zi']
    zf = roi['zf']

    crop_rf = np.zeros((zf-zi, xf-xi, math.floor((end-ini)/df)))
    [Z, X, cframes] = (crop_rf.shape)
    [_, _, frames] = (rfdata.shape)

    i = ini
    j = 0
    while i<=(frames-1) and j<= (cframes-1):
        crop_rf[:,:,j] = rfdata[zi:zf, xi:xf, i]
        i = i + df
        j = j + 1
        progress_bar_window.set_progress_value(int((i / (frames)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()

    return crop_rf

def calculateIQ(rfdata,prm,roi):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Calculating IQ Data")
    progress_bar_window.show()

    fs = prm['fs']
    dt = 1 / fs
    fc = prm['fc']
    df = prm['df']
    ini = prm['ini']
    end = prm['end']

    xi = roi['xi']
    xf = roi['xf']
    zi = roi['zi']
    zf = roi['zf']

    crop_rf = np.zeros((zf-zi, xf-xi, math.floor((end-ini)/df)))
    [Z, X, cframes] = (crop_rf.shape)
    [_, _, frames] = (rfdata.shape)

    i = ini
    j = 0
    while i<=(frames-1) and j<= (cframes-1):
        crop_rf[:,:,j] = rfdata[zi:zf, xi:xf, i]
        i = i + df
        j = j + 1

    t = np.arange(0, Z, 1)
    t = t*dt
    t2D = np.tile(t, (X, 1)).T
    IQbuffer = np.zeros((Z,X,cframes), dtype=complex)
    for i in range(0,cframes):
        IQbuffer[:,:,i] = hilbert2(crop_rf[:,:,i])*(np.exp(-2j * np.pi * fc * t2D) / np.sqrt(2))
        progress_bar_window.set_progress_value(int((i / (cframes)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()

    return IQbuffer

def USiniavg (rfdata,N):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Average of initial frames")
    progress_bar_window.show()
    [Z, X, frames] = (rfdata.shape)
    avgUS = np.zeros((Z, X, round(frames-(N-1))))
    aux = np.zeros((Z, X))
    for i in range(0, N):
        aux = aux + rfdata[:, :, i]
    avgUS[:, :, 0] = aux / N
    count = 1
    for i in range(N, frames):
        avgUS[:, :, count] = rfdata[:, :, i]
        count = count + 1
        progress_bar_window.set_progress_value(int((i/ frames) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return avgUS

def USavg(rfdata,N):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Average of frames")
    progress_bar_window.show()
    [Z, X, frames] = (rfdata.shape)
    avgUS = np.zeros((Z, X, round(frames / N)))
    i = 0
    count = 0
    while i <= (frames-N):
        aux = np.zeros((Z, X))
        j = i
        while j <= (i + (N-1)):
            aux = aux + rfdata[:,:,j]
            j = j + 1
        avgUS[:,:,count] = aux/N
        i = i + N
        count = count + 1;
        progress_bar_window.set_progress_value(int((i / (frames-N)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return avgUS

def USinterp (rfdata):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Interpolating Channels")
    progress_bar_window.show()
    [Z, X, frames] = (rfdata.shape)
    interpUS = np.zeros((Z, 2*X, frames))
    new_x = np.linspace(0, rfdata.shape[0] - 1, 1 * rfdata.shape[0])  # Double the size along the first dimension (rows)
    new_z = np.linspace(0, rfdata.shape[1] - 1, 2 * rfdata.shape[1])  # Double the size along the second dimension (columns)

    for i in range(0, frames):
        aux = rfdata[:,:,i]
        x_values = np.arange(aux.shape[0])  # Coordinates along the first dimension
        z_values = np.arange(aux.shape[1])  # Coordinates along the second dimension

        # Create a RegularGridInterpolator
        interpolator = RegularGridInterpolator((x_values, z_values), aux, method='linear', bounds_error=False,
                                       fill_value=None)

        # Create the meshgrid for evaluation
        xx, zz = np.meshgrid(new_x, new_z, indexing='ij')

        interpUS[:,:,i] = interpolator((xx, zz))
        progress_bar_window.set_progress_value(int((i/ frames) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return interpUS

#------------------------------------Displacement Map--------------------------------------------------
def saveDispChoice():
    fdialog = SaveDisp_Dialog()
    # Define variables to store ini and end values
    choice = None

    # Connect the frames_index signal to update ini and end variables
    def update_choice(c):
        nonlocal choice
        choice = c

    fdialog.saveas.connect(update_choice)

    # Execute the dialog
    fdialog.exec()
    return choice




        #----------------------------Time Filtering------------------------------------------

def integrateDisp(dispdata):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Integrating Velocity")
    progress_bar_window.show()

    [Z, X, frames] = (dispdata.shape)
    integrateDisp = np.zeros((Z, X, frames))
    for f in range(0,frames):
        integrateDisp[:,:,f] = np.sum(dispdata[:,:,0:f],axis=2)
        progress_bar_window.set_progress_value(int((f / frames) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return integrateDisp

def diffDisp(dispdata):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Diff Displacement")
    progress_bar_window.show()

    [Z, X, frames] = (dispdata.shape)
    difDisp = np.zeros((Z, X, frames-1))
    for f in range(0,frames-1):
        difDisp[:,:,f] = (dispdata[:,:,f+1]-dispdata[:,:,f])/2
        progress_bar_window.set_progress_value(int((f / (frames-1)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return difDisp

def DispLPfilter(dispdata, cutoff_freq):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Low-Pass Filter")
    progress_bar_window.show()

    [Z, X, frames] = (dispdata.shape)
    # Create a low-pass Butterworth filter
    b, a = butter(3, cutoff_freq, btype='low', analog=False)

    filtered_data = np.zeros((Z, X, frames))
    aux_data = np.zeros((frames))
    for x in range(0, X):
        for z in range(0, Z):
            aux_data = np.squeeze(dispdata[z, x, :])
            aux_data = aux_data - np.mean(aux_data)
            filtered_data[z, x, :] = lfilter(b, a, aux_data)
        progress_bar_window.set_progress_value(int((x / (X)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()

    '''filtered_data = np.zeros((Z, X, frames+2))
    # Apply the filter to the data
    aux_data = np.zeros((frames+2))
    for x in range(0, X):
        for z in range(0, Z):
            aux_data[1:-1] = np.squeeze(dispdata[z, x, :])
            filtered_data[z, x, :] = lfilter(b, a, aux_data)
        progress_bar_window.set_progress_value(int((x / (X)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    filtered_data = filtered_data[:,:,1:-1]'''

    return filtered_data

def Dispinterp (dispdata):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Interpolating Frames")
    progress_bar_window.show()
    [Z, X, frames] = (dispdata.shape)
    interpDisp = np.zeros((Z, X, 2*frames))
    frames_new = np.linspace(0, frames, 2*frames)
    frames_old = np.linspace(0, frames, frames)
    for x in range(0, X):
        for z in range(0,Z):
            aux = np.squeeze(dispdata[z,x,:])
            # Create interpolation function
            interp_func = interp1d(frames_old, aux, kind='linear')
            interpaux = interp_func(frames_new)
            interpDisp[z,x,:] = interpaux
        progress_bar_window.set_progress_value(int((x/ X) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return interpDisp

def cropFrames(dispdata):
    [Z, X, frames] = (dispdata.shape)
    fdialog = Frames_Dialog()

    fdialog.sbEnd.setMaximum(frames)
    fdialog.sbEnd.setValue(frames)

    fdialog.sbIni.setMinimum(1)
    fdialog.sbIni.setValue(1)

    # Define variables to store ini and end values
    ini = None
    end = None
    # Connect the frames_index signal to update ini and end variables
    def update_ini_end(i, e):
        nonlocal ini, end
        ini, end = (i-1, e-1)

    fdialog.frames_index.connect(update_ini_end)

    # Execute the dialog
    fdialog.exec()

    if ini == -2 and end == -2:
        return dispdata
    else:
        crop = np.zeros((Z,X,(end-ini)))
        count = 0
        for f in range(ini,end):
            crop[:,:,count] = dispdata[:,:,f]
            count = count + 1
        return crop

def Dispdetrend(dispdata):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Detrending")
    progress_bar_window.show()

    [Z, X, frames] = (dispdata.shape)
    filtered_data = np.zeros((Z, X, frames))
    aux_data = np.zeros((frames))
    for x in range(0, X):
        for z in range(0, Z):
            aux_data = np.squeeze(dispdata[z, x, :])
            filtered_data[z, x, :] = detrend(aux_data)
        progress_bar_window.set_progress_value(int((x / (X)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return filtered_data

        # ----------------------------Spatial Filtering------------------------------------------

def medianFilter(dispdata, mx, mz):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Median Filter")
    progress_bar_window.show()

    [Z, X, frames] = (dispdata.shape)
    filtered_data = np.zeros((Z, X, frames))
    kernel_size = (mz, mx)
    for f in range(0, frames):
        filtered_data[:,:,f] = medfilt(dispdata[:,:,f], kernel_size=kernel_size)
        progress_bar_window.set_progress_value(int((f / (frames)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return filtered_data

def leeFilter(dispdata, mx, mz):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Lee's Filter")
    progress_bar_window.show()

    [Z, X, frames] = (dispdata.shape)
    filtered_data = np.zeros((Z, X, frames))
    hmx = math.floor(mx/2)
    hmz = math.floor(mz/2)

    for f in range(0,frames):
        img_mean = uniform_filter(dispdata[:,:,f], (mz, mx))
        img_sqr_mean = uniform_filter(dispdata[:,:,f] ** 2, (mz, mx))
        img_variance = img_sqr_mean - img_mean ** 2

        overall_variance = variance(dispdata[:,:,f])

        #img_weights = img_variance / (img_variance + overall_variance)
        img_weights = img_variance / (img_variance + 0.5)
        filtered_data[:,:,f] = img_mean + img_weights * (dispdata[:,:,f] - img_mean)
        progress_bar_window.set_progress_value(int((f / (frames)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return filtered_data

def removeDC(dispdata):
    app = QApplication.instance()
    progress_bar_window = ProgressBarWindow()
    progress_bar_window.center_on_screen()
    progress_bar_window.set_title("Remove DC")
    progress_bar_window.show()

    [Z, X, frames] = (dispdata.shape)
    filtered_data = np.zeros((Z, X, frames))
    for f in range(0, frames):
        filtered_data[:, :, f] = dispdata[:, :, f] - np.mean(dispdata[:, :, f])
        progress_bar_window.set_progress_value(int((f / (frames)) * 100))
        progress_bar_window.repaint()
        app.processEvents()
    progress_bar_window.close_window()
    return filtered_data

#------------------------------------Time analisys--------------------------------------------------
def powerSpectrum(signal,fs):
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

def areaDisp(dispmap,prm):
    app = QApplication.instance()
    area = Disp_area(dispmap,prm)
    area.exec()
    area.close()

#------------------------------------Utilities--------------------------------------------------
def mirror_border(matrix, m_x, m_y):
    rows, cols = matrix.shape
    mirrored_matrix = np.zeros((rows + 2 * m_x, cols + 2 * m_y), dtype=matrix.dtype)
    mirrored_matrix[m_x:rows + m_x, m_y:cols + m_y] = matrix

    # Top border
    for i in range(m_x):
        mirrored_matrix[i, :] = mirrored_matrix[2 * m_x - i - 1, :]

    # Bottom border
    for i in range(m_x):
        mirrored_matrix[rows + m_x + i, :] = mirrored_matrix[rows + m_x - i - 1, :]

    # Left border
    for i in range(m_y):
        mirrored_matrix[:, i] = mirrored_matrix[:, 2 * m_y - i - 1]

    # Right border
    for i in range(m_y):
        mirrored_matrix[:, cols + m_y + i] = mirrored_matrix[:, cols + m_y - i - 1]

    return mirrored_matrix