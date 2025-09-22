# pyBeamformer

# How to Install

1.    Install Python 3.12.3 from https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe.

2.    Install VSCode (or your preferred IDE) from https://code.visualstudio.com/download.

3.    Download the project (Download ZIP) and extract it.

4.    In VSCode, go to File > Open Folder and select the pyBeamformer-main folder.

5.    Click Terminal > New Terminal and create a virtual environment: python -m venv .venv

6.    Activate the virtual environment: .venv\Scripts\activate

7.    Install the required modules: pip install -r requirements.txt

8.    Run the pyBeamformer.py file.
    

# How to use
1. Download an example pre-beamformed file

    Source: https://drive.google.com/drive/folders/1aR4OlLSaYJPKgbIBnFYGUApsIQnY-BsJ?usp=sharing

    Data Citation: This data is from Uliana, J. H., Sampaio, D. R. T., Fernandes, G. S. P., Brassesco, M. S., Nogueira-Barbosa, M. H., Carneiro, A. A. O., & Pavan, T. Z. (2020). Multiangle long-axis lateral illumination photoacoustic imaging using linear array transducer. Sensors (Switzerland), 20(14), 1–19. https://doi.org/10.3390/s20144052

    Data Description: The examples represent a B-mode image and a Photoacoustic (PA) image of a human index finger.

2. Load Data: Click File > Load .mat file

3. Data Display: The pre-beamformed data will be shown in the left axis

4. Configure Parameters: Select appropriate ultrasound parameters and image modality (conventional ultrasound or photoacoustic)

5. Choose Beamforming Method: Select your preferred beamforming algorithm

6. Preview (Optional): Click "Preview" to beamform and display only the current frame

7. Process All Frames: Click "Run" to beamform all frames (this may be slow for large datasets)

8. Save Results: Use the "Save" button to export the beamformed data as a .mat file

    
