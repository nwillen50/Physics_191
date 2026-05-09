# Import Libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from numpy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

# single file process
 
# Graphs envelope over time
def env_graph(file):
    ''' 
    1. Download File
    2. Trim first 20 lines of files
    3. Check dataframe size
    4. Create data arrays
    5. Remove DC Offset
    6. Graph
    '''

    df = pd.read_csv(f"{file}", skiprows=20)

    time = df["TIME"]
    env = df["CH1"]

    # DC Offset
    env = env - np.median(env)

    plt.figure(figsize=(30,8))
    plt.plot(time, env)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Time vs Voltage")

    plt.show()

    return

# Graphs oscillations over time
def osc_graph(file):
    ''' 
    1. Download File
    2. Trim first 20 lines of files
    3. Check dataframe size
    4. Create data arrays
    5. Remove DC Offset
    6. Graph
    '''

    df = pd.read_csv(f"{file}", skiprows=20)

    time = df["TIME"]
    osc = df["CH2"]

    # DC Offset
    osc = osc - np.median(osc)

    plt.figure(figsize=(30,8))
    plt.plot(time, osc)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Time vs Voltage")

    plt.show()

    return

# Transforms oscillation data
def ft_graph(file):
    '''
    1. Download File
    2. Trim first 20 lines of files
    3. Create data arrays
    4. Remove DC Offset
    5. Fourier transform data
    6. Graph
    '''
    df = pd.read_csv(f"{file}", skiprows=20)

    time = df["TIME"]
    osc = df["CH2"]

    # DC offset
    osc = osc - np.median(osc)

    # Average Sampling Interval
    dt = np.mean(np.diff(time))

    # Fourier transformation of oscillations (normalized)
    ft_osc = 2 * np.abs(rfft(osc)) / len(osc)
    # Associated frequencies
    freq = rfftfreq(len(osc), dt)
    
    peaks, _ = find_peaks(ft_osc)

    top_two = peaks[np.argsort(ft_osc[peaks])[-2:][::-1]]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    ax1.plot(time, osc)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("MRI FID")
    ax1.grid()
    ax1.minorticks_on()
    ax1.grid(visible=True, which='minor', color='lightgray', linestyle='--', alpha=0.5)

    ax2.plot(freq/1000, ft_osc)
    ax2.grid()
    ax2.minorticks_on()
    ax2.grid(visible=True, which='minor', color='lightgray', linestyle='--', alpha=0.5)
    ax2.set_xlim(2e1, 3e1)
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Voltage (V)")
    ax2.set_title("MRI Fourier Transfrom")

    fig.tight_layout()
    fig.show()    

    fig.savefig(f"MRI.png", dpi=300, bbox_inches="tight")


    return freq, ft_osc, top_two




