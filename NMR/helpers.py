# Import Libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from numpy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import erf

# single file process
 
def process(metal, power, b_line):
    ''' 
    1. Download File
    2. Define Constants
    3. Pull Columns from File as Arrays
    4. Average Trial Runs
    5. Normalize and Index Data
    '''
    
    # Download File

    col_names = ['Position(mm)', 'Voltage(V)', 'Phase(Theta)'] # Assign custom names

    df = pd.read_csv(f"{metal}/data/{power}.txt", sep=r"\s+", header=None, names=col_names)

    ## Define Constants

    # speed of light
    c = 299792458

    # Reflectivity (Delta V)
    voltage = -1
    if metal == "bismuth":
        voltage = 80.15e-3
    if metal == "antimony":
        voltage = 1


    # Trial length (bismuth: 81, antimony: 161)
    trial_len = -1 
    if metal == "bismuth":
        trial_len = 81
    if metal == "antimony":
        trial_len = 161

    # Baseline length
    baseline_num = b_line

    # Create Arrays
        
    position = df["Position(mm)"]
    delta_v = df["Voltage(V)"] 
    time = 2 * (position / 10**3) / c
    if metal == "antimony":
        # Convert voltage from volts to microvolts (10^-6)
        delta_v = delta_v * 10**6

    # Convert time from seconds to picoseconds (10^-12)
    time = time * 10**12

    ## Average Trials

    # Separate and average trial runs for each power and each sample

    delta_v = delta_v.groupby(delta_v.index % trial_len).mean().reset_index(drop=True)
    time = time[ : trial_len]


    time = time.to_numpy()
    delta_v = delta_v.to_numpy()

    # Index time
    ''' t0_idx = np.argmin(delta_v)
    t0 = time[t0_idx]
    time -= t0 '''

    # Normalize y-axis
    baseline = delta_v[:baseline_num+1].mean()
    delta_v = delta_v - baseline
    
    if metal == "bismuth":
        voltage = (delta_v / voltage) * 10**3
    else:
        voltage = delta_v

    return voltage, time
