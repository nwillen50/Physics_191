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


# Estimate initial parameters for our phonon oscillations

def osc_initial_params(time, voltage, t0, tau0):
    ''' 
    2. Guess w1, w2
    3. Guess A01, A02
    4. Guess tau01, tau02
    5. Set bounds and initial guess
    '''


    N = len(time)
    dt_samp = float(np.median(np.diff(time)))
    span = time[-1] - time[0]


    # w1 and w2
    
    # fourier transform data
    spec = np.abs(np.fft.rfft(voltage - voltage.mean()))
    freqs = np.fft.rfftfreq(N, d=dt_samp)

    # find 2 largest peaks, order them, and convert to angular frequency
    peaks, _ = find_peaks(spec)
    top2 = peaks[np.argsort(spec[peaks])[-2:]]  # indices of the 2 tallest peaks
    f_sorted = np.sort(freqs[top2])[::-1] # fast first
    f1, f2 = float(f_sorted[0]), float(f_sorted[1])
    w1, w2 = 2*np.pi*f1, 2*np.pi*f2

    # A01, A02
    post = voltage[time >= t0]
    A_total = (post.max() - post.min()) / 2
    A1_guess, A2_guess = 0.7 * A_total, 0.3 * A_total

    # tau01, tau02
    post_span = max(span - (t0 - time[0]), dt_samp * 10)
    tau1_guess = post_span / 2
    tau2_guess = post_span

    # tau0
    tau0_guess = tau0


    # p0
    p0 = [tau0_guess,
          A1_guess, tau1_guess, w1, 0.0,
          A2_guess, tau2_guess, w2, 0.0]

    A_max = 5 * A_total
    low = [dt_samp/2, 0, dt_samp,      0.5*w1, -50,
          0,        dt_samp,      0.5*w2, -50]
    high = [post_span/4, A_max,    5*post_span,  1.5*w1,  50,
          A_max,    20*post_span, 1.5*w2,  50]
    
    bounds = (low, high)

    return p0, bounds



def env_model(t, t0, tau0, tau1, A0, A1):
    return (1/2) * (1 + erf((t-t0)/tau0)) * (A0 * np.exp(-(t-t0)/tau1) + A1)



def regression(metal, power, baseline):
    '''
    1. Curve Fit Envelope
    2. Remove Envelope from data
    3a. View Oscillations?
    3. View fourier transformed oscillations
    4. Curve Fit Oscillations
    5. Oscillation Residuals?
    6. Produce final model
    7. Graph final model against data
    '''
    # process file 
    voltage_np, time_np = process(metal, power, baseline)

    t0 = 139.18162010600014
    time_np = time_np-t0

    # t0
    peak_idx = np.argmin(voltage_np)
    t0_guess = time_np[peak_idx]

    # tau0

    # voltage range
    v_range = voltage_np.max() - voltage_np.min()

    # excitation amplitude
    amplitude = baseline - voltage_np.min()  # positive number

    p0 = [t0_guess, 0.3, 5.0, -amplitude, 0.1 * v_range]

    bounds = (
        [t0_guess - 0.5, 0.05, 0.1,  -5 * amplitude,  -2 * v_range],
        [t0_guess + 0.5, 3.0, 50, 0, 2 * v_range],
    )

    popt1, _ = curve_fit(env_model, time_np, voltage_np, p0=p0, bounds=bounds, maxfev=50000)
    t0, tau0, tau1, A0, A1 = popt1

    t_smooth = np.linspace(np.min(time_np), np.max(time_np), 200)


    osc = voltage_np-env_model(time_np, *popt1)

    def osc_model(t, tau0, A01, tau01, w01, phi01, A02, tau02, w02, phi02):
        dt = t - t0
        onset = (1/2) * (1 + erf(dt / tau0))
        
        exp1 = np.exp(-np.clip(dt, 0, None) / tau01) 
        exp2 = np.exp(-np.clip(dt, 0, None) / tau02)
        # Clip prevents errors from this exponential going to infinity

        osc1 = np.cos(w01*t + phi01)
        osc2 = np.cos(w02*t + phi02)
        
        return onset * (A01 * exp1 * osc1 + A02 * exp2 * osc2)

    # Guess and Bounds
    p0, bounds = osc_initial_params(time_np[20:], osc[20:], t0, tau0)

    # Curve Fit
    popt2, _ = curve_fit(osc_model, time_np[20:], osc[20:], p0=p0, bounds=bounds, maxfev=80000)

    resid = voltage_np - (env_model(t_smooth, *popt1) + osc_model(t_smooth, *popt2))
    R2 = 1 - np.sum(resid**2) / np.sum((osc - osc.mean())**2)

    return time_np, env_model(t_smooth, *popt1) + osc_model(t_smooth, *popt2)


