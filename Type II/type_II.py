# Code for analysis of type II superconductors



# Import necessary librarys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

def type_II_analysis(sc_data, t=0.3, smooth=True):
    
    k_array = []
    first_H = []
    second_H = []
    # Constants
    tau = t
    cutoff = 5 * tau

    for file_path, df_unclean in sc_data.items():
        
        # 1. Separate and Clean Data
        mask = df_unclean["Time(s)"] > cutoff
        df = df_unclean[mask].copy()
        
        # 2. Find split point (Peak of Channel 1)
        max_field_idx = df["Channel_1(V)"].abs().idxmax()
        
        # 3. Split data into segments
        up_sweep = df.loc[:max_field_idx].copy()
        down_sweep = df.loc[max_field_idx:].copy()
        
        # 4. Process UP Sweep
        t_up = up_sweep["Time(s)"]
        kGauss_up = up_sweep["Channel_1(V)"].abs()
        voltage_up = medfilt(up_sweep["Channel_2(V)"].abs(), kernel_size=13)
        
        # 5. Process DOWN Sweep
        t_down = down_sweep["Time(s)"]
        kGauss_down = down_sweep["Channel_1(V)"].abs()
        voltage_down = medfilt(down_sweep["Channel_2(V)"].abs(), kernel_size=13)



        # 6. Calculate dv/dt using numpy gradient
        # This calculates the derivative of voltage with respect to time

        window = 201   # must be odd
        poly = 3      # polynomial order

        v_sav_up = savgol_filter(voltage_up, window_length=window, polyorder=poly)
        d1v_up = savgol_filter(voltage_up, window_length=window, polyorder=poly, mode='mirror', deriv=1)
        d2v_up = savgol_filter(voltage_up, window_length=window, polyorder=poly,mode='mirror', deriv=2)

        v_sav_down = savgol_filter(voltage_down, window_length=window, polyorder=poly)
        d1v_down = savgol_filter(voltage_down, window_length=window, polyorder=poly, mode='mirror', deriv=1)
        d2v_down = savgol_filter(voltage_down, window_length=window, polyorder=poly, mode='mirror', deriv=2)

        # First crticial field up
        max_field_up_idx = np.argmax(d2v_up)
        one_crit_up = kGauss_up.iloc[max_field_up_idx]

        # Critical field down
        max_field_down_idx = np.argmax(d2v_down)
        one_crit_down = kGauss_down.iloc[max_field_down_idx]

        # Critical field
        one_critical_field = (one_crit_up + one_crit_down) / 2
        first_H.append(one_critical_field)

        # Second crticial field up
        max_field_up_idx = np.argmin(d2v_up)
        two_crit_up = kGauss_up.iloc[max_field_up_idx]

        # Critical field down
        max_field_down_idx = np.argmin(d2v_down)
        two_crit_down = kGauss_down.iloc[max_field_down_idx]

        # Critical field
        two_critical_field = (two_crit_up + two_crit_down) / 2
        second_H.append(two_critical_field)

        # Temperature
        temp1_K = file_path.split('/')[-1]
        temp2_K = temp1_K.rsplit('.', 1)[0]
        temp3_K = temp2_K.split('_', 1)[1] 
        k = float(temp3_K)
        k_array.append(k)

    return k_array, first_H, second_H



# 1. The Global Model: Shares Tc, allows different H0 values
def critical_field_model(combined_T, h01, h02, tc):
    # Split the concatenated T-array back into two equal parts
    n = len(combined_T) // 2
    t1 = combined_T[:n]
    t2 = combined_T[n:]
    
    # Calculate Hc1 and Hc2 using the shared Tc
    hc1 = h01 * (1 - (t1 / tc)**2)
    hc2 = h02 * (1 - (t2 / tc)**2)
    
    return np.concatenate([hc1, hc2])

# 2. Your plotting function
def meissner_plot(k_array, first_H, second_H, metal):
    # 1. Prepare data and perform the Global Fit
    T_combined = np.concatenate([k_array, k_array])
    H_combined = np.concatenate([first_H, second_H])
    p0 = [max(first_H), max(second_H), max(k_array)]
    
    # Run fit once and capture both parameters (popt) and covariance (pcov)
    popt, pcov = curve_fit(critical_field_model, T_combined, H_combined, p0=p0)
    h01_fit, h02_fit, tc_singular = popt
    
    # 2. Extract Standard Errors (1-sigma)
    perr = np.sqrt(np.diag(pcov))
    h01_err, h02_err, tc_err = perr

    # --- Plot 1: The Fit ---
    T_smooth = np.linspace(0, tc_singular, 200)
    plt.figure(figsize=(8, 5))
    plt.scatter(k_array, first_H, color="blue", alpha=0.6)
    plt.scatter(k_array, second_H, color="red", alpha=0.6)

    # Use f-strings to include errors in the legend
    plt.plot(T_smooth, h01_fit * (1 - (T_smooth/tc_singular)**2), color="blue", 
             label=f"$H_1$: {h01_fit:.2f}±{h01_err:.3f} kG")
    plt.plot(T_smooth, h02_fit * (1 - (T_smooth/tc_singular)**2), color="red", 
             label=f"$H_2$: {h02_fit:.3f}±{h02_err:.3f} kG")
    plt.plot([], [], ' ', label=f"$T_c$: {tc_singular:.2f}±{tc_err:.3f} K")

    plt.title(f"Critical Field vs Temperature: {metal}")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Critical Field (kG)")
    plt.legend()

    plt.xlim(0, tc_singular + 0.5)
    plt.ylim(0, h02_fit + 0.1)

    plt.show()

    return


