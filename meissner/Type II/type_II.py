# Code for analysis of type II superconductors

# Import necessary librarys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.stats import t

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

# Critical Field Model
def critical_field_model(T, H0, Tc):
    return H0 * (1- (T/Tc)**2)

# Produces single Tc, unlike optimizing each field separately
def solo_Tc_critical_field_model(combined_T, h01, h02, tc):
    # Split the concatenated T-array back into two equal parts
    n = len(combined_T) // 2
    t1 = combined_T[:n]
    t2 = combined_T[n:]
    
    # Calculate Hc1 and Hc2 using the shared Tc
    hc1 = critical_field_model(t1, h01, tc)
    hc2 = critical_field_model(t2, h02, tc)
    
    # Need to output data together to satisfy optimizer; trick of the function
    return np.concatenate([hc1, hc2])

# Data Confidence Intervals
def compute_confidence_band(T, model, params, pcov):

    # Curve fitted meissner model parameters (critical field & tempurature)
    H0, Tc = params
    
    # Array of predicted critical fields in (0, T_c)
    H = model(T, H0, Tc)
    
    # Linear approximating of the uncertainty in H

    # Derivative of H wrt H0
    dH_dH0 = 1 - (T/Tc)**2

    # Derivative of H wrt Tc
    dH_dTc = H0 * (2 * T**2) / Tc**3
    
    # Create Jacobian Matrix: rows: T points, columns: the derivatives
    J = np.vstack((dH_dH0, dH_dTc)).T
    
    # Variance propagation

    # Compute the variance in H
    # Computes the variance at each point T at the same time 
    # H_var= (dH/dH0)^2Var(H0) + (dH/dTc)^2Var(Tc) + 2(dH/dHO)(dH/dTc)Cov(H0,Tc)
    H_var = np.sum(J @ pcov * J, axis=1)

    # Compute the standard deviation in H
    H_std = np.sqrt(H_var)
    
    # Degrees of freedom (data points - parameters (2))
    # dof = len(T) - len(params)

    # Student T distribution, 95% confidence interval is greater than in a typical gaussian as data amount is small
    # tval = t.ppf(1 - alpha/2, dof)
    
    tval = 2

    # Error range
    delta = tval * H_std
    
    return H, H - delta, H + delta

# Plot meissner graphs of critical field vs temperature
def meissner_plot(k_array, first_H, second_H, metal):
    # Combine data and provide initial guess for H_01, H_02, Tc
    T_combined = np.concatenate([k_array, k_array])
    H_combined = np.concatenate([first_H, second_H])
    p0 = [max(first_H), max(second_H), max(k_array)]

    # Run curve fitting; Outputs: parameters
    params, pcov = curve_fit(solo_Tc_critical_field_model, T_combined, H_combined, p0=p0)

    # Error range (2 sigma = 95%)
    n_sigma = 2

    # Critical temperature parameters from our fitted model
    Tc = params[2] 
    Tc_err = n_sigma * np.sqrt(pcov[2,2])

    # First critical field parameters from our fitted model
    H01 = params[0]
    H01_err = n_sigma * np.sqrt(pcov[0, 0])
    H01_Tc_params = np.array([params[0], Tc])
    H01_Tc_pcov = pcov[np.ix_([0, 2], [0, 2])] # Hc1 & Tc covariance matrix

    # Second critical field parameters from our fitted model
    H02 = params[1]
    H02_err = n_sigma * np.sqrt(pcov[1, 1])
    H02_Tc_params = np.array([params[1], Tc])
    H02_Tc_pcov = pcov[np.ix_([1, 2], [1, 2])] # Hc2 & Tc covariance matrix

    T_smooth = np.linspace(0, Tc, 200)

    # First critical field confidence band
    H1_model, H1_lower, H1_upper = compute_confidence_band(
        T_smooth,
        critical_field_model,
        H01_Tc_params,
        H01_Tc_pcov
    )

    # Second critical field confidence band
    H2_model, H2_lower, H2_upper = compute_confidence_band(
        T_smooth,
        critical_field_model,
        H02_Tc_params,
        H02_Tc_pcov
    )

        # Create and size figure
    fig = plt.figure(figsize=(8,5))

    # Plot data
    plt.scatter(k_array, first_H, color="blue", alpha=0.6)
    plt.scatter(k_array, second_H, color="red", alpha=0.6)

    # Plot model for first critical field
    plt.plot(
        T_smooth,
        H1_model,
        color="blue",
        label=f"$H_1$ = {H01:.3f}±{H01_err:.3f} kG"
    )

    # Confidence band shading for first critical field
    plt.fill_between(T_smooth, H1_lower, H1_upper, color="blue", alpha=0.3)

    # Plot model for second critical field
    plt.plot(
        T_smooth,
        H2_model,
        color="red",
        label=f"$H_2$ = {H02:.3f}±{H02_err:.3f} kG"
    )

    # Confidence band shading for first critical field
    plt.fill_between(T_smooth, H2_lower, H2_upper, color="red", alpha=0.3)

    # Add T_c label
    plt.plot([], [], ' ', label=f"$T_c$: {Tc:.2f}±{Tc_err:.3f} K")

    plt.xlabel("Temperature (K)")
    plt.ylabel("Critical Field (kG)")
    plt.title(f"Critical Field vs Temperature: {metal}")

    plt.xlim(0, Tc + 0.5)
    plt.ylim(0, H02 + 0.2)

    plt.legend(loc="upper right")
    
    fig.savefig(f"{metal}_meissner_effect.png", dpi=300, bbox_inches="tight")

    plt.show()

    return

    