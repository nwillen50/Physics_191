# Code for analysis of type I superconductors

# Import necessary librarys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


def type_I_analysis(sc_data, t=0.3, smooth = True):
    k_array = []
    critical_fields = []
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
        voltage_up = medfilt(up_sweep["Channel_2(V)"], kernel_size=13)
        
        # 5. Process DOWN Sweep
        t_down = down_sweep["Time(s)"]
        kGauss_down = down_sweep["Channel_1(V)"].abs()
        voltage_down = medfilt(down_sweep["Channel_2(V)"], kernel_size=13)

        # 6. Calculate dv/dt using numpy gradient
        # This calculates the derivative of voltage with respect to time
        
        v_gauss_up = gaussian_filter1d(voltage_up, sigma=10)
        v_gauss_down = gaussian_filter1d(voltage_down, sigma=10)

        if smooth != True:
            v_gauss_up = voltage_up
            v_gauss_down = voltage_down

        d1v_up = np.gradient(v_gauss_up, t_up)
        d1v_down = np.gradient(v_gauss_down, t_down)


        # Critical field up
        max_field_up_idx = np.argmax(d1v_up)
        crit_up = kGauss_up.iloc[max_field_up_idx]

        # Critical field down
        max_field_down_idx = np.argmin(d1v_down)
        crit_down = kGauss_down.iloc[max_field_down_idx]

        # Critical field
        critical_field = (crit_up + crit_down) / 2
        critical_fields.append(critical_field)

        # Temperature
        temp1_K = file_path.split('/')[-1]
        temp2_K = temp1_K.rsplit('.', 1)[0]
        temp3_K = temp2_K.split('_', 1)[1] 
        k = float(temp3_K)
        k_array.append(k)

    return k_array, critical_fields




def critical_field_model(T, H0, Tc):
    return H0 * (1 - (T/Tc)**2)


def meissner_plot(k_array, critical_fields, metal):
    params, pcov = curve_fit(critical_field_model, k_array, critical_fields)

    H0_fit, Tc_fit = params

    # 2. Extract Standard Errors (1-sigma)
    perr = np.sqrt(np.diag(pcov))
    h0_err, tc_err = perr

    T_smooth = np.linspace(0, Tc_fit, 200)

    plt.figure(figsize=(8,5))

    plt.scatter(k_array, critical_fields)

    plt.plot(
        T_smooth,
        critical_field_model(T_smooth, H0_fit, Tc_fit),
        label=f"$H_0$ = {H0_fit:.3f}±{h0_err:.3f} kG, $T_c$ = {Tc_fit:.3f}±{tc_err:.3f} K"
    )

    plt.xlabel("Temperature (K)")
    plt.ylabel("Critical Field (kG)")
    plt.title(f"Critical Field vs Temperature: {metal}")

    plt.xlim(0, Tc_fit + 0.5)
    plt.ylim(0, H0_fit + 0.1)

    plt.legend(loc="upper right")

    plt.show()

    plt.savefig(f"{metal}_meissner_effect.png")
    
    return

   



