# Graph resisitivity vs temperature for our sample

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

# Import .csv file

df = pd.read_csv("Trial Data/Sample_Trial_5.csv", sep=",")

temp = df["Temp"]
resist = df["Volts(uV)"]/54.8

fig = plt.figure()  
plt.scatter(temp, resist)
plt.xlabel("Temperature (K)")
plt.ylabel("Resistivity (Ohm)")
plt.title("Resistivity vs Temperature")

fig.savefig(f"resist_temp_trial_5.pdf", dpi=300, bbox_inches="tight")

plt.show()