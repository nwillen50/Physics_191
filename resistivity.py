# Graph resisitivity vs temperature for our sample

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

# Import .csv file

df = pd.read_csv("resistivity/Sample_Trial_2.csv", sep=",")

temp = df["Temp"]
resist = df["Volts(uV)"]/54.8

plt.figure()  
plt.scatter(temp, resist)
plt.xlabel("Temperature (K)")
plt.ylabel("Resistivity (Ohm)")
plt.title("Resistivity vs Temperature")

plt.show()