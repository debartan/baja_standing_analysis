# simple analysis of baja standings

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

main_dir = os.path.dirname(__file__)
rel_path = "Data/Tennessee.csv"
file_path = os.path.join(main_dir, rel_path)

data = np.genfromtxt(file_path, delimiter=',', dtype=float, skip_header=1,
        filling_values=0)

data = data.T

f_cost = signal.savgol_filter(data[4],50,4)
cost = plt.subplot(3,3,1)
cost.plot(data[4])
cost.title.set_text("COST")
cost.plot(f_cost)

f_design = signal.savgol_filter(data[5],50,4)
design = plt.subplot(3,3,2)
design.plot(data[5])
design.title.set_text("DESIGN")
design.plot(f_design)

f_sales = signal.savgol_filter(data[6],50,4)
sales = plt.subplot(3,3,3)
sales.plot(data[6])
sales.title.set_text("SALES")
sales.plot(f_sales)

f_accel = signal.savgol_filter(data[7],50,4)
accel = plt.subplot(3,3,4)
accel.plot(data[7])
accel.title.set_text("accel")
accel.plot(f_accel)

f_man = signal.savgol_filter(data[8],50,4)
man = plt.subplot(3,3,5)
man.plot(data[8])
man.title.set_text("MANEUV")
man.plot(f_man)

f_sled = signal.savgol_filter(data[9],50,4)
sled = plt.subplot(3,3,6)
sled.plot(data[9])
sled.title.set_text("SLED")
sled.plot(f_sled)

f_susp = signal.savgol_filter(data[10],50,4)
susp = plt.subplot(3,3,7)
susp.plot(data[10])
susp.title.set_text("S&T")
susp.plot(f_susp)

f_end = signal.savgol_filter(data[11],50,4)
end = plt.subplot(3,3,8)
end.plot(data[11])
end.title.set_text("ENDURANCE")
end.plot(f_end)


plt.show()

print("done")
