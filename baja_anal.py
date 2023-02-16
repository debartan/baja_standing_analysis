# simple analysis of baja standings

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

comp_name = "Tennessee"

main_dir = os.path.dirname(__file__)
rel_path = "Data/"+comp_name+".csv"
file_path = os.path.join(main_dir, rel_path)

# mashing data to fit form I want
data = np.genfromtxt(file_path, delimiter=',',filling_values=0,dtype=None,
        encoding=None)
event_names = data[0]
data = data[1:]
data = data.astype('float64')
data = data.T

car_number = 94

# s-golay parameters
f_window = 75
f_order = 8

# determining plot grid size pre-determined width
width = 4
size = len(event_names)-2 # skipping first two cols
if size%width==0:
    height = size/width
else:
    height = size//width+1


for index, event in enumerate(event_names):
    if index == 0 or index == 1:
        continue; # want to skip first two data cols

    # sorting and filtering data for event
    sort_y = np.sort(data[index])
    sort_y = sort_y[::-1]
    filt_y = signal.savgol_filter(sort_y,f_window,f_order)

    # finding position our car number was in event
    sort_index = data[index].argsort()
    cars_sorted = data[1,sort_index[::-1]]
    event_pos = np.where(cars_sorted == car_number)

    # plotting
    end = plt.subplot(height,width,index-1) # -1 to account for skipped
    end.plot(sort_y)
    end.plot(filt_y)
    end.axvline(x=event_pos[0], color="red")
    end.title.set_text(event)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

print("done")
