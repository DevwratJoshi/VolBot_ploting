#This program is meant to calculate the required number of bed robots and their weight for a given bed size

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


#In theory, should be between 0.59 and 1.02
big_diameter = 2 #diameter of the large module in small module diameter units 
bed_height = 10.0 #Expressed in small module diameter 
bed_width = 40.0 #Expressed in small module diameter


N = np.linspace(50, 250, 11)
fraction = np.linspace(0, 1, 11)
C = np.zeros((fraction.shape[0], N.shape[0])) #rhos on x axis, diameters on y axis

for y in range(fraction.shape[0]):
    for x in range(N.shape[0]):
        C[y][x] = round(((N[x]*math.pi)/(bed_height * bed_width*4))*(fraction[y]*((big_diameter*big_diameter) - 1) + 1) , 2)

print(C)
x_ticks_list = [i for i in np.arange(0, N.shape[0], 1)]
y_ticks_list = [i for i in np.arange(0, fraction.shape[0], 1)]

fig, ax1 = plt.subplots(figsize=(10,10))

ax1.set_xticks(x_ticks_list)
ax1.set_yticks(y_ticks_list)
ax1.set_xlabel("Number of robots", fontsize=16)
ax1.set_ylabel("Fraction of large robots", fontsize=16)
fs = slice(0, fraction.shape[0], 1)
ns = slice(0, N.shape[0], 1)

ax1.set_yticklabels([str(round(i,2)) for i in fraction[fs]])
ax1.set_xticklabels([str(int(i)) for i in N[ns]])
ax1.tick_params(labelsize=14.0)

ax1.set_title("Packing fraction of the bed for \n bed_height = {}, bed_width = {}".format(bed_height, bed_width), fontsize='xx-large')
for y in range(fraction.shape[0]):
    for x in range(N.shape[0]):
        ax1.text(x, y, C[y][x], ha="center", va="center", color="w", fontsize=12)
im1 = ax1.imshow(C, cmap=plt.get_cmap('RdBu_r'), interpolation=None, origin='lower', norm=MidpointNormalize(0.35, 1, 0.8))
cb1 = fig.colorbar(im1, ax=ax1)
plt.show()

