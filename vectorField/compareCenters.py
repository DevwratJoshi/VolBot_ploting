#This program will take the center data collected from the VolBot_Flow sims and display the positions of the centers of the flow for particular angles
import sys
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import math
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import collections as mc
def makeColourTuple(*ctup): #Makes a colour tuple using rgb values out of 255
    return tuple([round(float(i/255.0), 2) if i < 255 else 255.0 for i in ctup])

url ="/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flow_Sims/PIV_data_constant_top_4/middle/sections_max20/centers"

data = pd.read_csv(url, index_col=False)

fig,ax = plt.subplots(1, 1, figsize=(12,12))
c = [(float(j)/5,0.,1-float(j/5.0))  for j  in [0, 1, 2, 3, 4,5]]

print(c)
#ax.scatter(data['center_x'].values[:5], data['center_y'].values[:5],c=c[:5], marker='o')
ax.scatter(data['center_x'].values[6:], data['center_y'].values[6:],c=c[1:], marker='o', s=64)


lines = []

for i in range(5,11):
    lines.append([(10, 20), (i*2, 0)])
c[0] = (0,0,0)
lc = mc.LineCollection(lines, colors=c, linewidths=2)

ax.add_collection(lc)
ax.set_title("Center positions for different angles")
ax.set_xlabel("X position", fontsize='xx-large')
ax.set_ylabel("Y position", fontsize='xx-large')
ax.set_xticks(np.arange(0, 20+1, 1))
ax.set_yticks(np.arange(0, 20+1, 1))
ax.tick_params(labelsize=16)

plt.show()
