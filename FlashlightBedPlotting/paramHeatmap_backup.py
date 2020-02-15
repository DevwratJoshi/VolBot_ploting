import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import math
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import sys
import math
box_bottom = 1800
mean_box_height = 10.0*1000/11.0
if len(sys.argv) < 5:
    print("Usage: plotPosChange.py {R/B/A (First parameter to change) } {R/B/A (Second parameter to change)} {Value of parameter to keep constant} {X or Y (depending on which coordinate to change)}")
    sys.exit()

#coord indicates the coordinate to use. default 0 (x-coordinate). Change to 1 for y coordinate
coord = 0
if sys.argv[4] == 'Y' or sys.argv[4] == 'y':
    coord = 1

big = 60*2
small = 20*2
mid = 30*2
R_list = ['120', '180', '240','300']
B_list = [str(int(i)) for i in np.linspace(20,180,9)]
A_list = [str(int(i)) for i in np.linspace(10,90,9)]
data_list = [str(int(i)) for i in np.linspace(1,10,10)]
freq = '20'
amp = '12'
xPosInit = np.zeros((101,)) # First element is initial element. Followed by last 100 element
fig,ax = plt.subplots(1,1, figsize=(10,10))

class MidpointNormalize(colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

if 'A' in sys.argv and 'B' in sys.argv:
    list1 = A_list
    list2 = B_list
    xData = np.zeros((len(list1), len(list2))) # The array with column 1 storing mean and column 2 storing std.dev of change in x coordinate
    R = sys.argv[3]
    if not R in R_list:
        print("This R is unavailable")
        sys.exit()
    
    for i in range(len(list2)):
        for j in range(len(list1)):
            for d in data_list:
                url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flashlight_Bed_sims/Freq" + freq + "Amp" + amp + "/" + "data_R_alpha_beta/R" + R  + "/Beta" + list2[i] + "/alpha_fold" + list1[j] + "/data" + d + "/Flash_" + R  + ".0_" + list2[i] + "_" + list1[j]
                data = pd.read_csv(url, header=None, index_col=False)
                if not coord:
                    xPosInit[0] = xPosInit[0] + data.values[0][0]
                    xPosInit[1:] = xPosInit[1:] + data.values[-101:-1, 0]
                else:
                    xPosInit[0] = xPosInit[0] + (mean_box_height -data.values[0][1])
                    xPosInit[1:] = xPosInit[1:] + (mean_box_height - data.values[-101:-1, 1])
            xPosInit = xPosInit/len(data_list)
            x_initial = xPosInit[0]
            x_final = sum(xPosInit[1:])/len(xPosInit[1:])
            xData[j][i] = (x_final - x_initial)/big
            xPosSdev = []
    

    plt.rcParams.update({'font.size':20})
    ax.set_xlabel("Angle  of the flashlight with the horizontal", fontsize='large')
    ax.set_ylabel("Angle width of the flashlight", fontsize='large')
    ax.set_xticklabels([str(int(i)-90) for i in list1])
    ax.set_yticklabels([int(i) for i in list2])

elif 'A' in sys.argv and 'R' in sys.argv:
    list1 = A_list
    list2 = R_list
    xData = np.zeros((len(list1), len(list2))) # The array with column 1 storing mean and column 2 storing std.dev of change in x coordinate
    beta = sys.argv[3]
    if not beta in B_list:
        print("This R is unavailable")
        sys.exit()
    
    for i in range(len(list2)):
        for j in range(len(list1)):
            for d in data_list:
                url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flashlight_Bed_sims/Freq" + freq + "Amp" + amp + "/" + "data_R_alpha_beta/R" + list2[i]  + "/Beta" + beta + "/alpha_fold" + list1[j] + "/data" + d + "/Flash_" + list2[i]  + ".0_" + beta + "_" + list1[j]
                data = pd.read_csv(url, header=None, index_col=False)
                if not coord:
                    xPosInit[0] = xPosInit[0] + data.values[0][0]
                    xPosInit[1:] = xPosInit[1:] + data.values[-101:-1, 0]
                else:
                    xPosInit[0] = xPosInit[0] + (mean_box_height -data.values[0][1])
                    xPosInit[1:] = xPosInit[1:] + (mean_box_height - data.values[-101:-1, 1])

            xPosInit = xPosInit/len(data_list)
            x_initial = xPosInit[0]
            x_final = sum(xPosInit[1:])/len(xPosInit[1:])
            xData[j][i] = (x_final - x_initial)/big
            xPosSdev = []

    ax.set_xlabel("Angle of the flashlight with the horizontal", fontsize='xx-large')
    ax.set_ylabel("Range of the flashlight [module lengths]", fontsize='xx-large')
    ax.set_xticklabels([str(int(i)-90) for i in list1])
    ax.set_yticklabels([str((int(i)-mid)/mid) for i in list2])

elif 'B' in sys.argv and 'R' in sys.argv:
    list1 = B_list
    list2 = R_list
    xData = np.zeros((len(list1), len(list2))) # The array with column 1 storing mean and column 2 storing std.dev of change in x coordinate
    alpha = sys.argv[3]
    if not alpha in A_list:
        print("This alpha  is unavailable")
        sys.exit()
    
    for i in range(len(list2)):
        for j in range(len(list1)):
            for d in data_list:
                url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flashlight_Bed_sims/Freq" + freq + "Amp" + amp + "/" +"data_R_alpha_beta/R" + list2[i]  + "/Beta" + list1[j] + "/alpha_fold" + alpha + "/data" + d + "/Flash_" + list2[i]  + ".0_" + list1[j] + "_" + alpha
                data = pd.read_csv(url, header=None, index_col=False)
                if not coord:
                    xPosInit[0] = xPosInit[0] + data.values[0][0]
                    xPosInit[1:] = xPosInit[1:] + data.values[-101:-1, 0]
                else:
                    xPosInit[0] = xPosInit[0] + (mean_box_height -data.values[0][1])
                    xPosInit[1:] = xPosInit[1:] + (mean_box_height - data.values[-101:-1, 1])

            xPosInit = xPosInit/len(data_list)
            x_initial = xPosInit[0]
            x_final = sum(xPosInit[1:])/len(xPosInit[1:])
            xData[j][i] = (x_final - x_initial)/big
            xPosSdev = []
    plt.rcparams.update({'font.size':20})
    ax.set_xlabel("Angle width of the flashlight")
    ax.set_ylabel("Range of the flashlight [module lengths]", fontsize='xx-large')
    ax.set_xticklabels([str(int(i)) for i in list1])
    ax.set_yticklabels([str((int(i)-mid)/mid) for i in list2])

plt.rcParams.update({'font.size': 22})
if not coord:
    levels = MaxNLocator(nbins=10).tick_values(-3, 6)

else :
    levels = MaxNLocator(nbins= 10).tick_values(-3, 6)

cmap = plt.get_cmap('RdBu')
if not coord:
    im = ax.pcolormesh(list1 + ['0'], list2 + ['0'], np.transpose(xData), cmap=cmap, norm=MidpointNormalize(-3, 6, 0.))
    cb = fig.colorbar(im, ax=ax)
    cb.set_ticks(np.arange(-3, 7, 1))
    ax.set_title("Migration along X axis")

else:
    im = ax.pcolormesh(list1 + ['0'], list2 + ['0'], np.transpose(xData), cmap=cmap, norm=MidpointNormalize(-3, 6, 0.))
    cb = fig.colorbar(im, ax=ax)
    cb.set_ticks(np.arange(-3, 7, 1))
    ax.set_title("Migration along Y axis")

cb.ax.tick_params(labelsize=16)
#ax.quiver(x,y,trueX_norm, trueY_norm, pivot='mid')

#ax.set_title("freq {} amp {} up {} down {}".format(freq, amp, up, down))
ax.set_xticks(0.5 +  np.arange(len(list1)))
ax.set_yticks(0.5 + np.arange(len(list2)))
ax.tick_params(labelsize=18)
#xu = float(up) *(20.0/(10.0))
#yu = 20
#xd = float(down)*(20.0/10.0)
#yd = 0


plt.show()
