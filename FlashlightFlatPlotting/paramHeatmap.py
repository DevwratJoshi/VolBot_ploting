import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import math
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import sys
import math
box_bottom = 1800

if len(sys.argv) < 4:
    print("Usage: plotPosChange.py {R/B/A (The parameter to change)} {value of second param in order RBA} {value of third param in RBA}")
    sys.exit()
big = 60*2
small = 20*2
mid = 30*2
R_list = ['120', '180', '240','300']
B_list = [str(int(i)) for i in np.linspace(20,180,9)]
A_list = [str(int(i)) for i in np.linspace(10,90,9)]
data_list = [str(int(i)) for i in np.linspace(1,10,10)]

xPosInit = np.zeros((101,)) # First element is initial element. Followed by last 100 element
fig,ax = plt.subplots(1,1, figsize=(10,10))

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
                url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flashlight_Flat_sims/data_R_alpha_beta/R" + R  + "/Beta" + list2[i] + "/alpha_fold" + list1[j] + "/data" + d + "/Flash_flat_" + R  + ".0_" + list2[i] + "_" + list1[j]
                data = pd.read_csv(url, header=None, index_col=False)
                xPosInit[0] = xPosInit[0] + data.values[0][0]
                xPosInit[1:] = xPosInit[1:] + data.values[699:-1, 0]

            xPosInit = xPosInit/len(data_list)
            x_initial = xPosInit[0]
            x_final = sum(xPosInit[1:])/len(xPosInit[1:])
            xData[j][i] = (x_final - x_initial)/mid
            xPosSdev = []
    

    ax.set_xlabel("Angle of the flashlight with the horizontal", fontsize='xx-large')
    ax.set_ylabel("Angle width of the flashlight", fontsize='xx-large')
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
                url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flashlight_Flat_sims/data_R_alpha_beta/R" + list2[i]  + "/Beta" + beta + "/alpha_fold" + list1[j] + "/data" + d + "/Flash_flat_" + list2[i]  + ".0_" + beta + "_" + list1[j]
                data = pd.read_csv(url, header=None, index_col=False)
                xPosInit[0] = xPosInit[0] + data.values[0][0]
                xPosInit[1:] = xPosInit[1:] + data.values[699:-1, 0]

            xPosInit = xPosInit/len(data_list)
            x_initial = xPosInit[0]
            x_final = sum(xPosInit[1:])/len(xPosInit[1:])
            xData[j][i] = (x_final - x_initial)/mid
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
                url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flashlight_Flat_sims/data_R_alpha_beta/R" + list2[i]  + "/Beta" + list1[j] + "/alpha_fold" + alpha + "/data" + d + "/Flash_flat_" + list2[i]  + ".0_" + list1[j] + "_" + alpha
                data = pd.read_csv(url, header=None, index_col=False)
                xPosInit[0] = xPosInit[0] + data.values[0][0]
                xPosInit[1:] = xPosInit[1:] + data.values[699:-1, 0]

            xPosInit = xPosInit/len(data_list)
            x_initial = xPosInit[0]
            x_final = sum(xPosInit[1:])/len(xPosInit[1:])
            xData[j][i] = (x_final - x_initial)/mid
            xPosSdev = []

    ax.set_xlabel("Angle width of the flashlight", fontsize='xx-large')
    ax.set_ylabel("Range of the flashlight [module lengths]", fontsize='xx-large')
    ax.set_xticklabels([str(int(i)) for i in list1])
    ax.set_yticklabels([str((int(i)-mid)/mid) for i in list2])



levels = MaxNLocator(nbins='auto').tick_values(0, 30)
cmap = plt.get_cmap('RdBu')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

im = ax.pcolormesh(list1 + ['0'], list2 + ['0'], np.transpose(xData), cmap=cmap, norm=norm)
cb = fig.colorbar(im, ax=ax)
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



#plt.plot([xu, xd], [yu, yd], color=(makeColourTuple(112, 48, 160)), linewidth=2)
plt.show()
