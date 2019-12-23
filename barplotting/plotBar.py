import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys

def sdev(data, mode=0):
    u = sum(data)
    u = u/len(data)
    #u contains the mean of data here
    s = sum((x-u)**2 for x in data)
    #print("mode is {}".format(mode))
    s = s/(len(data)-mode)
    return math.sqrt(s)


url = "/home/dev/sketchbookProcessing/BNEpaperSims/VolBot_Flashlight_flat/data/flashwidth"
beta = sys.argv[1]
folder_add = ["1/", "2/", "3/", "4/", "5/", "6/", "7/", "8/", "9/", "10/"]
common = "Flash_flat_300_4000_40_2_"
first_angle = 10
last_angle = 90
step = 10
angles = [i for i in range(first_angle, last_angle+step, step)]
heights = []
max_x = []
std_dev = []
for a in angles:
    mean = 0.0
    initial = 0.0
    dev = 0.0
    max_x = [] # list of max_values per angle per sample set (initial position). This is for use in calculating std.deviation

    for f in folder_add:
        data = pd.read_csv(url + str(beta) + "/" + "data"+ f  + common + str(a) + "_" + str(beta), header=None)
        adata = np.copy(data.values)
        initial = adata[0][0]
        x_values = adata[:, 0]
        max_x.append(np.max(x_values) - initial)
        mean += (max_x[-1])
         
       # print(x_values.shape)
    #print(max_x)
    mean = mean/len(folder_add) #len(folder_add is the number of sample sets)
    dev = sdev(max_x, mode=1)
    #print(str(a) + '    ' + str(temp) + '   ' + str(dev))
    std_dev.append(dev)
    heights.append(mean)
    
fig, ax = plt.subplots()
ax.set_xlim(left=0, right=(max(angles) + angles[0]))
ax.set_xticks(angles)
ax.set_xticklabels([str(a) for a in angles])
ax.set_ylim(bottom=0, top= (max(heights) + max(std_dev)))
ax.bar(angles, heights, width=5)
ax.errorbar(angles, heights, std_dev, ecolor='black', elinewidth=0.8, capsize=6)
plt.show()
