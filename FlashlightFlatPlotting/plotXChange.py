import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math

if len(sys.argv) < 4:
    print("Usage: plotPosChange.py {R/B/A (The parameter to change)} {value of second param in order RBA} {value of third param in RBA}")
    sys.exit()
big = 60*2
small = 20*2
mid = 30*2
R_list = ['120', '180', '240']
B_list = [str(int(i)) for i in np.linspace(20,180,9)]
A_list = [str(int(i)) for i in np.linspace(10,90,9)]
data_list = [str(int(i)) for i in np.linspace(1,10,10)]

def sdev(data, mode=0):
    u = sum(data)
    u = u/len(data)
    #u contains the mean of data here
    s = sum((x-u)**2 for x in data)
    #print("mode is {}".format(mode))
    s = s/(len(data)-mode)
    return math.sqrt(s)

xPosInit = np.zeros((101,)) # First element is initial element. Followed by last 100 elements
xPosSdev = []

fig, ax = plt.subplots(1,1, figsize=(10,10))



if(sys.argv[1] == 'R'):
    list1 = R_list
    xStats = np.zeros((len(R_list), 2)) # The array with column 1 storing mean and column 2 storing std.dev of change in x coordinate
    beta = sys.argv[2]
    alpha = sys.argv[3]
    if not str(sys.argv[2]) in B_list:
        print("3rd parameter not in list")
        sys.exit()
    if not str(sys.argv[3]) in A_list:
        print("4th parameter not in list")
        sys.exit()

    for i in range(len(list1)):
        for d in data_list:
            url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flashlight_Flat_sims/data_R_alpha_beta/R" + list1[i] + "/Beta" + beta + "/alpha_fold" + alpha + "/data" + d + "/Flash_flat_" + list1[i] + ".0_" + beta + "_" + alpha
            data = pd.read_csv(url, header=None, index_col=False)
            xPosInit[0] = xPosInit[0] + data.values[0][0]
            #print(np.median(data.values[699:-1,0]))
            xPosSdev.append(np.median(data.values[699:-1,0]))
            #xPosSdev.append(sum(data.values[699:-1,0])/len(data.values[699:-1,0]))
            xPosInit[1:] = xPosInit[1:] + data.values[699:-1, 0]
        
        xPosInit = xPosInit/len(data_list)
        x_initial = xPosInit[0]
        x_final = sum(xPosInit[1:])/len(xPosInit[1:])
        xStats[i][0] = (x_final - x_initial)/mid
        xStats[i][1] = sdev(xPosSdev, mode=1)/mid
        xPosSdev = []

elif(sys.argv[1] == 'B'):
    list1 = B_list
    xStats = np.zeros((len(B_list), 2)) # The array with column 1 storing mean and column 2 storing std.dev of change in x coordinate
    R = sys.argv[2]
    alpha = sys.argv[3]
    if not str(sys.argv[2]) in R_list:
        print("3rd parameter not in list")
        sys.exit()
    if not str(sys.argv[3]) in A_list:
        print("4th parameter not in list")
        sys.exit()

    for i in range(len(list1)):
        for d in data_list:
            url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flashlight_Flat_sims/data_R_alpha_beta/R" + R + "/Beta" + list1[i] + "/alpha_fold" + alpha + "/data" + d + "/Flash_flat_" + R + ".0_" + list1[i] + "_" + alpha
            data = pd.read_csv(url, header=None, index_col=False)
            xPosInit[0] = xPosInit[0] + data.values[0][0]
            #print(np.median(data.values[699:-1,0]))
            xPosSdev.append(np.median(data.values[699:-1,0]))
            #xPosSdev.append(sum(data.values[699:-1,0])/len(data.values[699:-1,0]))
            xPosInit[1:] = xPosInit[1:] + data.values[699:-1, 0]
        
        xPosInit = xPosInit/len(data_list)
        x_initial = xPosInit[0]
        x_final = sum(xPosInit[1:])/len(xPosInit[1:])
        xStats[i][0] = (x_final - x_initial)/mid
        xStats[i][1] = sdev(xPosSdev, mode=1)/mid
        xPosSdev = []

elif(sys.argv[1] == 'A'):
    list1 = A_list
    xStats = np.zeros((len(A_list), 2)) # The array with column 1 storing mean and column 2 storing std.dev of change in x coordinate
    R = sys.argv[2]
    beta = sys.argv[3]
    if not str(sys.argv[2]) in R_list:
        print("3rd parameter not in list")
        sys.exit()
    if not str(sys.argv[3]) in B_list:
        print("4th parameter not in list")
        sys.exit()

    for i in range(len(list1)):
        for d in data_list:
            url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flashlight_Flat_sims/data_R_alpha_beta/R" + R + "/Beta" + beta + "/alpha_fold" + list1[i] + "/data" + d + "/Flash_flat_" + R + ".0_" + beta + "_" + list1[i]
            data = pd.read_csv(url, header=None, index_col=False)
            xPosInit[0] = xPosInit[0] + data.values[0][0]
            #print(np.median(data.values[699:-1,0]))
            xPosSdev.append(np.median(data.values[699:-1,0]))
            #xPosSdev.append(sum(data.values[699:-1,0])/len(data.values[699:-1,0]))
            xPosInit[1:] = xPosInit[1:] + data.values[699:-1, 0]
        
        xPosInit = xPosInit/len(data_list)
        x_initial = xPosInit[0]
        x_final = sum(xPosInit[1:])/len(xPosInit[1:])
        xStats[i][0] = (x_final - x_initial)/mid
        xStats[i][1] = sdev(xPosSdev, mode=1)/mid
        xPosSdev = []
            
else:
    print("1st parameter wrong. Use R, B or A")
    sys.exit()



ax.set_ylim(bottom = 0)
ax.set_xticks(np.arange(len(list1)))
ax.set_yticks(np.linspace(0, mid*30/mid, 16))
ax.set_xticklabels([int(i) for i in list1])
ax.bar(list1, xStats[:,0], width=0.4)
ax.errorbar(list1, xStats[:,0], xStats[:,1], ecolor='black', elinewidth=0.8, capsize=6)
ax.tick_params(labelsize=18)
ax.set_xlabel("Angle width of the flashlight", fontsize='xx-large')
ax.set_ylabel("Horizontal distance travelled [module lengths]", fontsize='xx-large')
plt.show()
