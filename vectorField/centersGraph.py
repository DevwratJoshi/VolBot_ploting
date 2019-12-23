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

def sdev(data, mode=0):
    u = sum(data)
    u = u/len(data)
    #u contains the mean of data here
    s = sum((x-u)**2 for x in data)
    #print("mode is {}".format(mode))
    s = s/(len(data)-mode)
    return math.sqrt(s)

url_start ="/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flow_Sims/PIV_data_constant_top_"
url_end = "/middle/sections_max20/centers"
Ncount = ['1','2','3','4','5']
#Begin collecting center data
data = pd.read_csv(url_start + Ncount[0] + url_end, index_col=False)
centerData1 = data.values[:, 1:3]
data = pd.read_csv(url_start + Ncount[1] + url_end, index_col=False)
centerData2 = data.values[:, 1:3]
data = pd.read_csv(url_start + Ncount[2] + url_end, index_col=False)
centerData3 = data.values[:, 1:3]
data = pd.read_csv(url_start + Ncount[3] + url_end, index_col=False)
centerData4 = data.values[:, 1:3]
data = pd.read_csv(url_start + Ncount[4] + url_end, index_col=False)
centerData5 = data.values[:, 1:3]

#End collecting center data
centerlist = []
centerlist.append(centerData1)
centerlist.append(centerData2)
centerlist.append(centerData3)
centerlist.append(centerData4)
centerlist.append(centerData5)

for i in centerlist:
    i[:, 0] = i[:, 0] - 10
    i[:, 1] = 20 - i[:, 1]

centerCircData1 = np.zeros((11,2))
centerCircData2 = np.zeros((11,2))
centerCircData3 = np.zeros((11,2))
centerCircData4 = np.zeros((11,2))
centerCircData5 = np.zeros((11,2))

centerCircList = []
centerCircList.append(centerCircData1)
centerCircList.append(centerCircData2)
centerCircList.append(centerCircData3)
centerCircList.append(centerCircData4)
centerCircList.append(centerCircData5)

#Begin converting center data from cartesian to circular coordinates

for i in range(len(centerCircList)):
    centerCircList[i][:, 0] = centerlist[i][:,0]**2 + centerlist[i][:,1]**2
    centerCircList[i][:,0] = np.sqrt(centerCircList[i][:,0])
    t = centerlist[i][:,0]/centerlist[i][:,1]
    centerCircList[i][:,1] = t
    centerCircList[i][:,1] = np.arctan(centerCircList[i][:,1])

#End converting to cicular coordinates

#Begin calculating mean and std deviation

r_stats = np.zeros((11,2)) #This array will hold the means (1st column) and std deviations (2nd column) of the r data 
theta_stats = np.zeros((11,2)) #This array will hold the means (1st column) and std deviations (2nd column) of the theta data 
r_array = np.zeros((11, 5))
theta_array = np.zeros((11, 5)) # Rows = for different bottom positions. Columns = for different initial positions

for i in range(len(centerCircList)):
    r_array[:, i] = centerCircList[i][:,0]
    theta_array[:,i] = centerCircList[i][:,1]
    
theta_array = 180*theta_array/math.pi


for j in range(11):
    r_stats[j][1] = sdev(r_array[j,:], mode=1)
    theta_stats[j][1] = sdev(theta_array[j,:], mode=1)
    r_stats[j,0] = sum(r_array[j,:])/len(r_array[j,:])
    theta_stats[j,0] = sum(theta_array[j,:])/len(theta_array[j,:])

#End calculating mean and deviation
#Begin setting up the array holding the line angles 
bound_angle = []
for i in range(1,6):
    bound_angle.append(180*math.atan(i/10)/math.pi)

print(bound_angle)
#End setting boundary line angle array
fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(15,10))


lines = []

for i in range(11):
    lines.append([(10, 20), (i*2, 0)])
#lc = mc.LineCollection(lines, colors=c, linewidths=2)

#ax.add_collection(lc)
ax1.set_title("Distance from O_C for different boundary line angles")
ax2.set_title("Angle theta for different boundary line angles")
ax1.set_xlabel("Boundary line angle theta", fontsize='xx-large')
ax2.set_xlabel("Boundary line angle theta", fontsize='xx-large')
ax1.set_ylabel("Distance between flow center and O_C", fontsize='xx-large')
ax2.set_ylabel("Angle between vertical and position vector of center point", fontsize='xx-large')

ax1.set_xticks(np.arange(0, 30, 2))
ax1.set_yticks(np.arange(0, np.amax(r_stats[:,0])+2, 0.5))
ax2.set_xticks(np.arange(0, 30, 2))
ax2.set_yticks(np.arange(0, math.ceil(np.amax(theta_stats[:,0])+2), 1))

ax1.plot(bound_angle, r_stats[6:,0])
ax1.scatter(bound_angle,r_stats[6:,0], color='red', marker='x')
ax2.plot(bound_angle, theta_stats[6:,0])
ax2.scatter(bound_angle,theta_stats[6:,0], color='red', marker='x')
ax1.errorbar(bound_angle, r_stats[6:,0], r_stats[6:,1], ecolor='black', elinewidth=0.8, capsize=6)
ax2.errorbar(bound_angle, theta_stats[6:,0], theta_stats[6:,1], ecolor='black', elinewidth=0.8, capsize=6)
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
plt.show()
