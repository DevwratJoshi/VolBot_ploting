import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
X = "PIV_X" # The beginning of the X file
Y = "PIV_Y" # The beginning of the Y file
no = "100" # The number of robots
box_bottom = "900"
freq = "20"
amp = "8"
up = "3"
down = "3"
s = "_" # The seperator between the above in the file name
url = "/home/dev/sketchbookProcessing/BNEpaperSims/VolBot_Flow/data/"
extention = ".txt"
dataX = pd.read_csv(url + X + no + s + box_bottom + s + freq + s + amp + s + up + s  + down + extention, header=None, usecols = [i for i in range(10)])
dataY = pd.read_csv(url + Y + no + s + box_bottom + s + freq + s + amp + s + up + s  + down + extention, header=None, usecols = [i for i in range(10)])

#print(data.values[0][0]) # This gives us the [][]th value 

#Remember that data for velocity at (x,y) is stored at data.values[y][x]
#Flip the array using np.flipud
def makeColourTuple(*ctup): #Makes a colour tuple using rgb values out of 255
    return tuple([round(float(i/255.0), 2) if i < 255 else 255.0 for i in ctup])
    
trueX = np.copy(dataX.values)
trueX = np.flipud(trueX)
trueY = np.copy(dataY.values)
trueY = np.flipud(trueY)
x,y = np.meshgrid(np.linspace(1, 10, 10), np.linspace(1, 10, 10))

for i in range(trueX.shape[0]):
     for j in range(trueX.shape[1]):
             a = trueX[i][j]
             b = trueY[i][j]
             trueX[i][j] = trueX[i][j]/math.sqrt(a**2 + b**2)
             trueY[i][j] = trueY[i][j]/math.sqrt(a**2 + b**2)
fig, ax = plt.subplots(1,1)

ax.quiver(x,y,trueX, trueY)
ax.set_title("freq {} amp {} up {} down {}".format(freq, amp, up, down))
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
plt.xticks(np.arange(1, 10+1, 1))
plt.yticks(np.arange(0, 10+1, 1))

xu = float(up) *(11.0/(6.0))
yu = 11
xd = float(down)*(11.0/6.0)
yd = 0

plt.plot([xu, xd], [yu, yd], color=(makeColourTuple(112, 48, 160)), linewidth=2)
plt.show()
