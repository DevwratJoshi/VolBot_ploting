import sys
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import math
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
X = "PIV_X" # The beginning of the X file
Y = "PIV_Y" # The beginning of the Y file
no = "100" # The number of robots
box_bottom = "900"
freq = "20"
amp = "8"
up = "5"
down = "1"
s = "_" # The seperator between the above in the file name
url = "/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flow_Sims/PIV_data_constant_top_5/middle/sections_max20/"
extention = ""
#print(data.values[0][0]) # This gives us the [][]th value 

#Remember that data for velocity at (x,y) is stored at data.values[y][x]
#Flip the array using np.flipud
def makeColourTuple(*ctup): #Makes a colour tuple using rgb values out of 255
    return tuple([round(float(i/255.0), 2) if i < 255 else 255.0 for i in ctup])

if len(sys.argv) < 3:
        print("Usage: python plotCenter.py {down from 0 to 10} {1 if run newton-raphson, else 0} {initial x to run newton raphson} {initial y to run newton raphson} ")
        sys.exit()
if len(sys.argv) < 5:
    if int(sys.argv[2]) == 1:
        print("Please specify netwon raphson parameters")
        print("Usage: python plotCenter.py {down from 0 to 10} {1 if run newton-raphson, else 0} {initial x to run newton raphson} {initial y to run newton raphson} {times to run newton raphson (if last arg is true)} ")
        sys.exit()

down = str(sys.argv[1])

if int(down) > 10 or int(down) < 0:
    print("Down is out of range. Enter a value between 0 and 10 (inclusive)")
    sys.exit()


dataX = pd.read_csv(url + "X" + "/" + "bottom" + down +  extention, header=None, usecols = [i for i in range(20)])
dataY = pd.read_csv(url + "Y" + "/" + "bottom" + down +  extention, header=None, usecols = [i for i in range(20)])


trueX = np.copy(dataX.values)
trueX = np.flipud(trueX)
trueY = np.copy(dataY.values)
trueY = np.flipud(trueY)
xlin = np.linspace(0.5, 19.5, 20)
ylin = np.linspace(0.5, 19.5, 20)

x,y = np.meshgrid(xlin, ylin)
trueX_norm = np.copy(trueX)
trueY_norm = np.copy(trueY)
for i in range(trueX.shape[0]):
     for j in range(trueX.shape[1]):
             a = trueX[i][j]
             b = trueY[i][j]
             if(a == 0 and b == 0):
                 trueX_norm[i][j] = 0
                 trueY_norm[i][j] = 0
             else:
                trueX_norm[i][j] = trueX[i][j]/math.sqrt(a**2 + b**2)
                trueY_norm[i][j] = trueY[i][j]/math.sqrt(a**2 + b**2)

# The inperpolator is not going to zero reliably. Adding a padded layer of very high value vectors outside trueX_norm and trueY_norm to prevent it from moving outside

#Now to calculate the curl
# f describes the x component and g describes the y component of the vector at (x,y) of the continuous vector field f(x,y)i + g(x,y)j
#xlin = np.linspace(0, 20, 22)
#ylin = np.linspace(0, 20, 22)
f = interpolate.RectBivariateSpline(xlin, ylin, np.transpose(trueX_norm)) 
g = interpolate.RectBivariateSpline(xlin, ylin, np.transpose(trueY_norm))


fig, ax = plt.subplots(1,1, figsize=(12,12))

epsilon = 0.0001 
points = [] # This is a list that holds the points close to epsilon
# Using newton raphson method to find the zero point

xp = 10
yp = 10

x_low = xp
y_low = yp

X = np.zeros((2,1))
counter = 0
print(int(sys.argv[2]))
if int(sys.argv[2]) == 1:
    xp = float(sys.argv[3])
    yp = float(sys.argv[4])
    #Start newton raphson algorithm
    while 1:     
         X[0][0] = xp
         X[1][0] = yp
         J = np.zeros((2,2))
         F = np.zeros(X.shape)
         F[0][0] = f(X[0][0],  X[1][0])
         F[1][0] = g(X[0][0], X[1][0])
         J[0][0] = f.__call__(xp, yp, 1, 0)
         J[1][0] =  g.__call__(xp, yp, 1, 0)
         J[0][1] = f.__call__(xp, yp, 0, 1)
         J[1][1] = g.__call__(xp, yp, 0, 1)

         JI = inv(J)
         temp = np.matmul(JI, F)
         c = X - temp
         if math.sqrt((xp-c[0][0])**2 + (yp-c[1][0])**2) < epsilon:
             break
#End newton raphson algorithm if the distance between the discovered point and 0 is closer than epsilon
         counter += 1 # Count the number of times attempted
         if(counter > 1000):
             print("Counter too big")
             break #Break if the algorithm cannot find a solution
         
         xp = c[0][0]
         yp = c[1][0]

print(counter)
xp= [xp]
yp = [yp]
print("X_pos = {}, y_pos = {}".format(xp, yp))
ax.scatter([xp], [yp], color='red', marker='x', s=100)
#ax.scatter(x_low, y_low, color='blue', marker='x')
#ax.scatter(x_low, y_low, color='blue', marker='x')
#der = der[:-1, :-1]
#levels = MaxNLocator(nbins='auto').tick_values(der.min(), der.max())
#cmap = plt.get_cmap('RdBu')
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#im = ax.pcolormesh(derX, derY, der, cmap=cmap, norm=norm)
#fig.colorbar(im, ax=ax)

ax.quiver(x,y,trueX_norm, trueY_norm, pivot='mid')
bottom_counter = int(down) - int(up)
side = ""

if(bottom_counter > 0):
    side = 'right'
else:
    bottom_counter *= -1
    side = 'left'

ax.set_title("Freqency {}, Amplitude {}, {} side, i = {}".format(freq, amp, side, bottom_counter))
ax.set_xlabel("X position", fontsize='xx-large')
ax.set_ylabel("Y position", fontsize='xx-large')
plt.xticks(np.arange(0, 20+1, 1))
plt.yticks(np.arange(0, 20+1, 1))
ax.tick_params(labelsize=16)
xu = float(up) *(20.0/(10.0))
yu = 20
xd = float(down)*(20.0/10.0)
yd = 0


plt.gca().set_aspect('equal', adjustable='box')
plt.plot([xu, xd], [yu, yd], color=(makeColourTuple(112, 48, 160)), linewidth=2)
plt.show()
