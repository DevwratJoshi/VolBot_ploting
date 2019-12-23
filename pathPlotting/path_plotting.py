import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
X = "PIV_X" # The beginning of the X file
Y = "PIV_Y" # The beginning of the Y file
no = "100" # The number of robots
box_bottom = 960
big = 40

data_cw = pd.read_csv('/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flow_Sims/position_track_clockwise', header=None, index_col=False)
data_ccw = pd.read_csv('/home/dev/Devwrat/Masters/Research/Journal papers/AMAM special issue/data/VolBot_Flow_Sims/position_track_counter', header=None, index_col=False)

cw = data_cw.values
ccw = data_ccw.values

cw[:,0] = cw[:,0] - 1200/2 + box_bottom/2
ccw[:,0] = ccw[:,0] - 1200/2 + box_bottom/2
cw = cw/big
ccw = ccw/big


fig,(ax_cw, ax_ccw) = plt.subplots(1,2, figsize=(15,10))

ax_cw.plot(cw[50:1600,0], cw[50:1600,1], c=(0,0,1))
ax_ccw.plot(ccw[50:1600,0], ccw[50:1600,1], c=(0,1,0))

ax_cw.plot([box_bottom/(big*2), (4*box_bottom/(5*big))], [box_bottom/big, 0], c=(112/255,48/255,160/255), linewidth=2)
ax_ccw.plot([box_bottom/(big*2), (box_bottom/(10*big))], [box_bottom/big, 0], c=(112/255,48/255,160/255), linewidth=2)

ax_cw.set_xlabel("X-position [module-lengths]", fontsize='xx-large') 
ax_ccw.set_xlabel("X-position [module lengths]", fontsize='xx-large')
ax_cw.set_ylabel("Y-position [module-lengths]", fontsize='xx-large') 
ax_ccw.set_ylabel("Y-position [module lengths]", fontsize='xx-large')

ax_cw.set_xticks(np.arange(0, 1 +  box_bottom/big, 2))
ax_ccw.set_xticks(np.arange(0, 1 + box_bottom/big, 2))
ax_cw.set_yticks(np.arange(0, 1 +  box_bottom/big, 2))
ax_ccw.set_yticks(np.arange(0, 1 + box_bottom/big, 2))

ax_cw.tick_params(labelsize=16)
ax_ccw.tick_params(labelsize=16)
plt.show()
