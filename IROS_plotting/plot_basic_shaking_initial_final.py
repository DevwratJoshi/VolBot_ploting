import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
data_files = 5 #The number of data files
folder = "/home/dev/sketchbookProcessing/IROS_Sims/Horizontal_shaking_cluster_histogram/data_test/Segregator_Size34/mover_small_frac/"
ext = ".txt" #Data file extention
mover_small_frac = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0,8', '0.9']

big = 34.6
small = 20.0

fig,(ax_initial, ax_final) = plt.subplots(2,1, figsize=(14,12))

mover_small_frac = sys.argv[1]
small_point_size = 50
big_point_size = small_point_size *1.73*1.73
seg_point_size = big_point_size

for i in range(2, 3):
    data_initial = pd.read_csv(folder + mover_small_frac + "/Initial/" + str(i) + ext, header=None, index_col=False)
    data_final = pd.read_csv(folder + mover_small_frac+"/Final/" + str(i) + ext, header=None, index_col=False)


    initial_points_mover_small = data_initial.loc[(data_initial[2] == 'm') & (data_initial[4] == small)][[0,1]]
    initial_points_mover_big = data_initial.loc[(data_initial[2] == 'm') & (data_initial[4] == big)][[0,1]]
    initial_points_segregator = data_initial.loc[data_initial[2] == 's'][[0,1]]

    final_points_mover_small = data_final.loc[(data_final[2] == 'm') & (data_final[4] == small)][[0,1]]
    final_points_mover_big = data_final.loc[(data_final[2] == 'm') & (data_final[4] == big)][[0,1]]
    final_points_segregator = data_final.loc[data_final[2] == 's'][[0,1]]
    


    ax_initial.scatter(initial_points_mover_small.values[:,0]/(small*2), initial_points_mover_small.values[:,1]/(small*2), color='green', marker='o', s=small_point_size)
    ax_initial.scatter(initial_points_mover_big.values[:,0]/(small*2), initial_points_mover_big.values[:,1]/(small*2), color='green', marker='o', s=big_point_size)
    ax_initial.scatter(initial_points_segregator.values[:,0]/(small*2), initial_points_segregator.values[:,1]/(small*2), color='red', marker='o', s=seg_point_size)
    
    ax_final.scatter(final_points_mover_small.values[:,0]/(small*2), final_points_mover_small.values[:,1]/(small*2), color='green', marker='o', s=small_point_size)
    ax_final.scatter(final_points_mover_big.values[:,0]/(small*2), final_points_mover_big.values[:,1]/(small*2), color='green', marker='o', s=big_point_size)
    ax_final.scatter(final_points_segregator.values[:,0]/(small*2), final_points_segregator.values[:,1]/(small*2), color='red', marker='o', s=seg_point_size)
#plt.plot(initial_points_low_friction[:,0], initial_points_low_friction[:,1])
#plt.plot(final_points_low_friction)
ax_initial.set_title("Initial positions")
ax_final.set_title("Final positions")
ax_initial.set_xlabel("X position", fontsize='xx-large')
ax_initial.set_ylabel("Y position", fontsize='xx-large')
ax_final.set_xlabel("X position", fontsize='xx-large')
ax_final.set_ylabel("Y position", fontsize='xx-large')
plt.show()

