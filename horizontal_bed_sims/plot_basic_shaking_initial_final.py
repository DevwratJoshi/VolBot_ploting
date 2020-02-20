import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_files = 5 #The number of data files
folder = "/home/dev/sketchbookProcessing/Horizontal_Bed_Simulations/Basic_Shaking_Init_Final_Posns/data/Yes_Friction_Yes_Size/Big_High_Friction/"
ext = ".txt" #Data file extention
initial_points_low_friction = []
initial_points_high_friction = []
final_points_low_friction = []
final_points_high_friction = []

fig,(ax_initial, ax_final) = plt.subplots(1,2, figsize=(15,10))

for i in range(1, 2):
    data_initial = pd.read_csv(folder+"Initial/" + str(i) + ext, header=None, index_col=False)
    data_final = pd.read_csv(folder+"Final/" + str(i) + ext, header=None, index_col=False)


    initial_points_low_friction = data_initial.loc[data_initial[2] == 'g'][[0,1]]
    initial_points_high_friction = data_initial.loc[data_initial[2] == 'r'][[0,1]]
    final_points_low_friction = data_final.loc[data_final[2] == 'g'][[0,1]]
    final_points_high_friction = data_final.loc[data_final[2] == 'r'][[0,1]]
    


    ax_initial.scatter(initial_points_high_friction.values[:,0], initial_points_high_friction.values[:,1], color='red', marker='o', s=112.5)
    ax_initial.scatter(initial_points_low_friction.values[:,0], initial_points_low_friction.values[:,1], color='green', marker='o', s=50)
    ax_final.scatter(final_points_high_friction.values[:,0], final_points_high_friction.values[:,1], color='red', marker='o', s=112.5)
    ax_final.scatter(final_points_low_friction.values[:,0], final_points_low_friction.values[:,1], color='green', marker='o', s=50)
#plt.plot(initial_points_low_friction[:,0], initial_points_low_friction[:,1])
#plt.plot(final_points_low_friction)
ax_initial.set_title("Initial positions")
ax_final.set_title("Final positions")
ax_initial.set_xlabel("X position", fontsize='xx-large')
ax_initial.set_ylabel("Y position", fontsize='xx-large')
ax_final.set_xlabel("X position", fontsize='xx-large')
ax_final.set_ylabel("Y position", fontsize='xx-large')
plt.show()

