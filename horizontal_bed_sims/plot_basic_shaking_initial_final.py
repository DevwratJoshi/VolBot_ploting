import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_files = 5 #The number of data files
folder = "/home/dev/sketchbookProcessing/Horizontal_Bed_Simulations/Basic_Shaking_Init_Final_Posns/data/"
ext = ".txt" #Data file extention
initial_points_low_friction = []
initial_points_high_friction = []
final_points_low_friction = []
final_points_high_friction = []

fig,(ax_initial, ax_final) = plt.subplots(1,2, figsize=(15,10))

for i in range(1, 5+1):
    data_initial = pd.read_csv(folder+"Initial/" + str(i) + ext, header=None, index_col=False)
    print(folder + "Final/" + str(i) + ext)
    data_final = pd.read_csv(folder+"Final/" + str(i) + ext, header=None, index_col=False)


    initial_points_low_friction = data_initial.loc[data_initial[2] == 'g'][[0,1]]
    initial_points_high_friction = data_initial.loc[data_initial[2] == 'r'][[0,1]]
    final_points_low_friction = data_final.loc[data_final[2] == 'g'][[0,1]]
    final_points_high_friction = data_final.loc[data_final[2] == 'r'][[0,1]]
    


    ax_initial.scatter(initial_points_low_friction.values[:,0], initial_points_low_friction.values[:,1], color='green', marker='o')
    ax_initial.scatter(initial_points_high_friction.values[:,0], initial_points_high_friction.values[:,1], color='red', marker='o')
    ax_final.scatter(final_points_low_friction.values[:,0], final_points_low_friction.values[:,1], color='green', marker='o')
    ax_final.scatter(final_points_high_friction.values[:,0], final_points_high_friction.values[:,1], color='red', marker='o')
#plt.plot(initial_points_low_friction[:,0], initial_points_low_friction[:,1])
#plt.plot(final_points_low_friction)

plt.show()

