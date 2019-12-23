import pandas as pd
import matplotlib.pyplot as plt
data_cw = pd.read_csv('Big60.0Small30.0Clockwisefreq20.0amp6.txt', header=None)
data_ccw = pd.read_csv('Big60.0Small30.0CounterClockwisefreq20.0amp6.txt', header=None)

header = ['X-position', 'Y-position']

data_cw.columns = header
data_ccw.columns = header

data_cw['Y-position'] = 900 - data_cw['Y-position']
data_ccw['Y-position'] = 900 - data_ccw['Y-position']

data_cw['Y-position'] = data_cw['Y-position'] - 200
data_ccw['Y-position'] = data_ccw['Y-position'] - 200

data_cw['X-position'] = data_cw['X-position'] - 100
data_ccw['X-position'] = data_ccw['X-position'] - 100 

ax = data_cw.plot(x = 'X-position', y = 'Y-position', kind='line', color='g', legend=False)
data_ccw.plot(x = 'X-position', y = 'Y-position', kind='line', color='b' ,ax=ax, legend=False)

left, right = plt.xlim() # getting the x and y axis limits of the current figure 
#print("{} {}".format(left, right))


plt.xlim(0, 800)
plt.ylim(bottom=0)

plt.show()