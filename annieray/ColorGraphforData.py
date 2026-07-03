import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random


dat = open('data.txt', 'r')
data = dat.read()
dat.close()


#x_vals = np.linspace(-1.2-0.5*(2.4/13.), 1.2+0.5*(2.4/13.), 13)
#y_vals = np.linspace(-1.2-0.5*(2.4/13.), 1.2+0.5*(2.4/13.), 13)

x_vals = np.linspace(-1.2, 1.2, 13)
y_vals = np.linspace(-1.2, 1.2, 13)
XX,YY = np.meshgrid(x_vals, y_vals)
fig,ax = plt.subplots()
plt.pcolormesh(XX,YY,data, cmap='viridis', shading='auto',edgecolors = 'w',linewidths=0.5)
ax.set_xticks(x_vals)
ax.set_yticks(y_vals)

plt.colorbar()

plt.show()