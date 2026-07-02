import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random


#Taking data from txt file for now
with open("MuonStartsAndDirecs.txt","r") as file:
    data = file.read()

#Straight from stackoverflow
data = np.random.rand(10, 10) * 20
# create discrete colormap
cmap = colors.ListedColormap(['red', 'blue'])
bounds = [0,1,2] #[0,1) are red, [1,2) are blue
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(-.5, 10, 1));
ax.set_yticks(np.arange(-.5, 10, 1));

plt.show()