import matplotlib.pyplot as plt
import numpy as np

# Overall size of the plot | Must be a number N such that sqrt(N-1) = int
x_fine = np.arange(17) # N-1 
y_fine = np.arange(17)

numX = len(x_fine)
numY = len(y_fine)

X_f, Y_f = np.meshgrid(x_fine, y_fine)

#Creating fake data
Z_fine = np.random.rand(numX-1,numY-1) #Make dependent on length of two data sets | Currently making random data fro 16x16
for i in range(numY): #Rows
    for j in range(numX): #Columns
        if i%4 == 0 or i == 0:
            Z_fine[i-1,j-1] = 0


xGroupSize = np.sqrt(numX-1)
yGroupSize = np.sqrt(numY-1)

#Grouping the data correctly


# 2. Generate macro mesh boundaries (Grouped by blocks of 4)
x_macro = np.arange(0, numX,xGroupSize) # 0 -> numX in group of size sqrt(numX-1)
y_macro = np.arange(0, numY,yGroupSize)
X_m, Y_m = np.meshgrid(x_macro, y_macro)

fig, ax = plt.subplots(figsize=(7, 7)) #Determines size of the plot on the computer screen

# Layer 1: Plot the actual colorized data with faint cell walls
ax.pcolormesh(X_f, Y_f,Z_fine, cmap="plasma", edgecolor="white", linewidth=0.5)

# Layer 2: Overlay empty faces with white grid outlines
# facecolor='none' keeps the underlying cell data completely visible
ax.pcolormesh(X_m, Y_m, np.zeros((4, 4)), facecolor="none", edgecolor="white", linewidth=2.5) #the zeros should match to the size of the groups for x and y macro

plt.show()