import numpy as np
#NOTE: The up direction is z and not y, currently the code considers y to be the beam direction instead of z


#Generating vector origins amd directions| x,y,z,t0,dx,dy,dz is the order the txt file must be in
x = 1000*np.linspace(-1.2,1.2, 13) #Converting to mm
y=1000*np.zeros(169) #Converted to mm
z=1000*np.linspace(0.8,3.2, 13) #Converted to mm

t0=np.zeros(169)
dx=np.zeros(169)
dy=np.full(169,1)
dz=np.zeros(169)

xx,zz = np.meshgrid(x,z) #generates coordinates from the x and y values

xzPairs = np.vstack([xx.ravel(),zz.ravel()]).T #coherent coordinates for muon

#Combine into a matrix for easy file writing
muonStartsDirecs = np.column_stack((xzPairs[:,0],y,xzPairs[:,1],t0,dx,dy,dz))

#Writing the txt file
with open("MuonStartsAndDirecs.txt","w",encoding="utf-8") as file:
    for item in muonStartsDirecs:
        #Must convert tuples to strings and join them
        line = " ".join(str(element) for element in item)
        file.write(line + "\n")

