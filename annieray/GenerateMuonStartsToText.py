import numpy as np
#NOTE: The up direction is z and not y, currently the code considers y to be the beam direction instead of z


#Generating vector origins amd directions| x,y,z,t0,dx,dy,dz is the order the txt file must be in
x = 1000*np.linspace(-1.2,1.2, 13) #Converting to mm
y=1000*np.zeros(169) #Converted to mm
z=1000*np.linspace(0.8,3.2, 13) #Converted to mm


#Default values
t0=np.zeros(169)
dx=np.zeros(169)
dy=np.full(169,1)
dz=np.zeros(169)

numTrackParams = 7 #Number of parameters for each track (x,y,z,t0,dx,dy,dz)

xx,zz = np.meshgrid(x,z) #generates coordinates from the x and z values

xzPairs = np.vstack([xx.ravel(),zz.ravel()]).T #coherent coordinates for muon

#Adding in angle grid for variation in x and z direction | Reassign the default values to actual values
dx = np.linspace(-0.5,0.5,5) #5 evenly spaced dx values 
dz = np.linspace(-0.5,0.5,5) #5 evenly spaced dz values 

ddx,ddz = np.meshgrid(dx,dz)
dxdzPairs = np.vstack([ddx.ravel(),ddz.ravel()]).T #coordinates for dx and dz

muonStartsDirecs = np.zeros((len(xzPairs)*len(dxdzPairs),numTrackParams)) 

for i in range(len(xzPairs)):
    for j in range(len(dxdzPairs)):
        muonStartsDirecs[i*25+j,0] = xzPairs[i,0] #x
        muonStartsDirecs[i*25+j,1] = y[i] #y
        muonStartsDirecs[i*25+j,2] = xzPairs[i,1] #z
        muonStartsDirecs[i*25+j,3] = t0[i] #t0
        muonStartsDirecs[i*25+j,4] = dxdzPairs[j,0] #dx
        muonStartsDirecs[i*25+j,5] = dy[i] #dy
        muonStartsDirecs[i*25+j,6] = dxdzPairs[j,1] #dz


#Writing the txt file
with open("MuonStartsAndDirecs.txt","w",encoding="utf-8") as file:
    for item in muonStartsDirecs:
        #Must convert tuples to strings and join them
        line = " ".join(str(element) for element in item)
        file.write(line + "\n")

