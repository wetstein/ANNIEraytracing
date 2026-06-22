#DO NOT USE THIS FILE FOR ANY SIMULATION WORK
#DO NOT USE THE CORRESPONDING DATA txt FILE FOR ANY ANALYSIS
#This file is only for testing the basic Cherenkov algorithm and generating data to compare against 

import numpy as np
import random as rng
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd


print("CherenkovAlgoTest.py is Running \n __________________________ ")

#Constants
c = 299792458 #Speed of light in m/s

#Rounding Function
def truncate(number, decimals = 0):
    factor = 10 ** decimals

    return math.trunc(number*factor)/factor

#Takes in angles to make coefficients for x,y,z components
def getComp(thetaC,alpha):
    lam = np.sin(thetaC)*np.cos(alpha)
    kap = np.sin(thetaC)*np.sin(alpha)
    eta = np.cos(thetaC)

    return lam,kap,eta

#Takes in a vector and generates two orthonormal vectors as outputs 
def getBasis(vector):
    mu1,mu2,mu3 = vector
    if mu3 !=0: #Prevent division by zero error
        b1One = 1
        b1Two = 1
        b1Three = -(mu1+mu2)/mu3

        b1 = np.array([b1One,b1Two,b1Three])
        b1 = b1/np.linalg.norm(b1) #Normalizing
    
        b2 = np.cross(vector,b1)
        b2 = b2/np.linalg.norm(b2) #Normalizing
    elif mu2 != 0: #Prevent division by zero error
        b1One = 1
        b1Two = -(mu3+mu1)/mu2
        b1Three = 1

        b1 = np.array([b1One,b1Two,b1Three])
        b1 = b1/np.linalg.norm(b1) #Normalizing
    
        b2 = np.cross(vector,b1)
        b2 = b2/np.linalg.norm(b2) #Normalizing
    elif mu1 != 0: #Prevent division by zero error
        b1One = -(mu3+mu2)/mu1
        b1Two = 1
        b1Three = 1

        b1 = np.array([b1One,b1Two,b1Three])
        b1 = b1/np.linalg.norm(b1) #Normalizing
    
        b2 = np.cross(vector,b1)
        b2 = b2/np.linalg.norm(b2) #Normalizing 
    else:
        print('Non physical vector input: ')
     
    return b1,b2

#Takes in the vector to generate two orthogonal vectors along with two angles
#Calls the previous two functions to generate the global photon vector
def getPhotonVec(vector,thetaC,alpha):
    b1,b2 = getBasis(vector)
    lam, kap,eta = getComp(thetaC,alpha)
    normVector = vector/np.linalg.norm(vector)
    vector = lam*b1 + kap*b2 + eta*normVector #Photon vector in global cords

    return vector


#User set parameters
beta = 0.999999 # Speed of the muon as a fraction of the speed of light
n =1.34 #Refractive index of medium
muonPath = np.array([0,0,4]) #Total distance traveled by the muon in m
trackPrec = 2 #control the number of decimal places (in m) that you want for the measurement of the muon track as well as number of steps in the for loop
muonStart = np.array([0,0,0,0]) #Set cords for where and when in the tank the muon starts (x,y,z,t) | x,y,z = (m), t = (ns)
photonNum = 150 #Number of photons generated per cm along track | Need to make accurate later

    #Just for plotting the graphs
DetectorPlaneDistance = 5 #Only used for the second graph to test code
j = 1000 #Choosing the number of photons to graph


#Begin Cherenkov photon generation algorithm
muonDirec = muonPath/np.linalg.norm(muonPath) # Direction of muon travel as a unit vector

thetaC = np.arccos(1/(beta*n)) #Cherenkov angle in radians

LengthTravel = truncate(np.sqrt(muonPath[0]**2+muonPath[1]**2+muonPath[2]**2),trackPrec) #Basic linear muon path approximation, the last variable controls decimal place
i = int(LengthTravel*10**trackPrec) #convert m to the desired precision for use as number of iterations in for loop

photonDat = [] #initializing list to hold photon id, segment generation id, global start position, and global direction vectors
netK = [-1] #Initializing list to hold total k iterations
createTime = [] #Initializing a list that will track run time and photon creation time
muonSpeed = beta*c #Speed of muon in m/s
#Calculting time delay and recording the generation time of each photon
timeDelay = ((10**(-trackPrec)) / (beta*c))/(10**-9) # Time delay between photon generation in ns


#This will write over any previous version of the file every time the code is run
with open("CherenkovAlgoTestData.txt", "w") as file:
    file.write("NOT FOR USE IN ANY PART OF THE CODE OR ANALYSIS \n Segment ID, Photon ID, Time of Creation (ns), Global Xhat, Global Yhat, Global Zhat, Global X (m), Global Y (m), Global Z (m)\n") # Write header to file   
    #Generating one photon per unit length along the track
    for j in range(i+1): #+1 to include the end point of the track as well as the zeroth point where the first photon is generated (actually a nice case of 0 indexing working in favor)        
        photonSegmentID = j+1 #This is the ID for the segment of the track where the photon is generated

        #Cords of muon in the global frame 
        posX = (muonDirec[0])*j*(10**-trackPrec) +muonStart[0]#X cord in m
        posY = (muonDirec[1])*j*(10**-trackPrec)+muonStart[1] #Y cord in m
        posZ = (muonDirec[2])*j*(10**-trackPrec)+muonStart[2] #Z cord in m
        timeMuon = j*timeDelay+muonStart[3] #This is the muon time (s)
        print('Photon Generation Event:',j, '\n') #Nice to have some way of measuring progress is happening
        
        #Generate photonNum photons per unit length along the track and perform any necessary calculations
        for k in range(photonNum): 

            #Run spatial probability of photon generation
            photonProb = rng.uniform(0,1)
            photonPos = j*(10**-trackPrec)+photonProb*(10**-trackPrec) #Position of photon along track in m
            
            #Calculate photon creation time in ns from some given start time | Remember the start time is defined in seconds
            createTime.append(((photonPos)/(muonSpeed)+muonStart[3])*(10**9))

            #Generate photon ID
            photonID = netK[-1]+2 #Should be the only line required to assign the id

            #Finding photon position in global
            photonX = photonPos * muonDirec[0] + muonStart[0] #Getting global X photon cord (m) 
            photonY = photonPos * muonDirec[1] + muonStart[1] #Getting global Y photon cord (m) 
            photonZ = photonPos * muonDirec[2] + muonStart[2] #Getting global Z photon cord (m) 

            #Finding directional unit vector of photon compared in global frame
            photonAlpha = rng.uniform(0, 2 * np.pi) #Random azimuthal angle for photon emission
            xDirec, yDirec, zDirec = getPhotonVec(muonDirec,thetaC,photonAlpha)

            #This is the total photon id, +1 has been added to k to avoid multiple zeros in the photon id
            netK.append(k+(j*150)) 

            photonDat.append((photonSegmentID, photonID, createTime[-1], xDirec, yDirec, zDirec, photonX, photonY, photonZ)) # Append photon id, time, direction, and position
            
            #This must be in the k for loop in order to work properly 
            #Having a seperate file to view the output is the easiest way to check that the code is running properly
            file.write(f"{photonSegmentID}, {photonID}, {createTime[-1]}, {xDirec},{yDirec},{zDirec}, {photonX}, {photonY}, {photonZ}\n") # Write photon id, time, and azimuthal angle to file

#Plotting section

#The following is for the first graph
testX = [] #Initialize arrays that will store photon vectors from muon path
testY = []
testZ = []
xOrigin = []#Initialize origins 
yOrigin = []
zOrigin = []
colorArray = plt.colormaps['coolwarm'](np.linspace(0,1,j))

#The following is for the second graph
scatterPlotData = [] #Initializing empty list

for k in range(0,j):
    ranNum = rng.randint(1,len(photonDat)) #Used to get a random selection of photons

    #For the first plot
    testX.append(LengthTravel*photonDat[ranNum][3]+photonDat[ranNum][6]) #Gets the X component
    testY.append(LengthTravel*photonDat[ranNum][4]+photonDat[ranNum][7]) #Gets the Y component
    testZ.append(LengthTravel*photonDat[ranNum][5]+photonDat[ranNum][8]) #Gets the Z component

    xOrigin.append(photonDat[ranNum][6]) #Gets the X origin
    yOrigin.append(photonDat[ranNum][7]) #Gets the Y origin
    zOrigin.append(photonDat[ranNum][8]) #Gets the Z origin
    
    #For the Second Plot
    Lz = DetectorPlaneDistance-zOrigin[-1] #Gets most recent photon's z position and finds distance to detector plane
    photD = Lz/np.cos(thetaC) #Gets total distance traveled by photon
    time2Hit = photD/(c/n)+photonDat[ranNum][2] #Time to reach detector plane in ns
    phi = math.atan2(testY[-1],testX[-1]) % (2*np.pi) #Gets the tan of the photon

    photCR = Lz*np.tan(thetaC)
    xSpot = photCR*np.cos(phi)
    ySpot = photCR*np.sin(phi)
    scatterDF = pd.DataFrame({'index' : [k],'xSpot' : xSpot,'ySpot' : ySpot,'time2Hit' : time2Hit}) #Makes the data into a pandas data frame
    scatterPlotData.append(scatterDF)#Appends data as a pandas data frame

#Muon Path | Graph One
U,V,W = muonPath
x, y, z = 0,0,0
testZ.sort() #Arranges z to be in ascending order so that the color map works
zOrigin.sort() #Arranges the origin in ascending order to match the above line

fig1 = plt.figure()
ax1 = fig1.add_subplot(111,projection='3d')
ax1.quiver(xOrigin,yOrigin,zOrigin,testX,testY,testZ, color = colorArray)
ax1.quiver(x,y,z,U,V,W, color = 'g')
ax1.set_xlabel('X Axis')
ax1.set_ylabel('y Axis')
ax1.set_zlabel('Z Axis')
ax1.set_xlim([-2,5])
ax1.set_ylim([-2,5])
ax1.set_zlim([-2,5])


#Graph Two
scatterDatDF = pd.concat(scatterPlotData) #Full data frame version of the generated data
sortedScatterDat = scatterDatDF.sort_values(by = 'time2Hit', ascending = False) #Puts the data into a descending order
sortedScatterDat.plot.scatter(x= 'xSpot', y = 'ySpot', color = colorArray)

#Allows for all entries of the dataframe to be seen with a print statement
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(sortedScatterDat)


#Show both graphs
plt.show()


print(f"Muon path length is approximately, {LengthTravel} m\n") 
print("\nCherenkovAlgoTest.py is Finished Running \n")
