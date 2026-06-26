
import numpy as np
import random as rng
import math



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


#User set default parameters
beta = 0.999999 # Speed of the muon as a fraction of the speed of light
n =1.34 #Refractive index of medium
trackPrec = 2 #control the step size of the track (currently cm steps along track)
photonNum = 150 #Default number of photons generated per cm along track 
LengthTravel = 4
defaultWavelength = 1234 #Just here because the original file has a singular specified wavelength
def generate_cherenkov_photons(
    muon_pos,
    muonDirec,
    photonNumUnused,
    thetaC: float = 0.73,
    rng: np.random.Generator | None=None, #Only here because the original has it here
    wavelength: float = defaultWavelength #Only here because the original has it here
    ) -> tuple[np.ndarray,np.ndarray]:
    #If no start time is given make it a 
    if len(muon_pos) != 4:
        muonStart = tuple([muon_pos[0],muon_pos[1],muon_pos[2],0])
    else:
        muonStart = muon_pos

    i = int(LengthTravel*10**trackPrec) #convert m to the desired precision for use as number of iterations in for loop
    photonDat = [] #initializing list to hold photon id, segment generation id, global start position, and global direction vectors
    netK = [-1] #Initializing list to hold total k iterations
    createTime = [] #Initializing a list that will track run time and photon creation time
    muonSpeed = beta*c #Speed of muon in m/s
    #Calculting time delay and recording the generation time of each photon
    timeDelay = ((10**(-trackPrec)) / (beta*c))/(10**-9) # Time delay between photon generation in ns
    
    muonPath = tuple(x*LengthTravel for x in muonDirec) #specified muon path

    for j in range(i+1): #+1 to include the end point of the track as well as the zeroth point where the first photon is generated (actually a nice case of 0 indexing working in favor)        
        photonSegmentID = j+1 #This is the ID for the segment of the track where the photon is generated

        #Cords of muon in the global frame 
        posX = (muonDirec[0])*j*(10**-trackPrec) +muonStart[0]#X cord in m
        posY = (muonDirec[1])*j*(10**-trackPrec)+muonStart[1] #Y cord in m
        posZ = (muonDirec[2])*j*(10**-trackPrec)+muonStart[2] #Z cord in m
        
        timeMuon = j*timeDelay+muonStart[3]*(10**-9) #This is the muon time (s)
        #print('Photon Generation Event:',j, '\n') #Nice to have some way of measuring progress is happening
        
        #Generate photonNum photons per unit length along the track and perform any necessary calculations
        for k in range(photonNum): 

            #Run spatial probability of photon generation
            photonProb = rng.uniform(0,1)
            photonPos = j*(10**-trackPrec)+photonProb*(10**-trackPrec) #Position of photon along track in m
            
            #Calculate photon creation time in ns from some given start time | Remember the start time is defined in seconds
            createTime.append(((photonPos)/(muonSpeed)+muonStart[3])*(10**9))

            #Generate photon ID
            photonID = netK[-1]+2 #Should be the only line required to assign the id

            #Wavelength placeholder
            photWave = 234 #placeholder
            
            photonX = photonPos * muonDirec[0] * 1000 + muonStart[0]  # mm
            photonY = photonPos * muonDirec[1] * 1000 + muonStart[1]  # mm
            photonZ = photonPos * muonDirec[2] * 1000 + muonStart[2]  # mm

            #Finding directional unit vector of photon compared in global frame
            photonAlpha = rng.uniform(0, 2 * np.pi) #Random azimuthal angle for photon emission
            xDirec, yDirec, zDirec = getPhotonVec(muonDirec,thetaC,photonAlpha)

            #This is the total photon id, +1 has been added to k to avoid multiple zeros in the photon id
            netK.append(k+(j*150)) 

            photonDat.append((photonSegmentID, photonID, createTime[-1], xDirec, yDirec, zDirec, photonX, photonY, photonZ, photWave)) # Append photon id, time, direction, and position
    #Initialize lists and assign data
    origins = np.empty((photonNum*(i+1),3),dtype=np.float32) 
    directions = np.empty((photonNum*(i+1),3),dtype=np.float32) 
    
    origins[:,0] = [row[6] for row in photonDat]
    origins[:,1] = [row[7] for row in photonDat]
    origins[:,2] = [row[8] for row in photonDat]

    directions[:,0] = [row[3] for row in photonDat]
    directions[:,1] = [row[4] for row in photonDat]
    directions[:,2] = [row[5] for row in photonDat]
    return origins, directions
#print('Cherenkov.py is finished')
