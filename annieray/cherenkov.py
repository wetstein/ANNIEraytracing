
import numpy as np
import math

print("CherenkovAlgoTest.py is Running \n __________________________ ")

#Constants
c = 299792458 #Speed of light in m/s

#Rounding Function
def truncate(number, decimals = 0):
    factor = 10 ** decimals

    return math.trunc(number*factor)/factor

#Takes in angles to make coefficients for x,y,z components
#Expects a multi row array for alpha
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
#Expects a multi row array for alpha
def getPhotonVec(vector,b1,b2,thetaC,alpha):
    lam, kap,eta = getComp(thetaC,alpha)
    normVector = np.asarray(vector) / np.linalg.norm(vector)
    vector = (lam[:, np.newaxis] * b1
              + kap[:, np.newaxis] * b2
              + eta * normVector)

    return vector


#User set default parameters
beta = 0.999999 # Speed of the muon as a fraction of the speed of light
n =1.34 #Refractive index of medium
trackPrec = 2 #control the step size of the track (currently cm steps along track)
LengthTravel = 4
defaultWavelength = 1234 #Just here because the original file has a singular specified wavelength
def generate_cherenkov_photons(
    muon_pos,
    muonDirec,
    photons_per_cm: int = 150,
    thetaC: float = 0.73,
    rng: np.random.Generator | None=None,
    wavelength: float = defaultWavelength,
    track_length: float = 4.0,
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    if len(muon_pos) != 4:
        muonStart = tuple([muon_pos[0],muon_pos[1],muon_pos[2],0])
    else:
        muonStart = muon_pos

    n_steps = int(track_length * 10**trackPrec) + 1
    muonSpeed = beta*c

    b1,b2 = getBasis(muonDirec)

    muonArray = rng.uniform(0, track_length, size=n_steps * photons_per_cm)
    muonArray.sort() #Sorts the array in ascending order

    createTime = (muonArray / muonSpeed) * 10**9 + muonStart[3] * 10**9 #Measured in ns

    photonAlpha = rng.uniform(0, 2 * np.pi, size=n_steps * photons_per_cm)

    photStartPosArray = muonArray[:, np.newaxis] * np.array(muonDirec)
    photonDirec = getPhotonVec(muonDirec, b1, b2, thetaC, photonAlpha)

    origins = np.empty((n_steps * photons_per_cm, 3), dtype=np.float32)
    directions = np.empty((n_steps * photons_per_cm, 3), dtype=np.float32)

    origins[:,0] = 1000 * photStartPosArray[:,0] + muonStart[0]
    origins[:,1] = 1000 * photStartPosArray[:,1] + muonStart[1]
    origins[:,2] = 1000 * photStartPosArray[:,2] + muonStart[2]

    directions[:,0] = photonDirec[:,0]
    directions[:,1] = photonDirec[:,1]
    directions[:,2] = photonDirec[:,2]

    return origins, directions, createTime.astype(np.float32)
