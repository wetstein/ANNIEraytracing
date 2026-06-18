#THIS CODE WAS JUST TO PROOF OUT PHOTON DIRECTION UNIT VECTORS TO PUT INTO THE CHERENKOVALGOTEST.PY


import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rng

print("CherenkovSymmetryTest.py is Running! \n")


def getComp(thetaC,alpha):
    lam = np.sin(thetaC)*np.cos(alpha)
    kap = np.sin(thetaC)*np.sin(alpha)
    eta = np.cos(thetaC)
    return lam,kap,eta

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
    normVector = np.linalg.norm(vector)
    vector = lam*b1 + kap*b2 + eta*normVector #Photon vector in global cords

    return vector


#User Parameters
thetaC = np.deg2rad(42)

#VECTOR ZERO
goalVec = np.array([3,2,7]) #Vector you want to align to
normGoal = goalVec / np.linalg.norm(goalVec)
photonVecArray = [normGoal] #Start with the normalized goal vector so that it is visible for comparison (this acts as the stand in for the muon direction)
colorArray = ['g'] #The first color is green so that the "pseudo-muon" direction vector will apear green on the plot

#VECTOR ONE
goalVec1 = np.array([3,2,7]) #Vector you want to align to
normGoal1 = goalVec1 / np.linalg.norm(goalVec1)
photonVecArray1 = [normGoal1] #Same as above but based on different vector
colorArray1 = ['y'] #The first color is green so that the "pseudo-muon" will apear yellow on the plot (I think this should make it yellow)

    #Generating random vectors based on goalVec with different alpha values 
for i in range(0,50):
    alpha = rng.uniform(0,2*np.pi)

    #Creating photon vector in global coordinates
    photonVecArray.append(getPhotonVec(normGoal, thetaC, alpha)) 
    colorArray.append('b')

    #Generating random vectors based on goalVec1 with different alpha values
for i in range(0,50):
    alpha = rng.uniform(0,2*np.pi)

    #Creating photon vector in global coordinates
    photonVecArray1.append(getPhotonVec(normGoal1, thetaC, alpha)) 
    colorArray1.append('r')
vectorDifArray = np.array([photonVecArray])-np.array([photonVecArray1])
print(vectorDifArray)

#Plotting here
origin = [0,0,0]
X,Y,Z = origin
U,V,W = zip(*photonVecArray)
U1,V1,W1 = zip(*photonVecArray1) #This is for the other vector
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.quiver(X,Y,Z,U,V,W, color = colorArray)
#ax.quiver(X,Y,Z,U1,V1,W1, color = colorArray1)
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_xlabel('X Axis')
ax.set_ylabel('y Axis')
ax.set_zlabel('Z Axis')
plt.show()





#Print statements to check if it is working as intended
print(f'The original vector is:{goalVec} \n')
print(f'The original unit vector is: {normGoal}\n')






print("CherenkovSymmetryTest.py is Done! \n")