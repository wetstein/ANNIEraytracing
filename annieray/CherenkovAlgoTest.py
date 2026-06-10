#DO NOT USE THIS FILE FOR ANY SIMULATION WORK
#DO NOT USE THE CORRESPONDING DATA txt FILE FOR ANY ANALYSIS
#This file is only for testing the basic Cherenkov algorithm and generating data to compare against 

import numpy as np
import random as rng

print("CherenkovALgoTest.py is Running \n __________________________ ")
#Constants
c = 299792458 #Speed of light in m/s


#User set parameters
muonDirec = [0,0,1] # Direction of muon travel as a unit vector | for future implementation, choose muon direction and use it to find the length of track  
beta = 0.99 # Speed of the muon as a fraction of the speed of light
n =1.34 #Refractive index of medium
theta = np.arccos(1/(beta*n)) #Cherenkov angle in radians
LengthTravel = 10 #Distance from muon track to detector and muon track length in m (assuming the muon track is normal to detector plane) | for future use muon velocity to find length normal to detector
photonNum = 150 #Number of photons generated per cm along track




#Begin Cherenkov photon generation algorithm
i = LengthTravel*100 #convert 10 m to cm for use as number of iterations in for loop
photonDat = [] #initializing list to hold photon id, segment generation id, azimuthal angle, and time taken to reach end of the detector
netK = [-1] #Initializing list to hold total k iterations

#Considerations for error
photonTimeError = (LengthTravel/i) *(10**9) / (c/n) #Time error between photon generation events in ns


#This will write over any previous version of the file every time the code is run
with open("CherenkovAlgoTestData.txt", "w") as file:
    file.write("NOT FOR USE IN ANY PART OF THE CODE OR ANALYSIS \n Segment ID, Photon ID, Time to reach detector (ns), Azimuthal Angle (rad), X Coordinate (m), Y Coordinate (m)\n") # Write header to file   
    #Generating one photon every cm along the track
    for j in range(i+1): #+1 to include the end point of the track as well as the zeroth point where the first photon is generated (actually a nice case of 0 indexing working in favor)
        
        photonSegmentID = j+1 #This is the ID for the segment of the track where the photon is generated

        #Calculating travel time to detector, not yet accounting for time delay between photon generation events
        travelTime = (LengthTravel-(j)*(10**-2))*np.cos(theta)/(c/n)/(10**-9) # Time to reach detector in ns
        timeDelay = ((10**-2) / (beta*c))/(10**-9) # Time delay between photon generation in ns
        
        #There should be some sort of check against error here, but I am not sure what that should look like
        #if travelTime == timeDelay:
            #travelTime = 0 #This is for the end case where the photon is generated at the end of the track/at the detector itself | Could be wrong
        print('Photon Generation Event:',j, '\n') #Nice to have some way of measuring progress is happening
        travelTimeTotal = travelTime + timeDelay # Total travel time to detector in ns
        #Generate photonNum photons per cm along the track and perform any necessary calculations
        for k in range(photonNum): 
            #Generate photon ID
            photonID = netK[-1]+1 #Should be the only line required to assign the id

            #Finding spatial coordinates of photon at detector plane
            photonPhi = rng.uniform(0, 2 * np.pi) #Random azimuthal angle for photon emission
            photonRadial = travelTime * (c/n) *(10**-9) #Radial distance from center (muon track serves as origin) in m
            photonX = photonRadial * np.cos(photonPhi) #X coordinate of photon at detector plane
            photonY = photonRadial * np.sin(photonPhi) #Y coordinate of photon at detector plane

            #This is the total photon id, +1 has been added to k to avoid multiple zeros in the photon id
            netK.append(k+(j*150)) 


            photonDat.append((photonSegmentID, photonID, travelTimeTotal, photonPhi, photonX, photonY)) # Append photon id, time, and azimuthal angle to list
            
            #This must be in the k for loop in order to work properly 
            #Having a seperate file to view the output is the easiest way to check that the code is running properly
            file.write(f"{photonSegmentID}, {photonID}, {travelTimeTotal}, {photonPhi}, {photonX}, {photonY}\n") # Write photon id, time, and azimuthal angle to file



        
#Nice to have
print(f"Photon time error is approximately {photonTimeError:.6f}, ns \n") 
print("\nCherenkovAlgoTest.py is Finished Running \n")
