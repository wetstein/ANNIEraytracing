#DO NOT USE THIS FILE FOR ANY SIMULATION WORK
#DO NOT USE THE CORRESPONDING DATA txt FILE FOR ANY ANALYSIS
#This file is only for testing the basic Cherenkov algorithm and generating data to compare against 

import numpy as np
import random as rng


#Constants
c = 299792458 #Speed of light in m/s


#User set parameters
muonDirec = [0,1,0] # Direction of muon travel as a unit vector | not needed for basic algorithm/not current in use
beta = 0.99 # Speed of the muon as a fraction of the speed of light
n =1.34 #Refractive index of medium
theta = np.arccos(1/(beta*n)) #Cherenkov angle in radians
LengthTravel = 10 #Perpendicular distance from muon track to detector and muon track length in m
photonNum = 150 #Number of photons generated per cm along track


#Begin Cherenkov photon generation algorithm
i = LengthTravel*100 #convert 10 m to cm for use as number of iterations in for loop
photonDat = [] #initializing list to hold photon id, segment generation id, azimuthal angle, and time taken to reach end of the detector
photonSegDat = [] #initializng list to hold photon id and the corresponding azimuthal angle
netK = [0] #Initializing list to hold total k iterations

with open("CherenkovAlgoTestData.txt", "w") as file:
    file.write("Segment ID, Photon ID, Time to reach detector (ns), Azimuthal Angle (rad)\n") # Write header to file   
    #Generating one photon every cm along the track
    for j in range(i+1): #+1 to include the end point of the track as well as the zeroth point where the first photon is generated (actually a nice case of 0 indexing working in favor)
        #Generate photon id 
        photonSegmentID = j+1
        travelTime = (LengthTravel-(j)*(10**-2))*np.cos(theta)/(c/n)/(10**-9) # Time to reach detector in ns
        timeDelay = ((10**-2) / (beta*c))/(10**-9) # Time delay between photon generation in ns
        if travelTime == timeDelay:
            travelTime = 0 #This is for the end case where the photon is generated at the end of the track/at the detector itself

        for k in range(photonNum): #Generate photonNum photons per cm along the track
            #Generate photon id within segment
            if netK[-1] == 0:
                photonID = 1
            else:
                photonID = netK[-1] #Should be the only line required to assign the id
            photonPhi = rng.uniform(0, 2 * np.pi) #Random azimuthal angle for photon emission
            photonRadial = travelTime * (c/n) #Radial distance from center (muon track serves as origin)
            photonX = photonRadial * np.cos(photonPhi) #X coordinate of photon at detector plane
            photonY = photonRadial * np.sin(photonPhi) #Y coordinate of photon at detector plane

            netK.append((k+1)*(j+1)) # Append k iteration to list
            #print('Photon Segment ID:', photonSegmentID, 'Photon ID:', photonID, 'Time to reach detector (ns):', travelTime, 'Azimuthal Angle (rad):', photonPhi) # Print photon id, time, and azimuthal angle to console
            photonSegDat.append((photonID, photonPhi, photonX, photonY)) # Append photon id and azimuthal angle to list

        
        #The total travel time must be calculated after finding the x/y coordinates, otherwise the time delay component of travelTime would overshoot the total distance traveled
        travelTime += timeDelay # Add time delay to travel time for each photon


        photonDat.append((photonSegmentID, photonID, travelTime, photonPhi)) # Append photon id, time, and azimuthal angle to list
        file.write(f"{photonSegmentID}, {photonID}, {travelTime}, {photonPhi}\n") # Write photon id, time, and azimuthal angle to file


print("CherenkovAlgoTest.py is Finished Running")
