# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 23:37:39 2023

@author: dnyan
"""

#%% 
## Initial Setup

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import matplotlib.colors
import time
import winsound

def FD_cordinates(xy):
    '''
    Function for updating flow accumulation values along the stream
    Parameters
    ----------
    xy : tupule
        function which returns cordinates of pixel where flow is flowing
        suppose FD for Pi is 1 then this will return a tuple with cordinates of pixel right of pi

    Returns
    tupule of cordinates of nerby pixel as per flow direction
    '''
    if FD[xy[0]][xy[1]]==1:
        return((xy[0],xy[1]+1))
    if FD[xy[0]][xy[1]]==2:
        return((xy[0]+1,xy[1]+1))
    if FD[xy[0]][xy[1]]==4:
        return((xy[0]+1,xy[1]))
    if FD[xy[0]][xy[1]]==8:
        return((xy[0]+1,xy[1]-1))
    if FD[xy[0]][xy[1]]==16:
        return((xy[0],xy[1]-1))
    if FD[xy[0]][xy[1]]==32:
        return((xy[0]-1,xy[1]-1))
    if FD[xy[0]][xy[1]]==64:
        return((xy[0]-1,xy[1]))
    if FD[xy[0]][xy[1]]==128:
        return((xy[0]-1,xy[1]+1))

def Get_cord(ix):
    i=int(ix/grid_size)
    j=ix%grid_size
    return((i,j))

def Get_id(i,j):
    return grid_size*i+j

#colourmap for figures
norm=plt.Normalize(0,1)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","blue"])

#%%
grid_size=250    # Should be even for initial setup
gama=0   # probability exponent

#flow direction and flow accumulation matrices
FD=np.zeros((grid_size,grid_size))
Facc=np.zeros((grid_size,grid_size))    
Area=np.zeros(grid_size*grid_size)      # single array

## Initial FD setup MAtrix should be of even size for symmetric even distribution 

# Assigning FD to diagonal pixels
for i in range(int(grid_size/2)):
            FD[i,i]=64
            FD[i,grid_size-1-i]=1
            FD[grid_size-1-i,i]=16
            FD[grid_size-1-i,grid_size-1-i]=4      
# Assigning for non-diagonal pixels
for j in range(int(grid_size/2)-1):
    FD[j,j+1:grid_size-1-j]=64
    FD[j+1:grid_size-1-j,grid_size-1-j]=1
    FD[grid_size-1-j,j+1:grid_size-1-j]=4
    FD[j+1:grid_size-1-j,j]=16

# Initial flow accumulation
Facc[:,:]=1

for i in range(grid_size):
    for j in range(grid_size):
        current_pi=(i,j)
        while(not(current_pi[0]==0 or current_pi[0]==grid_size-1 or current_pi[1]==0 or current_pi[1]==grid_size-1)):
            current_pi=FD_cordinates(current_pi)
            Facc[current_pi[0]][current_pi[1]]+=1
ixx=0            
for i in range(grid_size):
    for j in range(grid_size):
        Area[ixx]=Facc[i,j]
        ixx+=1

W=np.where(Facc>grid_size/5,1,0)
plt.imshow(W, cmap=cmap, origin='lower',extent=[0,grid_size,0,grid_size])
plt.title("Initial stream Network", fontsize=8)
plt.savefig("Initial Network.png", bbox_inches='tight',dpi=300)

#%% Evolution

n=100
EE=[]
tic=time.time()

# Initial EE
### ENERGY EXPENDITURE ###
li=np.zeros((grid_size,grid_size))
    
for i in range(grid_size):
    for j in range(grid_size):
        if (FD[i][j]==1 or FD[i][j]==4 or FD[i][j]==16 or FD[i][j]==64):   
            li[i][j]=1
        else :
            li[i][j]=np.sqrt(2)
    
#Get flow acc matrix
#formula for total energy expenditure
energy=np.multiply(np.power(Facc,0.5),li)
total_energy=np.sum(energy)
EE.append(total_energy)

for it in range(n):
    # Creating Initial Conditions
    Pot_ids=[]

    Label=dict()
    #All pixel labelled as unassigned "n"

    for i in range(grid_size):
        for j in range(grid_size):
            Label.update({(i,j):"n"})
            
    #Updating initial labels of all surrounding pixels to assigned or "y"
    for i in range(grid_size):
        Label.update({(i,0):"y"})
        Label.update({(0,i):"y"})
        Label.update({(grid_size-1,i):"y"})
        Label.update({(i,grid_size-1):"y"})

    #Updating labels for potential first inside layer pixels
    for i in range(grid_size-2):
        Label.update({(i+1,1):"p"})
        Label.update({(1,i+1):"p"})
        Label.update({(grid_size-2,i+1):"p"})
        Label.update({(i+1,grid_size-2):"p"})
    
    #Now updating lists first only border elements without corner elements 
    #And later corner elements
       
    #For list of potential pixels in first step
    for i in range(grid_size-4):    
        Pot_ids.append(grid_size*(2+i)+1)
        Pot_ids.append(grid_size*1+i+2)
        Pot_ids.append(grid_size*(grid_size-2)+i+2)
        Pot_ids.append(grid_size*(i+2)+grid_size-2)
    
    Pot_ids.append(grid_size*1+1)
    Pot_ids.append(grid_size*(grid_size-2)+1)
    Pot_ids.append(grid_size*1+grid_size-2)
    Pot_ids.append(grid_size*(grid_size-2)+grid_size-2)
        
    #Surrounding pixel and their correcsponding flow direction values
    ij=[(0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128)]

    while(len(Pot_ids)>0):
        ## Choosing Potential Pixel
        pot_facc=Area[Pot_ids]
        facc_gama=pot_facc**gama
        cum_facc=np.cumsum(facc_gama)
        
        rand_n=random.randint(1,int(cum_facc[-1]))
        
        pi_id=-1
        for j in range(len(Pot_ids)):
            if rand_n<=cum_facc[j]:
                pi_id=Pot_ids[j]
                break
            
        pi=Get_cord(pi_id)
        
        # before updating FD, saving old next pixel for updating Facc of older stream later  
        next_old_pi=FD_cordinates(pi)
        FD_old=FD[pi[0],pi[1]]
        # Choose flow direction directly by choosing max Facc surrounding 
        Values=[]
        ids=[]
        for ix,y in enumerate(ij):
            if Label[(pi[0]+y[0],pi[1]+y[1])]=='y':
                Values.append(Facc[pi[0]+y[0]][pi[1]+y[1]])
                ids.append(y[2])
        
        max_value = max(Values)
        max_indices = [index for index, value in enumerate(Values) if value == max_value]
        # Select random index among the maximum values
        random_index = random.choice(max_indices)
        FD[pi[0]][pi[1]]=ids[random_index]
        
        #Now updating labels and list for pi 
        #Potential.remove((pi))
        Pot_ids.remove(pi_id)
        Label.update({pi:"y"})
          
        # #Updating labels and lists for and surrounding pixels of pi
        for x in ij:
            if Label[(pi[0]+x[0],pi[1]+x[1])]=='n':
                Label[(pi[0]+x[0],pi[1]+x[1])]='p'
                Pot_ids.append(grid_size*(pi[0]+x[0])+pi[1]+x[1])
                   
    #Now updating flow accumulation value for pi and then along the stream if flow direction is updated or keep it same
    
        if not(FD_old==FD[pi[0],pi[1]]):
        # Here we have to add new Facc values to newer stream and subtract Facc from older stream based on older flow direction
            current_pi=pi
            Facc_pi=Facc[pi[0],pi[1]]
            Facc[next_old_pi[0],next_old_pi[1]]-=Facc_pi
            Area[grid_size*next_old_pi[0]+next_old_pi[1]]-=Facc_pi
        # Subtracting this Facc from older streams
            while(not(next_old_pi[0]==0 or next_old_pi[0]==grid_size-1 or next_old_pi[1]==0 or next_old_pi[1]==grid_size-1)):
                next_old_pi=FD_cordinates(next_old_pi)
                Facc[next_old_pi[0],next_old_pi[1]]-=Facc_pi
                Area[grid_size*next_old_pi[0]+next_old_pi[1]]-=Facc_pi
        # Adding Facc to new stream
            while(not(current_pi[0]==0 or current_pi[0]==grid_size-1 or current_pi[1]==0 or current_pi[1]==grid_size-1)):            
                current_pi=FD_cordinates(current_pi)
                Facc[current_pi[0]][current_pi[1]]+=Facc_pi
                Area[grid_size*current_pi[0]+current_pi[1]]+=Facc_pi
 
    ### ENERGY EXPENDITURE ###
    li=np.zeros((grid_size,grid_size))
        
    for i in range(grid_size):
        for j in range(grid_size):
            if (FD[i][j]==1 or FD[i][j]==4 or FD[i][j]==16 or FD[i][j]==64):   
                li[i][j]=1
            else :
                li[i][j]=np.sqrt(2)
        
    #Get flow acc matrix
    #formula for total energy expenditure
    energy=np.multiply(np.power(Facc,0.5),li)
    total_energy=np.sum(energy)
    EE.append(total_energy)
    
    if it%5==0:
        W=np.where(Facc>grid_size/5,1,0)
        plt.imshow(W, cmap=cmap, origin='lower',extent=[0,grid_size,0,grid_size])
        plt.title(f"stream Network{it} \u03B3={gama}", fontsize=8)
        plt.savefig(f"stream Network{it} \u03B3={gama}.png", bbox_inches='tight',dpi=300)

toc=time.time()
time_elapsed=toc - tic

duration=1000 #milliseconds
freq=440 #Hz
winsound.Beep(freq, duration)

#%%
plt.plot(EE,"o",markersize=3)
plt.plot(EE)
plt.title("Energy Expenditure with Iteartions")
plt.ylabel("EE")
plt.xlabel("Iterations")
plt.savefig("EE Evolution_100it.png",bbox_inches="tight",dpi=300)
plt.savefig("EE Evolution_100it.svg",bbox_inches="tight",dpi=300)  