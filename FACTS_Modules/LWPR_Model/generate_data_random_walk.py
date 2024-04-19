# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 02:46:27 2020

@author: Jessica Gaines
"""

import numpy as np
import time
import maeda as mda
from makeTrainingData import find_artic_params
#from makeTrainingData import find_artic_params
import os
import pandas as pd

full_start = time.time()

write_path = 'training_data/11_3_2023'
if not os.path.isdir(write_path):
    try:
        os.mkdir(write_path)
    except OSError:
        print('Creation of directory %s failed.' %write_path)

def is_valid_formant(formant):
    #if min(formant) > 100 and max(formant)<10000:
    if formant[0]>250 and formant[0]<900 and min(formant) > 0 and max(formant)<10000:
        return True
    else:
        return False

def is_valid_config(AM):
    return True

# set constants
max_step = 0.1
max_n_points_per_walk = 200
min_n_steps = 10
min_n_points = 12
AM_dims = 6
task_dims = 7
TC = np.array([1,1,0,0], 'float32')
PC = np.array([0.00114,35000,1600,1.5,300000], 'float32')
anc = 0.0
palateCon=np.loadtxt("palate_contour.txt")
row_dump = 5000
print_rate = 5
total_points = 0
n_walks = 0
artic_df = pd.DataFrame(columns=['jaw','tongue','shape','apex','lip_ht','lip_pr'])
formant_df = pd.DataFrame(columns=['F1','F2','F3','F4','F5'])
task_df = pd.DataFrame(columns=['TT_Den','TT_Alv','TB_Pal','TB_Vel','TB_Pha','LPRO','LA'])
walk_df = pd.DataFrame(columns=['Walk_N'])
# repeatability
np.random.seed(100)
while total_points < min_n_points:
    points_in_walk = 0
    # find starting point
    start = time.time()
    AM = np.zeros(7,dtype="float32") 
    rand = np.random.random(size=AM_dims)
    scaled_rand = (rand * 6) - 3
    AM[0:AM_dims] = scaled_rand
    formant,internal_x,internal_y,external_x,external_y= mda.maedaplant(5,29,29,29,29,TC,PC,AM,anc)
    # initialize data structures
    artic_array = np.zeros((max_n_points_per_walk,AM_dims))
    formant_array = np.zeros((max_n_points_per_walk,5))
    task_array = np.zeros((max_n_points_per_walk,task_dims))
    # randomly walk
    while is_valid_formant(formant) and is_valid_config(AM) and points_in_walk < max_n_points_per_walk:
        # get task params
        task_params = find_artic_params(internal_x,internal_y,external_x,external_y,palateCon,plot=False,verbose=False)
        # append data to arrays
        artic_array[points_in_walk] = AM[0:AM_dims]
        formant_array[points_in_walk] = formant
        task_array[points_in_walk] = task_params
        # take a step
        rand = np.random.random(size=AM_dims)
        scaled_rand = (rand * max_step*2) - max_step
        AM[0:AM_dims] += scaled_rand
        #rand_n_artics = round(np.random.random()*6)
        #rand = np.random.normal(loc=0,scale=max_step/3,size=6)
        #pos = np.random.choice(range(6),rand_n_artics,replace=False)
        #AM[pos] += scaled_rand[pos]
        #AM = np.clip(AM,-3,3)
        formant,internal_x,internal_y,external_x,external_y= mda.maedaplant(5,29,29,29,29,TC,PC,AM,anc)
        points_in_walk = points_in_walk + 1
    if points_in_walk > min_n_steps:
        total_points = total_points + points_in_walk
        walk_array = np.ones(points_in_walk) * n_walks
        walk_array = walk_array.reshape(points_in_walk,1)
        artic_df = artic_df.append(pd.DataFrame(artic_array[:points_in_walk,:],columns=artic_df.columns))
        formant_df = formant_df.append(pd.DataFrame(formant_array[:points_in_walk,:],columns=formant_df.columns))
        task_df = task_df.append(pd.DataFrame(task_array[:points_in_walk,:],columns=task_df.columns))
        walk_df = walk_df.append(pd.DataFrame(walk_array,columns=walk_df.columns))
        #data = np.concatenate((artic_array[:points_in_walk,:],formant_array[:points_in_walk,:],task_array[:points_in_walk,:],walk_array),axis=1)
        end = time.time()
        print('Walk ' + str(n_walks) + ' completed with ' + str(points_in_walk) + ' points.  ' + str(round(end-start,2)) + ' s runtime.')
        n_walks += 1
        if n_walks % print_rate == 0:
            print(str(total_points) + ' out of ' + str(min_n_points) + ' points total.')
data_df = pd.concat([artic_df,formant_df,task_df,walk_df],axis=1)
print(data_df)
print(str(total_points) + ' rows saved. Total time: ' + str(time.time()-full_start))
data_df.to_csv(write_path + '/random_walks_data.csv')

'''
#if AM[0]<-2.5 or AM[0]>1.5:
if AM[0]<-1.7 or AM[0] >1.5:
    return False
elif AM[1] <-3.0 or AM[1] > 3.0:
    return False
#elif AM[2] <-1.0 or AM[2] > 2.5:
elif AM[2] < -1.0 or AM[2] > 2.0: 
    return False
elif AM[3] <-3.0 or AM[3] > 1.0:
    return False
elif AM[4] <-1.3 or AM[4] > 1.3:
    return False
elif AM[5] <-0.8 or AM[5] > 1.8:
    return False
else:
    return True
'''
'''
if max(AM) > 3:
    return False
if min(AM) < -3:
    return False
'''