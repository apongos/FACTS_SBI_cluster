# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 02:46:27 2020
@author: Jessica Gaines 
Kwang implemented 9/18/20 
"""
import multiprocessing 
import sys
import numpy as np
import time
import maeda as mda
from makeTrainingData_onlyCD import find_artic_params
import os

AM_random_bounds = [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (0-3.0, 3.0), (-3.0, 3.0)]
AM_starting_point_means = [-1.6, 0.01, 0.11, 0.07, 1.1, -0.02]
AM_starting_point_stds = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
#full_start = time.time()
def is_valid_formant(formant):
    if formant[0]<350 or formant[1]<1500:
        return False
    elif formant[0]>700 or formant[1]>2500:
        return False
    elif min(formant)<350:
        return False
    else:
        return True

def is_valid_AM(AM):
    if AM[0] < AM_random_bounds[0][0] or AM[0]> AM_random_bounds[0][1]:
        return False
    elif AM[1] < AM_random_bounds[1][0] or AM[1] > AM_random_bounds[1][1]:
        return False
    elif AM[2] < AM_random_bounds[2][0] or AM[2] > AM_random_bounds[2][1]:
        return False
    elif AM[3] < AM_random_bounds[3][0] or AM[3] > AM_random_bounds[3][1]:
        return False
   # elif AM[4] <-1.3 or AM[4] > 1.3:
    elif AM[4] < AM_random_bounds[4][0] or AM[4] > AM_random_bounds[4][1]: # Changed this based on plots
        return False
    elif AM[5] < AM_random_bounds[5][0] or AM[5] > AM_random_bounds[5][1]:
        return False
    else:
        return True

def random_AM_Init(means, stds):
    # Generate random numbers within the specified bounds for each index
    #random_numbers = [np.random.uniform(low=bound[0], high=bound[1]) for bound in bounds]
    random_numbers = [np.random.normal(loc=mean, scale=std) for mean, std in zip(means, stds)]

    # Convert the list of random numbers to a NumPy array
    random_array = np.array(random_numbers)
    return random_array


if len(sys.argv) < 2:
    print("Usage: python generate_data_files.py <number of values per input>")
    sys.exit(0)
try:
    n = int(sys.argv[1])
except:
    print("Usage: python generate_data_files.py <number of values per input>")
    sys.exit(0)

write_path = 'ArtictoState_March18HRMaeda_2024'
if not os.path.isdir(write_path):
    try:
        os.mkdir(write_path)
    except OSError:
        print('Creation of directory %s failed.' %write_path)

def generate_data_files_parallel(process):
    # set constants
    max_step = 0.025
    max_n_steps = 100
    min_n_points = 1000
    AM_dims = 6
    TC = np.array([1,1,0,0], 'float32')
    PC = np.array([0.00114,35000,1600,1.5,300000], 'float32')
    anc = 0.0
    palateCon=np.loadtxt("palate_contour.txt")
    pCon_x = palateCon[0,]
    pCon_y = palateCon[1,]
    mm2cm= 10
    alpha = np.linspace(0, 1, 50000)
    ref= np.array([10,10])
    area=np.zeros(1400,'double')
    #row_dump = 5000
    #print_update = 100
    # initialize data structures
    AM_array = np.zeros((min_n_points+max_n_steps,7))
    formant_array = np.zeros((min_n_points+max_n_steps,5))
    vocal_tract_array = np.zeros((min_n_points+max_n_steps,7))
    walk_start_marker = np.zeros(min_n_points+max_n_steps)
    i = 0
#    prev_i = 0
#    j = 0
    walks = 0
    #we do not want repeatability
    np.random.seed()
    while i < min_n_points:
        # find starting point
        #start = time.time()
        AM = np.zeros(7,dtype="float32") 
        #AM[0:6] = [-1.6, 0.01, 0.11, 0.07, 1.1, -0.02]
        #rand = np.random.random(size=AM_dims)
        scaled_rand = random_AM_Init(AM_starting_point_means, AM_starting_point_stds)
        AM[0:AM_dims] = scaled_rand
        print(AM)
        formant,internal_x,internal_y,external_x,external_y= mda.maedaplant(5,29,29,29,29,TC,PC,AM,anc)
        count = 0
        walk_start_marker[i] = 1
        # randomly walk
        while is_valid_formant(formant) and count <= max_n_steps and is_valid_AM(AM):
            # store valid params
            count += 1
            print("before find_artic_params")
            vocal_tract = find_artic_params(internal_x,internal_y,external_x,external_y,pCon_x,pCon_y,mm2cm,alpha,ref,area,plot=False,verbose=False)
            print(vocal_tract)
            AM_array[i] = AM
            formant_array[i] = formant
            vocal_tract_array[i] = vocal_tract
            i += 1
            if i > prev_i and i % row_dump == 0:
                np.savetxt(write_path + '/AM_' + str(process) + '_' + str(j) + '.csv',AM_array,delimiter=',')
                np.savetxt(write_path + '/formants_' + str(process) + '_'  + str(j) + '.csv',formant_array,delimiter=',')
                np.savetxt(write_path + '/vocal_tract_' + str(process) + '_' + str(j) + '.csv',vocal_tract_array,delimiter=',')
                print(str(i) + ' rows saved.')
                prev_i = i
                j += 1
            # take a step
            step = np.zeros(7,dtype="float32")
            rand = np.random.random(size=6)
            scaled_rand = (rand * max_step*2) - max_step
            step[0:6] = scaled_rand
            AM += step
            formant,internal_x,internal_y,external_x,external_y= mda.maedaplant(5,29,29,29,29,TC,PC,AM,anc)
            
        #end = time.time()
        if count > 0:
            walks += 1
            #print("walk complete")
            #print('Walk ' + str(walks) + ' completed with ' + str(count) + ' points.  ' + str(round(end-start,2)) + ' s runtime.')

    #walk_start_marker_full = np.zeros(i)
    #print(walk_start_marker_full)
    #print(walk_start_marker)
    #walk_start_marker_full[0:min_n_points] = walk_start_marker
    
    #print(AM_array)
    #i = i % row_dump
    AM_array = AM_array[0:i]
    formant_array = formant_array[0:i]
    vocal_tract_array = vocal_tract_array[0:i]
    walk_start_marker = walk_start_marker[0:i]
    np.savetxt(write_path + '/walk_start_marker_' + str(process+50) + '.csv',walk_start_marker,delimiter=',')            

    np.savetxt(write_path + '/AM_' + str(process+50) + '.csv',AM_array,delimiter=',')
    np.savetxt(write_path + '/formants_' + str(process+50) + '.csv',formant_array,delimiter=',')
    np.savetxt(write_path + '/vocal_tract_' + str(process+50) + '.csv',vocal_tract_array,delimiter=',')
    #print(str(i) + ' rows saved. Total time: ' + str(time.time()-full_start))


if __name__ == '__main__': 
    pool = multiprocessing.Pool() 
    #values = np.linspace(-3,3,n)
    #paramlist = list(itertools.product(values,values))
    paramlist = list(range(n))
    pool.map(generate_data_files_parallel, paramlist)
    
