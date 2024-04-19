# # -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:15:27 2020

@author: LSPMC
"""


from lwpr import *
#from random import *
#from math import *
import numpy as np
#import os
import glob
import random
import time
import pdb

#### Load fine-tuning data ####
path = './'#ArtictoState_Mar9'#'training_data_files_nlsix_16' #'training_data_files_random_walk' #'tdf9range-2to2' #'training_data_files/backup0-1' #'training_data_files_n=5' #'E:/training_data_files' # 'training_data_files/backup0-1' #'training_data_files_n=5'

AMdata = np.zeros((0,7));
for file in sorted(glob.glob(path + '/' + 'AM*.csv')):
    loaded= np.loadtxt(file,delimiter=',')
    AMdata=np.concatenate((AMdata, loaded), axis=0)
    print(file)
    
TVdata = np.zeros((0,7));
for file in sorted(glob.glob(path + '/' + 'vocal_tract*.csv')):
    loaded= np.loadtxt(file,delimiter=',')
    TVdata=np.concatenate((TVdata, loaded), axis=0)
    print(file)
#model = LWPR('C:\KwangUCSF\FACTS\TrainingData\k4_v5.txt')

#('k2.xml')
Ntr = len(AMdata) #100000
step = 100

##### Load old LWPR Model upon which we will fine-tune ######
file = '_June22HRMaeda'
model = LWPR(str('ArtictoState' + file + '.txt'))

# Maybe initialize these things?
model.norm_in = np.array([1, 1, 1, 1, 1, 1],'double')
#model.norm_out= np.array([30,55,105,55,35,20],'double')

model.init_D = 2*np.eye(6) #10 #8 #7
model.update_D = True
model.diag_only = False
model.penalty = 1e-6
#tempa = 0.001*np.ones([6,6])
#tempa[2,:] = 0.00005

model.init_alpha = 0.01*np.ones([6,6]) #0.05 #0.1 #0.25
model.meta = True
model.meta_rate = 0.005 
model.w_gen = 0.2
model.w_prune =  0.7 #0.6 #0.7

##### Train the model ######
#Ntr = 1000
start = time.time()
for k in range(Ntr):
    #ind = random.randint(0, len(AMdata))
    #mse = 0
    #model.update(AMdata[ind][0:6],TVdata[ind])
    #pdb.set_trace()
    model.update(AMdata[k][0:6],TVdata[k])
    print(k)
    # if (k%step)==0:
    #     mse[ind] = np.mean((model.predict(AMdata[k][0:6])-TVdata[k])**2)
    #     #if mse[ind]>25:
    #     #    print("BIG ERROR")
    #     #    print(mse)
    #     #    print(model.predict(AMdata[k][0:6]))
    #     #    print(TVdata[k])
    #     ind = ind + 1
print(model.num_rfs)
end = time.time()
print("duration: "+  str(end - start))
print(model.num_rfs)
#np.savetxt('mse_nlsix.txt', mse, fmt='%d')
#print(model.predict_J(np.array([0. ,0. , 0. , 0. , 0. , 0.5 ])))
model.write_binary('ArtictoState_March2024HRMaeda.txt')










