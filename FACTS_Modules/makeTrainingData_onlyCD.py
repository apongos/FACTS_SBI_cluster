# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:08:53 2020

@author: Kwang Seob Kim, Jessica Gaines
"""

import numpy as np
import time
from scipy.interpolate import interp1d
import relokate as rlk
import matplotlib.pyplot as plt

def find_artic_params(internal_x,internal_y,external_x,external_y,pCon_x,pCon_y,mm2cm,alpha,ref,area,plot=True,verbose=True):

    ## need to import angle data here too
    inter = np.array([internal_x,internal_y]).T
    if inter[-1].all() == inter[-2].all():
        inter = inter[0:-1]
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(inter, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    
    interpolator =  interp1d(distance, inter, kind='cubic', axis=0)
    interpolated_points = interpolator(alpha)
    
    Tx = interpolated_points[0:len(alpha),0]-ref[0]
    Ty = interpolated_points[0:len(alpha),1]-ref[1]

    rlk_start = time.time()
    print("before relokate")
    #plt.scatter(pCon_x, pCon_y)
    # plt.figure()
    #plt.scatter(Tx, Ty)
    #plt.show()
    newTx, newTy, err = rlk.res2center(1401,1401,1401,pCon_x,pCon_y,Tx,Ty)
    print(f" {newTx} {newTy}")
    rlk_end = time.time()
    if verbose:
        print(rlk_end - rlk_start)
        print(np.mean(err))
        print(np.max(err))
        
    #for x in range(0, 1400):
    #area[x] = np.linalg.norm([pCon_x[x*10]-newTx[x*10], pCon_y[x]-newTy[x]])

    #TT40-80
    #TB80-180
    #TTind = np.where(area == min(area[0:2801]))[0][0]
    #TBind = np.where(area == min(area[4302:14000]))[0][0]
    #TTCL=(TTind+4000)/100
    #TTCD=min(area[0:2801])*mm2cm
    #TBCL=(TBind+4000)/100
    #TBCD=min(area[4302:14000])*mm2cm    
    
    CD_Den = np.linalg.norm([pCon_x[290]-newTx[29], pCon_y[290]-newTy[29]])*mm2cm #42.9 deg
    CD_Alv = np.linalg.norm([pCon_x[1800]-newTx[180], pCon_y[1800]-newTy[180]])*mm2cm #58.0
    CD_Pal = np.linalg.norm([pCon_x[5240]-newTx[524], pCon_y[5240]-newTy[524]])*mm2cm #92.4
    CD_Vel = np.linalg.norm([pCon_x[8110]-newTx[811], pCon_y[8110]-newTy[811]])*mm2cm #121.0
    CD_Pha = np.linalg.norm([pCon_x[13980]-newTx[1398], pCon_y[13980]-newTy[1398]])*mm2cm #179.8

    ##LA
    LA = (external_y[28] -internal_y[28])*mm2cm
    #print('LA = ' + str(LA))

    ##LP
    LP = (external_x[27] -external_x[28])*mm2cm
  
  #plt.quiver(*origin, PV[:,0], PV[:,1], angles='xy', scale_units=
    return [CD_Den,CD_Alv,CD_Pal,CD_Vel,CD_Pha,LA,LP]

    
#find_artic_params(internal_x,internal_y,external_x,external_y,palateCon,plot=True,verbose=True)
