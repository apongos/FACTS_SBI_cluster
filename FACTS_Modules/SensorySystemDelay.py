
#from .util import string2dtype_array
#import numpy as np
import global_variables as gv
import pdb

class SensorySystemDelay():
    def __init__(self,sensory_configs):
        # read in config data
        if sensory_configs:
            self.Auditory_sensor_delay = float(sensory_configs['Auditory_delay'])
            self.Somato_sensor_delay = float(sensory_configs['Somato_delay'])
            #nAuditory = int(sensory_configs['nAuditory'])
            #norms_Auditory = string2dtype_array(sensory_configs['norms_Auditory'],'float32')#will have to tune later KSK 1/20/2021
            #norms_AADOT = string2dtype_array(sensory_configs['norms_AADOT'],'float32') #will have to tune later KSK 1/20/2021
            # set class data
            #self.R_Auditory = 1e0*Auditory_sensor_scale*np.ones(nAuditory)*norms_Auditory
            #self.R_Somato = 1e0*Somato_sensor_scale*np.ones(gv.a_dim*2)*norms_AADOT
            #print(self.R_Auditory)
    def run(self,ms_frm,i_frm,Auditory_sense,Somato_sense,Auditory_record,Somato_record):
        #Simulating delay by adding the current plant sensory info in the higher index in _record
        #which will be read some time later by the estimators)

        #print(Auditory_record.shape)
        #print(Somato_record.shape)
        #print(Auditory_record[5])
        #print(int(self.Auditory_sensor_delay/ms_frm))
        try:
            Auditory_record[i_frm + int(self.Auditory_sensor_delay/ms_frm),] = Auditory_sense
            Somato_record[i_frm + int(self.Somato_sensor_delay/ms_frm),] = Somato_sense
            
            Auditory_sense = Auditory_record[i_frm,]
            Somato_sense = Somato_record[i_frm,]

            return  Auditory_sense, Somato_sense, Auditory_record, Somato_record
        except Exception as e:
            print(e)
            pdb.set_trace()


        