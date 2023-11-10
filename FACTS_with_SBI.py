import sys
import torch
import global_variables as gv
import sys
import numpy as np
import matplotlib.pyplot as plt
import configparser
from FACTS_Modules.Model import model_factory
from FACTS_Modules.util import string2dtype_array
from FACTS_Modules.TADA import MakeGestScore
from facts_visualizations import single_trial_plots, multi_trial_plots
import os 
import pdb
#import seaborn as sns

from sbi.inference import infer, SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
import pickle
import scipy.io


def simulator(theta):
    ini='DesignC_AUKF_onlinepertdelay_SBI.ini'
    gFile='GesturalScores/KimetalOnlinepert2.G'
    config = configparser.ConfigParser()
    config.read(ini)
    # print('DEBUGG')
    # Replace the parameter value from ini file
    #pdb.set_trace()
    try:
        if theta.dim() > 1:
#             pdb.set_trace()
            #print(theta.numel())
            config['SensoryNoise']['Auditory_sensor_scale'] = str(theta[0][0].item())
            config['SensoryNoise']['Somato_sensor_scale'] = str(theta[0][1].item())
            
            config['TaskStateEstimator']['process_scale'] = str(theta[0][2].item())
            config['TaskStateEstimator']['covariance_scale'] = str(theta[0][3].item())
            config['ArticStateEstimator']['process_scale'] = str(theta[0][4].item())
            config['ArticStateEstimator']['covariance_scale'] = str(theta[0][5].item())

            config['SensoryDelay']['Auditory_delay'] = str(theta[0][6].item())
            config['SensoryDelay']['Somato_delay'] = str(theta[0][7].item())
            config['TaskStateEstimator']['cc_discount_from_delay'] = str(theta[0][8].item())

#             config['TaskStateEstimator']['estimated_auditory_delay'] = str(theta[0][6].item())
#             config['ArticStateEstimator']['estimated_somat_delay'] = str(theta[0][7].item())
            
        else:
            #pdb.set_trace()
            config['SensoryNoise']['Auditory_sensor_scale'] = str(theta[0].item())
            config['SensoryNoise']['Somato_sensor_scale'] = str(theta[1].item())
            
            config['TaskStateEstimator']['process_scale'] = str(theta[2].item())
            config['TaskStateEstimator']['covariance_scale'] = str(theta[3].item())
            config['ArticStateEstimator']['process_scale'] = str(theta[4].item())
            config['ArticStateEstimator']['covariance_scale'] = str(theta[5].item())

            config['SensoryDelay']['Auditory_delay'] = str(theta[6].item())
            config['SensoryDelay']['Somato_delay'] = str(theta[7].item())
            config['TaskStateEstimator']['cc_discount_from_delay'] = str(theta[8].item())

        # Note from Alvince, need to pass this in   
        config['TaskStateEstimator']['Auditory_delay']  = config['SensoryDelay']['Auditory_delay'] 
        config['ArticStateEstimator']['Somato_delay']  = config['SensoryDelay']['Somato_delay'] 

        # Note from Alvince, need to pass this in  for TSE    
        # config['TaskStateEstimator']['Auditory_delay']  = config['SensoryDelay']['Auditory_delay'] 
#             config['TaskStateEstimator']['estimated_auditory_delay'] = str(theta[6].item())
#             config['ArticStateEstimator']['estimated_somat_delay'] = str(theta[7].item())
    except:
        pdb.set_trace()

    model = model_factory(config)
    #pdb.set_trace()
    if 'MultTrials' in config.sections(): 
        ntrials = int(config['MultTrials']['ntrials'])
        target_noise= float(config['MultTrials']['Target_noise'])
    else: 
        ntrials = 1
        target_noise = 0

    #pdb.set_trace()
    gest_name = gFile.split('/')[-1].split('/')[-1]
    np.random.seed(100)
    GestScore, ART, ms_frm, last_frm = MakeGestScore(gFile,target_noise)
    
    # initialize vectors to monitor position at each timestep
    buffer_size_auditory = int(float(config['SensoryDelay']['Auditory_delay']) / 5)  # default used to be 20
    buffer_size_somato = int(float(config['SensoryDelay']['Somato_delay']) / 5)
    buffer_size = max(buffer_size_auditory, buffer_size_somato)
    
    x_tilde_delaywindow = np.full([buffer_size,gv.x_dim*2], np.nan) #a new variable that state estimators will have a partial access to
    a_tilde_delaywindow = np.full([buffer_size,gv.a_dim*2], np.nan) #a new variable that state estimators will have a partial access to


    x_tilde_record = np.full([last_frm+buffer_size,gv.x_dim*2], np.nan) #changed
    somato_record = np.full([last_frm+buffer_size,gv.a_dim*2], np.nan) #changed
    formant_record = np.full([last_frm+buffer_size,3], np.nan) #changed
    a_tilde_record = np.full([last_frm+buffer_size,gv.a_dim*2], np.nan) #changed
    formants_produced_record = np.full([last_frm,3], np.nan)

    x_tilde_record_alltrials = np.empty([ntrials,last_frm+buffer_size,gv.x_dim]) #changed
    somato_record_alltrials = np.full([ntrials,last_frm+buffer_size,gv.a_dim*2], np.nan) #changed
    formant_record_alltrials = np.full([ntrials,last_frm+buffer_size,3], np.nan) #changed
    shift_record_alltrials = np.full([ntrials,last_frm+buffer_size,3], np.nan) #changed
    formants_produced_record_alltrials = np.full([ntrials,last_frm,3], np.nan)

    
    a_tilde_record_alltrials = np.empty([ntrials,last_frm+buffer_size,gv.a_dim])
    a_dot_record_alltrials = np.empty([ntrials,last_frm+buffer_size,gv.a_dim])
    a_dotdot_record_alltrials = np.empty([ntrials,last_frm+buffer_size,gv.a_dim])
    predict_formant_record_alltrials = np.empty([ntrials,last_frm+buffer_size,3])

    #Check if catch trials (no perturbation) are specified in the config file
    if 'CatchTrials' in config.keys():
        catch_trials = string2dtype_array(config['CatchTrials']['catch_trials'], dtype='int')
        catch_types = string2dtype_array(config['CatchTrials']['catch_types'], dtype='int')
        if len(catch_trials) != len(catch_types):
            raise Exception("Catch trial and catch type lengths not matching, please check the config file.")
    else: catch_trials = np.array([])

    #Run FACTS for each trial
    for trial in range(ntrials):
        #print("trial:", trial)
        #Gestural score (task)
        GestScore, ART, ms_frm, last_frm = MakeGestScore(gFile,target_noise)         #this is similar with MakeGest in the matlab version

        # initial condition
        x_tilde_delaywindow[0] = string2dtype_array(config['InitialCondition']['x_tilde_init'],'float')
        a_tilde_delaywindow[0] = string2dtype_array(config['InitialCondition']['a_tilde_init'],'float')
        x_tilde_record[0] = string2dtype_array(config['InitialCondition']['x_tilde_init'],'float')
        a_tilde_record[0] = string2dtype_array(config['InitialCondition']['a_tilde_init'],'float')
        a_actual = string2dtype_array(config['InitialCondition']['a_tilde_init'],'float')
        model.artic_sfc_law.reset_prejb() #save the initial artic-to-task model.

        if trial in catch_trials: catch = catch_types[np.where(catch_trials==trial)[0][0]]
        else: catch = False
        #print("catch:", catch)
        
        for i_frm in range(last_frm): #gotta change this hardcoded number to aud delay later
            #model function runs FACTS by each frame
            x_tilde_delaywindow, a_tilde_delaywindow, a_actual, somato_record, formant_record, adotdot, y_hat, formants_produced = model.run_one_timestep(x_tilde_delaywindow, a_tilde_delaywindow, a_actual, somato_record, formant_record, GestScore, ART, ms_frm, i_frm, trial, catch)
            if (formants_produced == -1).all():
                formants_produced_record[i_frm:] = [-1, -1, -1]
                a_tilde_record[i_frm:] = np.tile(-10000, 12)
                x_tilde_record[i_frm:] = np.tile(-10000, 14)
                break
            else:
                a_tilde_record[i_frm+1] = a_tilde_delaywindow[0,:] #0 is always the most recnet current frame
                x_tilde_record[i_frm+1] = x_tilde_delaywindow[0,:] #0 is always the most recnet current frame
                formants_produced_record[i_frm] = formants_produced 
            
        
        predict_formant_record_alltrials[trial,] = y_hat
        formants_produced_record_alltrials[trial,] = formants_produced_record
        
        a_tilde_record_alltrials[trial,] = a_tilde_record[:,0:gv.a_dim]
        #a_dot_record[trial, ] = a_tilde[gv.a_dim:]
        x_tilde_record_alltrials[trial,] = x_tilde_record[:,0:gv.x_dim]
        formant_record_alltrials[trial,] = formant_record
        somato_record_alltrials[trial,] = somato_record
        
        model.task_state_estimator.update(catch)
        
        del x_tilde_record
        del a_tilde_record
        del formant_record
        del somato_record
    #pdb.set_trace()
    return formants_produced_record_alltrials[:,:,0].squeeze() 

def main(num_sim, num_workers, load_and_train):

    print(os.getcwd())
    print(os.listdir(os.curdir))

    # If environment variables are passed, use them
    if os.environ.get('ENV_NUM_WORKERS') is not None:
        num_workers = int(os.environ.get('ENV_NUM_WORKERS'))
    if os.environ.get('ENV_NUM_SIMULATIONS') is not None:
        num_sim = int(os.environ.get('ENV_NUM_SIMULATIONS'))
    if os.environ.get('ENV_LOAD_AND_TRAIN') is not None:
        load_and_train = os.environ.get('ENV_LOAD_AND_TRAIN')

    # Import real observed data
    singularity_path = '/home/FACTS' #'./' #'/wynton/home/nagarajan/apongos/FACTS_with_SBI/FACTS_SBI_output' #'/home/FACTS'
    # trial_cells_times = scipy.io.loadmat(singularity_path+'/sbi_resources/formant_pert_time_cleaned.mat')['time_matrix'].T
    # trial_cells_mat = scipy.io.loadmat(singularity_path+'/sbi_resources/formant_pert_data_cleaned.mat')['cleaned_matrix'].T # 1797 x 194 == trials by time
    # trial_cells_times = trial_cells_times[:,0:150]
    # trial_cells_mat = trial_cells_mat[:,0:150]

    # Alter the trial_cells so that they center to where FACTS centers
    #trial_cells_mat = trial_cells_mat + 531

    load_instead = False

    #Auditory
    # High - 0.04
    # Low - 0.0001
    #Somatosensory
    # High - 10
    # Low - .002
    # import your simulator, define your prior over the parameters
    #prior_mean = 0.002
    prior_min= [0.0001, 0.002, 0.0001, 0.0000001, 10e-12, 10e-12, 20, 20, 1]
    prior_mmax = [0.04, 1.0, 5, 5, 10e-6, 10e-4, 120, 120, 60] 
    #num_sim = 100000

    # prior = torch.distributions.Uniform(torch.as_tensor(mmin), torch.as_tensor(mmax) )
    prior = utils.torchutils.BoxUniform(torch.as_tensor(prior_min), torch.as_tensor(prior_mmax) )
    simulator2, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE(prior)

    if not load_and_train:

        inference = SNPE(prior)
        print(f'{num_sim}, {num_workers}, {load_and_train}')
        theta, x = simulate_for_sbi(simulator2, proposal=prior, num_simulations=num_sim, num_workers=num_workers)
        #parameter_posterior = infer(simulator, prior, method='SNPE', num_simulations=num_sim, num_workers=num_workers)
        # density_estimator = inference.append_simulations(theta, x).train()
        # posterior = inference.build_posterior(density_estimator)
        
        # Save the theta and x
        # Old file path /sbi_resources/ModelC_auditory_soma_noise_TSE_ASE_Delay_KWANG_70-120_20-70_posterior_{num_sim}.pkl'
        with open(singularity_path+f'/sbi_resources/ModelC_auditory_soma_noise_TSE_ASE_Delay_cc_reduction_from_delay_theta_x_{num_sim}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([theta, x], f)
        # Save the posterior
        # with open(singularity_path+f'/sbi_resources/ModelC_auditory_soma_noise_TSE_ASE_Delay_cc_reduction_from_delay_posterior_{num_sim}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump([posterior], f)
        
    else:
        with open(singularity_path + '/' + load_and_train, 'rb') as f:  # Python 3: open(..., 'wb')
            object_file = pickle.load(f)
            theta2, x2 = object_file
                
            # Run more simulations
            theta3, x3 = simulate_for_sbi(simulator2, proposal=prior, num_simulations=num_sim, num_workers=num_workers, density_estimator='mdn')
            
            # Append 
            theta4 = torch.cat((theta2, theta3))
            x4 = torch.cat((x2, x3))
            
            density_estimator = inference.append_simulations(theta4, x4).train(force_first_round_loss=True) # Look more into force_first_round_loss=True
            posterior2 = inference.build_posterior(density_estimator)

            # Save
            new_num_simulation = int(find_between( load_and_train, 'theta_x_', '.pkl' ))+num_sim
            with open(singularity_path+f'/sbi_resources/ModelC_auditory_soma_noise_TSE_ASE_Delay_cc_reduction_from_delay_theta_x_{new_num_simulation}.pkl', 'wb') as f2:  # Python 3: open(..., 'wb')
                pickle.dump([theta4, x4], f2)
                # Save the posterior
            with open(singularity_path+f'/sbi_resources/ModelC_auditory_soma_noise_TSE_ASE_Delay_cc_reduction_from_delay_posterior_{new_num_simulation}.pkl', 'wb') as f2:  # Python 3: open(..., 'wb')
                pickle.dump([posterior2], f2)

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Peroform SBI on FACTS parameters')

    parser.add_argument('--num_sim', type=int, required=False, default=10,
                        help='number of simulations')
    parser.add_argument('--num_workers', type=int, required=False, default=2,
                        help='number of cores to use')
    parser.add_argument('--load_and_train', type=str, required=False, default=None,
                        help='file name of pickle file that contains SBIs theta and x on top of which you can run more simulate. Example ./sbi_resources/ModelC_auditory_soma_noise_TSE_ASE_Delay_theta_x_1000.pkl')

    args = parser.parse_args()
    main(num_sim=args.num_sim, num_workers=args.num_workers, load_and_train=args.load_and_train)


