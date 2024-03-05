"""
Created on February 2024

@author: jlittaye
"""

from sqlite3 import SQLITE_DROP_TEMP_TRIGGER
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.feature_extraction import image
import pandas as pd
import time
import os
import random
from math import *

# Source of Nitrogen of the model
global Q0
Q0 = 4+2.5+1.5+0


def function_NPZD(global_f, global_delta, N_0 = 4, P_0 = 2.5, Z_0 = 1.5, D_0 = 0, t_range = torch.arange(0, 365*5, 1/24), theta_values = torch.tensor([1., 1., 1., 1., 1., 1. ,1., 1., 1., 1.])) :
    """Function that simulates the states from background, forcings and parameters along a time tensor."""
    if len(theta_values.shape) == 1 :
        theta_values = torch.ones([global_f.shape[0], 10])
    nb_sample = global_f.shape[0]
    Kn_ref = theta_values[:, 0]*1
    Rm_ref = theta_values[:, 1]*2
    g_ref = theta_values[:, 2]*0.1
    lambda_Z_ref = theta_values[:, 3]*0.05
    epsilon_ref = theta_values[:, 4]*0.1
    alpha_ref = theta_values[:, 5]*0.3
    beta_ref = theta_values[:, 6]*0.6
    r_ref = theta_values[:, 7]*0.15
    phi_ref = theta_values[:, 8]*0.4
    Sw_ref = theta_values[:, 9]*0.1
    f0 = torch.index_select(global_f, 1, torch.round(t_range*24).int())
    delta = torch.index_select(global_delta, 1, torch.round(t_range*24).int())
    Vm = 1. # Maximum growth rate (per day)
    threshold = torch.tensor([0.01])
    dt = (t_range[1]-t_range[0])
    N_tensor = torch.zeros([nb_sample, len(t_range)])
    P_tensor = torch.zeros([nb_sample, len(t_range)])
    Z_tensor = torch.zeros([nb_sample, len(t_range)])
    D_tensor = torch.zeros([nb_sample, len(t_range)])
    
    N_tensor[:, 0] = N_0
    P_tensor[:, 0] = P_0
    Z_tensor[:, 0] = Z_0
    D_tensor[:, 0] = D_0

    gamma_N_ref_array = []
    zoo_graze_ref_array = []
    for idx in range(1, len(t_range)):
        t = idx - 1
        gamma_N_ref   = N_tensor[:, t] / (Kn_ref + N_tensor[:, t])
        zoo_graze_ref = Rm_ref * (1 - torch.exp(-lambda_Z_ref * torch.max(threshold, P_tensor[:, t]))) * torch.max(threshold, Z_tensor[:, t])
        gamma_N_ref_array.append(gamma_N_ref)
        zoo_graze_ref_array.append(zoo_graze_ref)
        N_tensor[:, idx] = dt * (-Vm*gamma_N_ref*f0[:, t]*torch.max(threshold, P_tensor[:, t]) + alpha_ref*zoo_graze_ref + epsilon_ref*P_tensor[:, t] + g_ref*Z_tensor[:, t] + phi_ref*D_tensor[:, t] + delta[:, t]*(Q0 - N_tensor[:, t])) + N_tensor[:, t]
        P_tensor[:, idx] = dt * (Vm*gamma_N_ref*f0[:, t]*torch.max(threshold, P_tensor[:, t]) - zoo_graze_ref - epsilon_ref*P_tensor[:, t] - r_ref*P_tensor[:, t] - delta[:, t]*P_tensor[:, t]) + P_tensor[:, t]
        Z_tensor[:, idx] = dt * (beta_ref*zoo_graze_ref - g_ref*Z_tensor[:, t] - delta[:, t]*Z_tensor[:, t]) + Z_tensor[:, t]  
        D_tensor[:, idx] = dt * (r_ref*P_tensor[:, t] + (1-alpha_ref-beta_ref)*zoo_graze_ref - phi_ref*D_tensor[:, t] - Sw_ref*D_tensor[:, t] - delta[:, t]*D_tensor[:, t]) + D_tensor[:, t]

    t = idx
    gamma_N_ref   = N_tensor[:, t] / (Kn_ref + N_tensor[:, t])
    zoo_graze_ref = Rm_ref * (1 - torch.exp(-lambda_Z_ref * torch.max(threshold, P_tensor[:, t]))) * torch.max(threshold, Z_tensor[:, t])
    gamma_N_ref_array.append(gamma_N_ref)
    zoo_graze_ref_array.append(zoo_graze_ref)

    return(t_range.repeat(nb_sample, 1), N_tensor, P_tensor, Z_tensor, D_tensor)


def Forcing_Gen(t_range, physical_bias, HF_var, file_size) :
    """Function that generates the physical forcings (light, mixing)"""
    Irradiance = torch.zeros(HF_var.shape[:-1]) # file_size, n_time, 
    f_min = 0.05
    Irradiance = ((physical_bias[:, 0]-f_min)/2)[:, None].repeat(1, len(t_range))*torch.cos(torch.pi + 20*torch.pi/365 + 2*torch.pi*t_range/365)[None, :].repeat(file_size, 1)+((physical_bias[:, 0]+f_min)/2)[:, None].repeat(1, len(t_range))+HF_var[:, :, 0]
    Irradiance = torch.max(torch.tensor([0]), Irradiance)
    Irradiance_averaged = torch.clone(Irradiance)
    Irradiance_averaged[:, 1:-1] = torch.mean(Irradiance.unfold(1, 3, 1), 2)
    Irradiance = Irradiance_averaged.unfold(1, 1, 3).expand(file_size, len(t_range)//3, 3).reshape(file_size, len(t_range))

    delta_min = 0.02
    Mixing = torch.zeros(HF_var.shape[:-1])
    Mixing = ((physical_bias[:, 1]-delta_min)/2)[:, None].repeat(1, len(t_range))*torch.cos(-physical_bias[:, 2][:, None].repeat(1, len(t_range)) + 20*torch.pi/365 + 2*torch.pi*t_range[None, :].repeat(file_size, 1)/365) + ((physical_bias[:, 1]+delta_min)/2)[:, None].repeat(1, len(t_range))+HF_var[:, :, 1]
    Mixing = torch.max(torch.tensor([0]), Mixing)
    Mixing_averaged = torch.clone(Mixing)
    Mixing_averaged[:, 1:-1] = torch.mean(Mixing.unfold(1, 3, 1), 2)
    Mixing = Mixing_averaged.unfold(1, 1, 3).expand(file_size, len(t_range)//3, 3).reshape(file_size, len(t_range))

    return Irradiance, Mixing


def Dataset_Gen(file_size, uncertainty_case) :
    """Function that generates the each dataset (BGC parameters, forcings and states)."""
    ti = time.time()
    global n_sum
    ## HF variability parameters of the physical forcings:
    if uncertainty_case == 0 :
        sigma_f = 0.0
        sigma_d = 0.0
        alpha_f = 0.0
        alpha_d = 0.0
    if uncertainty_case == 1 :
        sigma_f = 0.001
        sigma_d = 0.0005
        alpha_f = 0.99
        alpha_d = 0.95
    if uncertainty_case == 2 :
        sigma_f = 0.003
        sigma_d = 0.001
        alpha_f = 0.99
        alpha_d = 0.95
    if uncertainty_case == 3 :
        sigma_f = 0.001
        sigma_d = 0.0005
        alpha_f = 0.999
        alpha_d = 0.99
    ## Generation of the BGC parameters Theta:
    torch.random.seed()
    # torch.manual_seed(seed)
    Theta = torch.reshape(torch.tensor([random.uniform(0.8, 1.2) for i in range(10*file_size)]), (file_size, 10))
    
    ## Stats physical forcings :
    Forcing_stats_ref_value = torch.tensor([0.7, 0.1, torch.pi/5]) #fmax, deltamax, shift
    Forcing_stats = torch.tensor([random.uniform(0.8, 1.2) for sample in range(file_size*3)]).reshape(file_size, 3)*Forcing_stats_ref_value
    
    t_range = torch.arange(0, 365*5, 1/24)
    ## Generation of the HF variations of the physical forcings:
    HF_var = torch.zeros([file_size, len(t_range), 2, 2]) # Will contain the HF variations of the forcings
    seed = random.randint(0, 100000)
    torch.manual_seed(seed)
    for i_t in range(len(t_range)) :
        HF_var[:, i_t, 0, :] = HF_var[:, i_t, 0]*alpha_f + torch.normal(0, sigma_f, (1, file_size, 2))
        HF_var[:, i_t, 1, :] = HF_var[:, i_t, 1]*alpha_d + torch.normal(0, sigma_d, (1, file_size, 2))

    ## Generation of the HF variations of the physical forcings:
    Irradiance_True, Mixing_True = Forcing_Gen(t_range, Forcing_stats, HF_var[:, :, :, 0], file_size) ## True forcings
    Irradiance_False, Mixing_False = Forcing_Gen(t_range, Forcing_stats, HF_var[:, :, :, 1], file_size)  ## Forcings with uncertainty
    ## Generation of the states
    (x_1, N_1, P_1, Z_1, D_1) = function_NPZD(global_f = Irradiance_True, global_delta = Mixing_True, t_range = torch.arange(0, 365*3, 1/2), theta_values = Theta)
    States = torch.cat((x_1[:, 365*2*2:].unfold(1, 1, 1), N_1[:, 365*2*2:].unfold(1, 1, 1), P_1[:, 365*2*2:].unfold(1, 1, 1), Z_1[:, 365*2*2:].unfold(1, 1, 1), D_1[:, 365*2*2:].unfold(1, 1, 1)), dim = 2)
    t_range = time.time()-ti
    print("States generated within ", int(t_range//60), "minutes and ", int(t_range%60), "seconds.")

    return Theta, torch.cat((Irradiance_False[:, :, None], Mixing_False[:, :, None]), dim = 2), torch.cat((Irradiance_True[:, :, None], Mixing_True[:, :, None]), dim = 2), States, seed, Forcing_stats



###########################################################
#################  Generation Brut Data  ##################
###########################################################
## Only 100 data sets are generated for the DA-based method. It is not too heavy to go through sub_files.
name_save = "Generated_Datasets/DA/"
for case in [1, 2, 3] :
    ti_whole = time.time()
    name_file = "Case_"+str(case)+"/"
    if not os.path.isdir(name_save+name_file) :
        os.makedirs(name_save+name_file)        

    file_size = 100
    global n_sum
    n_sum = file_size

    Theta, False_forcings, True_forcings, States, seeds, Phi_stats = Dataset_Gen(file_size, case)

    torch.save(torch.tensor(seeds), name_save+name_file+"seeds.pt") # Seeds for the random variables if one wants to gets the exact same HF variations
    torch.save(Theta, name_save+name_file+"Theta.pt")
    torch.save(Phi_stats, name_save+name_file+"Phi_stats.pt")
    torch.save(False_forcings, name_save+name_file+"False_forcings.pt")
    torch.save(True_forcings, name_save+name_file+"True_forcings.pt")
    torch.save(States, name_save+name_file+"States.pt")


###########################################################################
################  Generation Dataset for the DA method  ###################
###########################################################################
    ti, tf, dt = 365*0, 365*1, 1/2
## Data are sampled (one sample per day)
    States_sampled_t = States.unfold(1, 1, int(1/dt))[:, :, :, 0] # Have been simulated with time-step of 1/2 day

    inputs = torch.zeros([False_forcings.shape[0], int((tf-ti)/7), 5, 1])
    outputs = torch.zeros([False_forcings.shape[0], int((tf-ti)/7), 4, (1+7)])
    for i in range(inputs.shape[0]) :
        for j in range(inputs.shape[1]) :
            inputs[i, j, :, 0] = torch.tensor([States_sampled_t[i, int((ti+j*7)), 0], States_sampled_t[i, int((ti+j*7)), 1], States_sampled_t[i, int((ti+j*7)), 2], States_sampled_t[i, int((ti+j*7)), 3], States_sampled_t[i, int((ti+j*7)), 4]])
            for k in range(outputs.shape[3]) :
                outputs[i, j, :, k] = torch.tensor([States_sampled_t[i, int((ti+j*7)+k), 1][None], States_sampled_t[i, int((ti+j*7)+k), 2][None], States_sampled_t[i, int((ti+j*7)+k), 3][None], States_sampled_t[i, int((ti+j*7)+k), 4][None]])
    dataset = DataLoader(TensorDataset(inputs, outputs), batch_size = 100, shuffle = False)
    torch.save(dataset, name_save + name_file + "/TrainLoader.pt")

## Generate the observation error matrix
    obs_noise_perc = 1. # Percentage of the observation variance for the noise
    mask_obs_error = torch.zeros([inputs.shape[0], inputs.shape[1], 4, 1])
    for dim_ch in range(4) :
        mask_obs_error[:, :, dim_ch, 0] += torch.normal(0., torch.std(inputs[:, :, dim_ch+1]).item()*obs_noise_perc/100, (inputs.shape[0], inputs.shape[1]))
    
    torch.save(mask_obs_error, name_save+name_file+"/Obs_matrix.pt")
    

###########################################################################
################  Generation Dataset for the NN method  ###################
###########################################################################
    
    mean_x = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_x.pt")
    mean_y = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_y.pt")
    std_x = torch.load(f"Generated_Datasets/NN/Case_{case}/std_x.pt")
    std_y = torch.load(f"Generated_Datasets/NN/Case_{case}/std_y.pt")

    Obs_noise = torch.zeros([States_sampled_t.shape[0], States_sampled_t.shape[1], 4]) ## Contains the observation error in the shape [n_data_set, time_length, n_ch]
    for i in range(52) :
        Obs_noise[:, i*7:(i+1)*7, :] = mask_obs_error[:, i, :, :7].moveaxis(1, 2)

    States_sampled = torch.clone(States).unfold(1, 1, int(1/dt))[:, :, 1:, 0] + Obs_noise

    False_forcings_sampled = False_forcings[:, 365*24*4:, :].unfold(1, 1, 24)[:, :, :, 0]
    True_forcings_sampled = True_forcings[:, 365*24*4:, :].unfold(1, 1, 24)[:, :, :, 0]

    ## Creates inputs : observations + physical forcing with uncertainty
    inputs_sampled = torch.cat((States_sampled, False_forcings_sampled), dim = 2)
    inputs_sampled_normalized = torch.zeros(inputs_sampled.shape)
    for ch in range(inputs_sampled_normalized.shape[2]) :
        inputs_sampled_normalized[:, :, ch] = (inputs_sampled[:, :, ch]-mean_x[ch])/std_x[ch]
    ## On normalise sortie (theta)
    outputs_normalized = (Theta-mean_y)/std_y

    DS_test_NN = TensorDataset(inputs_sampled_normalized.moveaxis(1, 2), outputs_normalized)
    torch.save(DS_test_NN, name_save+name_file+"/DS_test_NN")

    t_whole = time.time()-ti_whole
    print(f"Data for the case {case} have been generated in {int(t_whole//3600)}h:{int((t_whole%3600)//60)}min:{int(((t_whole%3600)%60))}sec.")