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


def Forcing_Gen(t_range, physical_bias, var_HF, file_size) :
    """Function that generates the physical forcings (light, mixing)"""
    Irradiance = torch.zeros(var_HF.shape[:-1]) # file_size, n_time, 
    f_min = 0.05
    Irradiance = ((physical_bias[:, 0]-f_min)/2)[:, None].repeat(1, len(t_range))*torch.cos(torch.pi + 20*torch.pi/365 + 2*torch.pi*t_range/365)[None, :].repeat(file_size, 1)+((physical_bias[:, 0]+f_min)/2)[:, None].repeat(1, len(t_range))+var_HF[:, :, 0]
    Irradiance = torch.max(torch.tensor([0]), Irradiance)
    Irradiance_averaged = torch.clone(Irradiance)
    Irradiance_averaged[:, 1:-1] = torch.mean(Irradiance.unfold(1, 3, 1), 2)
    Irradiance = Irradiance_averaged.unfold(1, 1, 3).expand(file_size, len(t_range)//3, 3).reshape(file_size, len(t_range))

    delta_min = 0.02
    Mixing = torch.zeros(var_HF.shape[:-1])
    Mixing = ((physical_bias[:, 1]-delta_min)/2)[:, None].repeat(1, len(t_range))*torch.cos(-physical_bias[:, 2][:, None].repeat(1, len(t_range)) + 20*torch.pi/365 + 2*torch.pi*t_range[None, :].repeat(file_size, 1)/365) + ((physical_bias[:, 1]+delta_min)/2)[:, None].repeat(1, len(t_range))+var_HF[:, :, 1]
    Mixing = torch.max(torch.tensor([0]), Mixing)
    Mixing_averaged = torch.clone(Mixing)
    Mixing_averaged[:, 1:-1] = torch.mean(Mixing.unfold(1, 3, 1), 2)
    Mixing = Mixing_averaged.unfold(1, 1, 3).expand(file_size, len(t_range)//3, 3).reshape(file_size, len(t_range))

    return Irradiance, Mixing


def Dataset_Gen(file_size, uncertainty_case, num_save = -1) :
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
    var_HF = torch.zeros([file_size, len(t_range), 2, 2]) # Will contain the HF variations of the forcings
    seed = random.randint(0, 100000)
    torch.manual_seed(seed)
    for i_t in range(len(t_range)) :
        var_HF[:, i_t, 0, :] = var_HF[:, i_t, 0]*alpha_f + torch.normal(0, sigma_f, (1, file_size, 2))
        var_HF[:, i_t, 1, :] = var_HF[:, i_t, 1]*alpha_d + torch.normal(0, sigma_d, (1, file_size, 2))

    ## Generation of the HF variations of the physical forcings:
    Irradiance_True, Mixing_True = Forcing_Gen(t_range, Forcing_stats, var_HF[:, :, :, 0], file_size) ## True forcings
    Irradiance_False, Mixing_False = Forcing_Gen(t_range, Forcing_stats, var_HF[:, :, :, 1], file_size)  ## Forcings with uncertainty
    ## Generation of the states
    (x_1, N_1, P_1, Z_1, D_1) = function_NPZD(global_f = Irradiance_True, global_delta = Mixing_True, t_range = torch.arange(0, 365*3, 1/2), theta_values = Theta)
    States = torch.cat((x_1[:, 365*2*2:].unfold(1, 1, 1), N_1[:, 365*2*2:].unfold(1, 1, 1), P_1[:, 365*2*2:].unfold(1, 1, 1), Z_1[:, 365*2*2:].unfold(1, 1, 1), D_1[:, 365*2*2:].unfold(1, 1, 1)), dim = 2)
    t_range = time.time()-ti
    print("States generated within ", int(t_range//60), "minutes and ", int(t_range%60), "seconds.")

    return Theta, torch.cat((Irradiance_False[:, :, None], Mixing_False[:, :, None]), dim = 2), torch.cat((Irradiance_True[:, :, None], Mixing_True[:, :, None]), dim = 2), States, seed, Forcing_stats



###########################################################
#################  Generation Brut Data  ##################
###########################################################
## Data is generated by "n_file = 5" packets of "file_size = 1000"
name_save = "Generated_Datasets/NN/"
for case in [1, 2, 3] :
    ti_whole = time.time()
    name_file = "Case_"+str(case)+"/"
    if not os.path.isdir(name_save+name_file) :
        os.makedirs(name_save+name_file)
    if not os.path.isdir("Loading_files") : # Contains transcient data
        os.makedirs("Loading_files")        

    n_file = 5
    file_size = 100
    global n_sum
    n_sum = n_file*file_size
    seeds = []
## Generating the sub_files one by one
    for nb_file in range(n_file) :
        Theta, False_forcings, True_forcings, States, seed, Phi_stats = Dataset_Gen(file_size, case, num_save = nb_file)
        seeds.append(seed)
        torch.save(Theta, "Loading_files/"+"Theta_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
        torch.save(Phi_stats, "Loading_files/"+"Phi_stats_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
        torch.save(False_forcings, "Loading_files/"+"False_forcings_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
        torch.save(True_forcings, "Loading_files/"+"True_forcings_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
        torch.save(States, "Loading_files/"+"States_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
        print("File n°", (nb_file+1), "/", n_file, " generated\n")
## Once the files are generated we concatenate them
    for nb_file in range(n_file) :
        if nb_file == 0 :
            Theta = torch.load("Loading_files/"+"Theta_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
            Phi_stats = torch.load("Loading_files/"+"Phi_stats_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
            False_forcings = torch.load("Loading_files/"+"False_forcings_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
            True_forcings = torch.load("Loading_files/"+"True_forcings_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
            States = torch.load("Loading_files/"+"States_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
            print("Shape of one state file: ", States.shape)
        else :
            Theta = torch.cat((Theta, torch.load("Loading_files/"+"Theta_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")), dim = 0)
            Phi_stats = torch.cat((Phi_stats, torch.load("Loading_files/"+"Phi_stats_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")), dim = 0)
            False_forcings = torch.cat((False_forcings, torch.load("Loading_files/"+"False_forcings_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")), dim = 0)
            True_forcings = torch.cat((True_forcings, torch.load("Loading_files/"+"True_forcings_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")), dim = 0)
            States = torch.cat((States, torch.load("Loading_files/"+"States_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")), dim = 0)
    print("Shape of all the states: ", States.shape)
## Either we keep the subfiles as we are saving the concatenated file or we remove them
    #     # os.remove("Loading_files/"+"Theta_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
    #     # os.remove("Loading_files/"+"Phi_stats_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
    #     # os.remove("Loading_files/"+"False_forcings_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
    #     # os.remove("Loading_files/"+"True_forcings_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
    #     # os.remove("Loading_files/"+"States_"+str(n_file*file_size)+"data_"+str(nb_file)+".pt")
    #     sys.stdout.write( "\rConcat file n°"+str(1+nb_file)+"/"+str(n_file) + ".")

    torch.save(torch.tensor(seeds), name_save+name_file+"seeds.pt") # Seeds for the random variables if one wants to gets the exact same HF variations
    torch.save(Theta, name_save+name_file+"Theta.pt")
    torch.save(Phi_stats, name_save+name_file+"Phi_stats.pt")
    torch.save(False_forcings, name_save+name_file+"False_forcings.pt")
    torch.save(True_forcings, name_save+name_file+"True_forcings.pt")
    torch.save(States, name_save+name_file+"States.pt")



#########################################################
################  Generation Dataset  ###################
#########################################################
## Data are sampled (one sample per day)
    States_sampled = States[:, :, 1:].unfold(1, 1, 2)[:, :, :, 0] # Have been simulated with time-step of 1/2 day
    False_forcings_sampled = False_forcings[:, 365*24*2:365*24*3, :].unfold(1, 1, 24)[:, :, :, 0] # Have been simulated with time-step of 1h
    True_forcings_sampled = True_forcings[:, 365*24*2:365*24*3, :].unfold(1, 1, 24)[:, :, :, 0] # Have been simulated with time-step of 1h

## Concatenate states + Forcings (with uncertainties)
    inputs_sampled = torch.cat((States_sampled, False_forcings_sampled), dim = 2)

## Generate the observation error matrix
    obs_noise_perc = 1. # Percentage of the observation variance for the noise
    mask_obs_error = torch.zeros([inputs_sampled.shape[0], inputs_sampled.shape[1], 4])
    for dim_ch in range(4) :
        mask_obs_error[:, :, dim_ch] += torch.normal(0., torch.std(inputs_sampled[:, :, dim_ch]).item()*obs_noise_perc/100, (inputs_sampled.shape[0], inputs_sampled.shape[1]))

## Adding obs error to obs
    inputs_sampled_noised = torch.clone(inputs_sampled)
    inputs_sampled_noised[:, :, :4] += mask_obs_error

## Normalizing inputs (obs + forcings)
    inputs_sampled_noised_normalized = torch.zeros(inputs_sampled_noised.shape)
    mean_input = torch.mean(inputs_sampled_noised, dim = (0, 1))
    std_input = torch.std(inputs_sampled_noised, dim = (0, 1))
    for ch in range(inputs_sampled_noised_normalized.shape[2]) :
        inputs_sampled_noised_normalized[:, :, ch] = (inputs_sampled_noised[:, :, ch]-torch.mean(inputs_sampled_noised[:, :, ch]))/torch.std(inputs_sampled_noised[:, :, ch])
    inputs_sampled_noised_normalized = inputs_sampled_noised_normalized.moveaxis(1, 2)
    print("input shape : ", inputs_sampled_noised_normalized.shape)

## Normalizing outputs (parameters)
    outputs_normalized = (Theta-torch.mean(Theta, dim = 0))/torch.std(Theta, dim = 0)
    print("output shape : ", outputs_normalized.shape)
    mean_output = torch.mean(Theta, dim = 0)
    std_output = torch.std(Theta, dim = 0)

## Saving mean, std, observation error
    torch.save(mean_output, name_save+name_file+"/mean_y.pt")
    torch.save(std_output, name_save+name_file+"/std_y.pt")
    torch.save(mean_input, name_save+name_file+"/mean_x.pt")
    torch.save(std_input, name_save+name_file+"/std_x.pt")

    torch.save(mask_obs_error, name_save+name_file+"/Obs_matrix.pt")

## 80% for training, 20% for validation
    DS_train = TensorDataset(torch.clone(inputs_sampled_noised_normalized[:int(n_file*file_size*0.8-1), :, :]), torch.clone(outputs_normalized[:int(n_file*file_size*0.8-1)]))
    DS_valid = TensorDataset(torch.clone(inputs_sampled_noised_normalized[int(n_file*file_size*0.8-1):, :, :]), torch.clone(outputs_normalized[int(n_file*file_size*0.8-1):]))
    torch.save(DS_train, name_save+name_file+"/DS_train")
    torch.save(DS_valid, name_save+name_file+"/DS_Valid")

## Deleting variables to save memory for the next generated case
    del mean_output, mean_input, std_output, std_input, mask_obs_error, inputs_sampled_noised_normalized, outputs_normalized, DS_train, DS_valid, Theta, inputs_sampled_noised, inputs_sampled, False_forcings, False_forcings_sampled, True_forcings_sampled, True_forcings

    t_whole = time.time()-ti_whole
    print(f"Data for the case {case+1} have been generated in {int(t_whole//3600)}h:{int((t_whole%3600)//60)}min:{int(((t_whole%3600)%60))}sec.")