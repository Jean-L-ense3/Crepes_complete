"""
Created on March 2024

@author: jlittaye
"""

nb_sample = 100
dt = 1/2
sampling_patt = [1, 1, 7, 7]
dt_NPZD = [int(sampling_patt[0]/dt), int(sampling_patt[1]/dt), int(sampling_patt[2]/dt), int(sampling_patt[3]/dt)]
case = 1
obs_noise_perc = 1.
nb_epochs = 100

name_file = "DA_"+str(sampling_patt[0])+"d_"+str(sampling_patt[1])+"d_"+str(sampling_patt[2])+"d_"+str(sampling_patt[3])+"d_case"+str(case)+"/"
print("Start for the experiment: ", name_file)

import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.feature_extraction import image
import pandas as pd
import time
import os
from os import listdir
import random
from math import *


if not os.path.isdir(name_file) :
    os.makedirs(name_file)

nb_version = 0
for file in listdir(name_file) :
    if file[:8] == "version_" : 
        nb_version += 1
os.makedirs(name_file+f"version_{nb_version}/")
file_name =  os.path.basename(sys.argv[0])
fin = open(file_name, "r")
fout = open(name_file+f"version_{nb_version}/Script.py", "x")
fout.write(fin.read())
fin.close()
fout.close()


global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


###################################################################################
################################### Functions #####################################
###################################################################################

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

############################################

def validation_params(data_obs, data_pred, dt = 1/2) :
    """Fonction pour quantifier la reconstruction en calculant : le décalage, la corrélation max, 
       le rapport des max/min, la corrélation entre mai et octobre."""
    shift = int((np.correlate(data_obs, data_obs, 'full').argmax()-np.correlate(data_obs, data_pred, 'full').argmax())*dt)
    ampl = (torch.max(data_pred)-torch.min(data_pred))/(torch.max(data_obs)-torch.min(data_obs))
    if shift > 0 :
        corr = torch.corrcoef(torch.moveaxis(torch.cat((data_obs[:-shift, None], data_pred[shift:, None]), dim = 1), 0, 1))[0, 1].item()
    elif shift < 0 :
        corr = torch.corrcoef(torch.moveaxis(torch.cat((data_obs[-shift:, None], data_pred[:shift, None]), dim = 1), 0, 1))[0, 1].item()
    else :
        corr = torch.corrcoef(torch.moveaxis(torch.cat((data_obs[:, None], data_pred[:, None]), dim = 1), 0, 1))[0, 1].item()
    return [corr, shift, ampl]

############################################

def clear_before_violinplot(corr, shift, ampl, crit = [1, 360, 100]) :
    df = pd.DataFrame(data = torch.cat((corr, shift, ampl), axis = 1), columns = ["Corr_N", "Corr_P", "Corr_Z", "Corr_D", "Shift_N", "Shift_P", "Shift_Z", "Shift_D", "Ampl_N", "Ampl_P", "Ampl_Z", "Ampl_D"])
    df_val = df.iloc[:, :][(1-df["Corr_N"] <= crit[0])&(1-df["Corr_P"] <= crit[0])&(1-df["Corr_Z"] <= crit[0])&(1-df["Corr_D"] <= crit[0])]

    df_val = df_val.iloc[:, :][(abs(df["Shift_N"]) <= crit[1])&(abs(df["Shift_P"]) <= crit[1])&(abs(df["Shift_Z"]) <= crit[1])&(abs(df["Shift_D"]) <= crit[1])]

    df_val = df_val.iloc[:, :][(abs(1-df["Ampl_N"]) <= crit[2])&(abs(1-df["Ampl_P"]) <= crit[2])&(abs(1-df["Ampl_Z"]) <= crit[2])&(abs(1-df["Ampl_D"]) <= crit[2])]

    return torch.tensor(df_val[["Corr_N", "Corr_P", "Corr_Z", "Corr_D"]].values), torch.tensor(df_val[["Shift_N", "Shift_P", "Shift_Z", "Shift_D"]].values), torch.tensor(df_val[["Ampl_N", "Ampl_P", "Ampl_Z", "Ampl_D"]].values)

############################################

def training(epochs, lr, train_loader, global_f, global_delta, sampling_patt, theta_target, dt = 1/24, obs_noise_perc = 0, lr_x0 = 1e-4) :
    global model_stud
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_sample = theta_target.shape[0]  
    ti = time.time()
    tepoch = time.time()

## If it does not exist, creates a file with the initial/prior parameters.
    if not os.path.isdir("Generated_Datasets/DA/Params_start_global.pt") :
        theta_biais = torch.tensor([[random.uniform(0.8, 1.2) for theta_i in range(10)] for i_sample in range(n_sample)])
        torch.save(theta_biais, "Generated_Datasets/DA/Params_start_global.pt")
    else : 
        theta_biais = torch.load("Generated_Datasets/DA/Params_start_global.pt")

    torch.save(theta_biais, name_file+f"version_{nb_version}/Tensor_initialtheta.pt")
    model_stud = Model_NPZD(theta_biais) ## Model that tends to estimate the parameters
    model_ref = Model_NPZD(theta_target) ## Model that already has the correct parameters to compare variational costs
    
    mask_obs_error = torch.load(f"Generated_Datasets/DA/Case_{case}/Obs_matrix.pt")
    for ch in range(4) :
        if sampling_patt[ch] == 0 :
            mask_obs_error[:, :, ch, :] = 0
    torch.save(mask_obs_error, name_file+f"version_{nb_version}/Tensor_obsmatrix.pt")
## Initialization of the background X0    
    with torch.no_grad() :
        for x, y in train_loader :
            for ch in range(4) :
                if sampling_patt[ch] :
                    model_stud.X0[:, :, ch] = torch.clone(x[:, :, 1+ch, 0]+mask_obs_error[:, :, ch, 0])
            break

    model_stud.to(device)
    model_ref.to(device)
    model_ref.eval()
    params_tensor = torch.cat([param.clone().detach()[:, None] for param in model_stud.parameters()], dim = 1)[:, :, None]
    x0_visu = torch.clone(model_stud.X0[:, :, :, None].detach())
    cost_array, cost_ref_array = [], []

    optim = torch.optim.Adam(model_stud.parameters(), lr=lr)
    optim_x0 = torch.optim.Adam([model_stud.X0], lr=lr_x0)
    
    mean_simu_time = torch.zeros([5]) ## For time estimation of the optimization process and the storage 

    print("Start of the optimization process")
    for i in range(epochs):
        # Iterates through training dataloader
        cost_mean, cost_ref_mean = [], []

        for x,y in train_loader :
            # Simulates states
            preds_param = model_stud(torch.cat((x[:, :, 0], (model_stud.X0 + mask_obs_error[:, :, :, 0])), dim = 2)[:, :, :, None], global_f, global_delta, sampling_patt = sampling_patt, dt = dt)
            preds_ref = model_ref(torch.cat((x[:, :, 0], model_stud.X0 + mask_obs_error[:, :, :, 0]), dim = 2)[:, :, :, None], global_f, global_delta, sampling_patt = sampling_patt, dt = dt)
            
            # Computes variational cost upon estimated parameters
            cost_params_estim = variational_cost(preds_param, y + mask_obs_error, sampling_patt)
            cost_ref_estim = variational_cost(preds_ref, y + mask_obs_error, sampling_patt)
            
            cost_mean.append(cost_params_estim.item())
            cost_ref_mean.append(cost_ref_estim.item())
            
            # Computes variational cost upon estimated background
            preds_x0 = model_stud(torch.cat((x[:, :, 0, :], model_stud.X0 + mask_obs_error[:, :, :, 0]), dim = 2)[:, :, :, None], global_f, global_delta, sampling_patt = sampling_patt, dt = dt)
            cost_x0 = variational_cost(preds_x0, y + mask_obs_error, sampling_patt)
            
            # Initializes gradient, computes it and backpropagates it
            optim_x0.zero_grad()
            cost_x0.backward()
            optim_x0.step()

            optim.zero_grad()
            cost_params_estim.backward()
            optim.step()

            if (i+1)%100 == 0 : # Stores backgorund every 100 epochs
                x0_visu = torch.cat((x0_visu, model_stud.X0[:, :, :, None].detach()), axis = 3)

        cost_ref_array.append(np.mean(cost_ref_mean)); cost_array.append(np.mean(cost_mean))
        params_tensor = torch.cat((params_tensor, torch.cat([param.clone().detach()[:, None] for param in model_stud.parameters()], dim = 1)[:, :, None]), dim = 2)

        ## Selects last measured time for epochs to estimate the whole duration
        if i == 0 : 
            mean_simu_time = torch.ones(mean_simu_time.shape)*(time.time()-tepoch)
        else :
            mean_simu_time = torch.roll(mean_simu_time, 1)
            mean_simu_time[0] = time.time()-tepoch
        remaining_time = (mean_simu_time.mean()*(epochs-i-1)).item()
        if (i+1)%10 == 0 :
            sys.stdout.write(f"\rEpoch {1+i}/{epochs} ["+int(np.floor(20*(i+1)/epochs))*"#"+int(20-np.floor(20*(i+1)/epochs))*'_'+f"] : Var_cost: {cost_params_estim.item():.4f} in {mean_simu_time.mean():.2f}s. Estimation: {int(remaining_time//3600)}h:{int(remaining_time%3600)//60}min:{remaining_time%60:.1f}s remaining.")
        tepoch = time.time()

    print(f"\nThe optimization took {int((tepoch-ti)//3600)}h:{int((tepoch-ti)%3600)//60}min:{(tepoch-ti)%60:.1f}s.")
    return(cost_array, cost_ref_array, params_tensor, x0_visu)

############################################

def get_topcorner(ax, lim = 0.9) :
    """To get the y position of the axis."""
    return ax.get_ybound()[0]+(ax.get_ybound()[1]-ax.get_ybound()[0])*lim
def get_rightcorner(ax, lim = 0.9) :
    """To get the x position of the axis."""
    return ax.get_xbound()[0]+(ax.get_xbound()[1]-ax.get_xbound()[0])*lim


###############################################################################
################################### Model #####################################
###############################################################################
global Q0
Q0 = 4+2.5+1.5+0

class Model_NPZD(torch.nn.Module) :
    def __init__(self, params_value) :
        super().__init__()
        self.Kn = torch.nn.Parameter(params_value[:, 0])
        self.Rm = torch.nn.Parameter(params_value[:, 1])
        self.g = torch.nn.Parameter(params_value[:, 2])
        self.lambd = torch.nn.Parameter(params_value[:, 3])
        self.epsilon = torch.nn.Parameter(params_value[:, 4])
        self.alpha = torch.nn.Parameter(params_value[:, 5])
        self.beta = torch.nn.Parameter(params_value[:, 6])
        self.r = torch.nn.Parameter(params_value[:, 7])
        self.phi = torch.nn.Parameter(params_value[:, 8])
        self.Sw = torch.nn.Parameter(params_value[:, 9])
        
        self.X0 = torch.autograd.Variable(torch.zeros((params_value.shape[0], int(365/7), 4), dtype = torch.double), requires_grad = True)
        
    def forward(self, X_in, global_f, global_delta, sampling_patt, dt) :
        Vm = 1.
        threshold = torch.tensor([0.01])
        param_values = torch.tensor([1, 2, 0.1, 0.05, 0.1, 0.3, 0.6, 0.15, 0.4, 0.1])
        # Kn 0, Rm 1, g 2, lambd 3, epsilon 4, alpha 5, beta 6, r 7, phi 8, Sw 9
        N = X_in[:, :, 1, 0][:, :, None]
        P = X_in[:, :, 2, 0][:, :, None]
        Z = X_in[:, :, 3, 0][:, :, None]
        D = X_in[:, :, 4, 0][:, :, None]
        
        f = torch.tensor([[value.item() for value in torch.index_select(global_f[i_batch, :], 0, torch.round(X_in[i_batch, :, 0, 0]*24+0).int().ravel())[:, None]] for i_batch in range(X_in.shape[0])])[:, :, None]
        delta = torch.tensor([[value.item() for value in torch.index_select(global_delta[i_batch, :], 0, torch.round(X_in[i_batch, :, 0, 0]*24+0).int().ravel())[:, None]] for i_batch in range(X_in.shape[0])])[:, :, None]

        gamma_N   = (N / ((self.Kn*param_values[0]).repeat(X_in.shape[1], 1).moveaxis(0, 1)[:, :, None] + N))
        zoo_graze = (self.Rm.repeat(X_in.shape[1], 1).moveaxis(0, 1)[:, :, None]*param_values[1] * (1 - torch.exp(-self.lambd.repeat(X_in.shape[1], 1).moveaxis(0, 1)[:, :, None]*param_values[3] * torch.max(threshold, P))) * torch.max(threshold, Z))
        
        for i_t in range(int(7/dt)) :
            N = torch.cat((N, (dt * (-Vm*gamma_N[:, :, i_t]*f[:, :, i_t]*torch.max(threshold, P[:, :, i_t]) + self.alpha.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[5]*zoo_graze[:, :, i_t] + self.epsilon.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[4]*P[:, :, i_t] + self.g.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[2]*Z[:, :, i_t] + self.phi.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[8]*D[:, :, i_t] + delta[:, :, i_t]*(Q0 - N[:, :, i_t])) + N[:, :, i_t])[:, :, None]), dim = 2)
            P = torch.cat((P, (dt * (Vm*gamma_N[:, :, i_t]*f[:, :, i_t]*torch.max(threshold, P[:, :, i_t]) - zoo_graze[:, :, i_t] - self.epsilon.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[4]*P[:, :, i_t] - self.r.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[7]*P[:, :, i_t] - delta[:, :, i_t]*P[:, :, i_t]) + P[:, :, i_t])[:, :, None]), dim = 2)
            Z = torch.cat((Z, (dt * (self.beta.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[6]*zoo_graze[:, :, i_t] - self.g.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[2]*Z[:,:, i_t] - delta[:, :, i_t]*Z[:, :, i_t]) + Z[:, :, i_t])[:, :, None]), dim = 2)
            D = torch.cat((D, (dt * (self.r.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[7]*P[:, :, i_t] + (1-self.alpha.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[5]-self.beta.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[6])*zoo_graze[:, :, i_t] - self.phi.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[8]*D[:, :, i_t] - self.Sw.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[9]*D[:, :, i_t] - delta[:, :, i_t]*D[:, :, i_t]) + D[:, :, i_t])[:, :, None]), dim = 2)

            f = torch.cat((f, torch.tensor([[value.item() for value in torch.index_select(global_f[i_batch, :], 0, torch.round(X_in[i_batch, :, 0, 0]*24+(i_t+1)*24).int().ravel())[:, None]] for i_batch in range(X_in.shape[0])])[:, :, None]), dim = 2)
            delta = torch.cat((delta, torch.tensor([[value for value in torch.index_select(global_delta[i_batch, :], 0, torch.round(X_in[i_batch, :, 0, 0]*24+(i_t+1)*24).int().ravel())[:, None]] for i_batch in range(X_in.shape[0])])[:, :, None]), dim = 2)
            gamma_N = torch.cat((gamma_N, (N[:, :, i_t+1] / (self.Kn.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[0] + N[:, :, i_t+1]))[:, :, None]), dim = 2)
            zoo_graze = torch.cat((zoo_graze, (self.Rm.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[1] * (1 - torch.exp(-self.lambd.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[3] * torch.max(threshold, P[:, :, i_t+1]))) * torch.max(threshold, Z[:, :, i_t+1]))[:, :, None]), dim = 2)

        return torch.cat((N[:, :, None, :], P[:, :, None, :], Z[:, :, None, :], D[:, :, None, :]), dim = 2).unfold(3, 1, int(1/dt))[:, :, :, :, 0]


def variational_cost(predictions, targets, sampling_patt, w_x0 = 1e-3):
    mask_NPZD = torch.zeros(predictions.shape)
    for ch in range(4) :
        if sampling_patt[ch] != 0 :
            for i_t in range(predictions.shape[3]) :
                if (i_t+1)%(sampling_patt[ch]) == 0 :
                    mask_NPZD[:, 18:43, ch, i_t] = sampling_patt[ch]/max(sampling_patt) # Focuses on the weeks 18 (May) to 43 (October) where the dynamic is more important

    difference = (predictions - targets)*mask_NPZD
    erreur_x0 = predictions[:, :-1, :, -1]-predictions[:, 1:, :, 0]
    return torch.sum(difference**2)/torch.sum(mask_NPZD)+torch.mean(erreur_x0**2)*w_x0

##############################################################################################################################
#####################################################      LEARNING      #####################################################
##############################################################################################################################


t_tensor = torch.arange(0, 365*5, 1/24)

global_f_0, global_delta_0 = torch.load(f"Generated_Datasets/DA/Case_{case}/True_forcings.pt")[:, :, 0], torch.load(f"Generated_Datasets/DA/Case_{case}/True_forcings.pt")[:, :, 1]
global_f_1, global_delta_1 = torch.load(f"Generated_Datasets/DA/Case_{case}/False_forcings.pt")[:, :, 0], torch.load(f"Generated_Datasets/DA/Case_{case}/False_forcings.pt")[:, :, 1]

train_loader = torch.load(f"Generated_Datasets/DA/Case_{case}/TrainLoader.pt")
theta_target = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt")
print("Forcings, states and parameters loaded")

for x, y in train_loader :
    print("Input shape: ", x.shape)
    print("Output shape: ", y.shape)
    break

print("######################################\n######################################\n        Start Learning        \n######################################\n######################################")
(cost_array, cost_ref_array, params_tensor, x0) = training(obs_noise_perc = obs_noise_perc, epochs = nb_epochs, lr = 1e-3, sampling_patt = sampling_patt, theta_target = theta_target, train_loader = train_loader, global_f = global_f_1, global_delta = global_delta_1, dt = dt)

## Stores all the use/obtained variables/data into the same file
torch.save(theta_target, name_file+f"version_{nb_version}/Tensor_thetatarget.pt")
torch.save(torch.cat((global_f_0[None, :], global_f_1[None, :], global_delta_0[None, :], global_delta_1[None, :]), dim = 0), name_file+f"version_{nb_version}/Tensor_forcings.pt")
torch.save(torch.cat((torch.Tensor(cost_array)[None, :], torch.Tensor(cost_ref_array)[None, :]), dim = 0), name_file+f"version_{nb_version}/Tensor_loss.pt")
torch.save(params_tensor, name_file+f"version_{nb_version}/Tensor_params.pt")
torch.save(train_loader, name_file+f"version_{nb_version}/Tensor_trainloader.pt")
torch.save(x0, name_file+f"version_{nb_version}/Tensor_x0.pt")


####################################################################
#######################    VALIDATION    ###########################
####################################################################

print("Validation and plot of the metrics")
mean_y = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/mean_y.pt")
std_y = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/std_y.pt")
theta_got = torch.clone(params_tensor[:, :, -1])

corr_tensor = torch.zeros([global_f_0.shape[0], 4])
shift_tensor = torch.zeros([global_f_0.shape[0], 4])
ampl_tensor = torch.zeros([global_f_0.shape[0], 4])

(x_ref, N_ref, P_ref, Z_ref, D_ref) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_delta = global_delta_1, theta_values = theta_target)
(x_val, N_pred, P_pred, Z_pred, D_pred) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_delta = global_delta_1, theta_values = theta_got)

ti = time.time()
for i_val in range(global_f_0.shape[0]) :
    tepoch = time.time()
    (corr_tensor[i_val, 0], shift_tensor[i_val, 0], ampl_tensor[i_val, 0]) = validation_params(N_ref[i_val, int(365*4/dt):].detach(), N_pred[i_val, int(365*4/dt):].detach())
    (corr_tensor[i_val, 1], shift_tensor[i_val, 1], ampl_tensor[i_val, 1]) = validation_params(P_ref[i_val, int(365*4/dt):].detach(), P_pred[i_val, int(365*4/dt):].detach())
    (corr_tensor[i_val, 2], shift_tensor[i_val, 2], ampl_tensor[i_val, 2]) = validation_params(Z_ref[i_val, int(365*4/dt):].detach(), Z_pred[i_val, int(365*4/dt):].detach())
    (corr_tensor[i_val, 3], shift_tensor[i_val, 3], ampl_tensor[i_val, 3]) = validation_params(D_ref[i_val, int(365*4/dt):].detach(), D_pred[i_val, int(365*4/dt):].detach())
    sys.stdout.write("\rValidation n°"+str(1+i_val)+"/"+str(global_f_0.shape[0])+" within %.2f" % (time.time()-tepoch)+"s, still %.2f" %((time.time()-tepoch)*(global_f_0.shape[0]-1-i_val))+"s.")

print("\n Validation ended in %.2f" %(time.time()-ti) + "s !\n")
torch.save(corr_tensor, name_file+f"version_{nb_version}/Validation_tensor_corr.pt")
torch.save(shift_tensor, name_file+f"version_{nb_version}/Validation_tensor_shift.pt")
torch.save(ampl_tensor, name_file+f"version_{nb_version}/Validation_tensor_ampl.pt")
corr_tensor, shift_tensor, ampl_tensor = clear_before_violinplot(corr_tensor, shift_tensor, ampl_tensor)


####################################################################
########################      PLOT      ############################
####################################################################
print("Plot of the variational cost evolution")
plt.figure(figsize = [6.4*2, 4.8*0.7])
plt.semilogy(cost_array[:], label = 'Estimated parameters')
plt.semilogy(cost_ref_array, linestyle = '--', label = 'Actual parameters')
plt.title("Variational cost evolution")
plt.xlabel("Epoch")
plt.ylabel("Cost value")
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(name_file+f"version_{nb_version}/Plot_varcost", dpi = 200)

#############################################
print("Plot of the parameters evolution")
params_list = ["Kn", "Rm", "g", "lambd", "epsilon", "alpha", "beta", "r", "phi", 'Sw']

fig, ax = plt.subplots(figsize = [6.4*1.5, 2.5*4.8], nrows = 5, ncols = 2)
with torch.no_grad() :
    for i in range(10) :
        ax[i%5, i//5].plot(torch.mean(abs(theta_target[:, i].repeat(params_tensor.shape[2], 1).moveaxis(0, 1)-params_tensor[:, i, :]), dim = 0))
        ax[i%5, i//5].set_title("Evolution of "+params_list[i])
        ax[i%5, i//5].set_xlabel("Epoch")
        ax[i%5, i//5].grid()
        ax[i%5, i//5].set_ylim(-1., 1.)
plt.tight_layout()

plt.savefig(name_file+f"version_{nb_version}/Plot_parameters", dpi = 200)

#############################################

color_params = [["#a055b5", "#8f04b5"]]
color_NPZD = [["#bebada", "#8d84d1"], ["#8dd3c7", "#33ab96"], ["#fb8072", "#eb4431"], ["#e5d8bd", "#deb766"]]

fig = plt.figure(figsize=(6.4*2.1, 6.4*0.7*2))
grid = fig.add_gridspec(nrows = 2, ncols=3)
ax_corr = fig.add_subplot(grid[0, 0])
ax_shift = fig.add_subplot(grid[0, 1])
ax_ampl = fig.add_subplot(grid[0, 2])
ax_param = fig.add_subplot(grid[1, :])

ax_corr.set_title('Correlation')
ax_corr.set_ylim(-0.05, 1.05)
ax_shift.set_title('Shift (day)')
ax_shift.set_ylim(-1, 30)
ax_ampl.set_title("Amplitude ratio")
ax_ampl.set_ylim(-0.2, 5.)
ax_param.set_title("Biogeochemical parameter error")
ax_param.set_ylim(-0.2, 5)
c_DA, s_DA, r_DA, v_DA = [], [], [], []

for i in range(1, 5) :
    corr_to_plot = torch.ones(corr_tensor.transpose(0, 1).shape)*1000
    corr_to_plot[i-1, :] = corr_tensor.transpose(0, 1)[i-1, :]
    shift_to_plot = torch.ones(shift_tensor.transpose(0, 1).shape)*1000
    shift_to_plot[i-1, :] = abs(shift_tensor.transpose(0, 1))[i-1, :]*dt
    ampl_to_plot = torch.ones(ampl_tensor.transpose(0, 1).shape)*1000
    ampl_to_plot[i-1, :] = ampl_tensor.transpose(0, 1)[i-1, :]
    c_DA.append(ax_corr.violinplot(corr_to_plot, showextrema=False))
    s_DA.append(ax_shift.violinplot(shift_to_plot, showextrema=False))
    r_DA.append(ax_ampl.violinplot(ampl_to_plot, showextrema=False))

for i in range(1, 11) :
    params_to_plot = torch.ones(theta_target.transpose(0, 1).shape)*1000
    params_to_plot[i-1, :] = (abs(theta_got.detach()*std_y[:10]+mean_y[:10]-theta_target)*dt).transpose(0, 1)[i-1, :]
    v_DA.append(ax_param.violinplot(params_to_plot, showextrema=False))

for i in range(10) :
    for pc in v_DA[i]['bodies']: # params
        pc.set_facecolor(color_params[0][0])
        pc.set_edgecolor(color_params[0][1])
        pc.set_linewidth(3)
    ax_param.vlines(i+1, (abs(theta_got.detach()*std_y[:10]+mean_y[:10]-theta_target)*dt)[:, i].min().item(), (abs(theta_got.detach()*std_y[:10]+mean_y[:10]-theta_target)/1)[:, i].max().item(), color=color_params[0][1], linestyle='-.', lw=2)
    ax_param.hlines((abs(theta_got.detach()*std_y[:10]+mean_y[:10]-theta_target)*dt)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=color_params[0][1], linestyle='-', lw=2)
    ax_param.hlines((abs(theta_got.detach()*std_y[:10]+mean_y[:10]-theta_target)*dt)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=color_params[0][1], linestyle='-', lw=2)

for i in range(4) :
    for pc in c_DA[i]['bodies']: #corr
        pc.set_facecolor(color_NPZD[i][0])
        pc.set_edgecolor(color_NPZD[i][1])
        pc.set_linewidth(3)
    ax_corr.vlines(i+1, corr_tensor[:, i].min().item(), corr_tensor[:, i].max().item(), color=color_NPZD[i][1], linestyle='-.', lw=2)
    ax_corr.hlines(corr_tensor[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=color_NPZD[i][1], linestyle='-', lw=2)
    ax_corr.hlines(corr_tensor[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=color_NPZD[i][1], linestyle='-', lw=2)
    for pc in s_DA[i]['bodies']: # shift
        pc.set_facecolor(color_NPZD[i][0])
        pc.set_edgecolor(color_NPZD[i][1])
        pc.set_linewidth(3)
    ax_shift.vlines(i+1, abs(shift_tensor/2)[:, i].min().item(), abs(shift_tensor/2)[:, i].max().item(), color=color_NPZD[i][1], linestyle='-.', lw=2)
    ax_shift.hlines(abs(shift_tensor/2)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=color_NPZD[i][1], lw=1)
    ax_shift.hlines(abs(shift_tensor/2)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=color_NPZD[i][1], lw=1)
    for pc in r_DA[i]['bodies']:
        pc.set_facecolor(color_NPZD[i][0])
        pc.set_edgecolor(color_NPZD[i][1])
        pc.set_linewidth(3)
    ax_ampl.vlines(i+1, ampl_tensor[:, i].min().item(), ampl_tensor[:, i].max().item(), color=color_NPZD[i][1], linestyle='-.', lw=2)
    ax_ampl.hlines(ampl_tensor[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=color_NPZD[i][1], linestyle='-', lw=2)
    ax_ampl.hlines(ampl_tensor[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=color_NPZD[i][1], linestyle='-', lw=2)

ax_corr.text(s = "a)", x = get_rightcorner(ax_corr, 0.9), y = get_topcorner(ax_corr, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
ax_shift.text(s = "b)", x = get_rightcorner(ax_shift, 0.9), y = get_topcorner(ax_shift, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
ax_ampl.text(s = "c)", x = get_rightcorner(ax_ampl, 0.9), y = get_topcorner(ax_ampl, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
ax_param.text(s = "d)", x = get_rightcorner(ax_param, 0.96), y = get_topcorner(ax_param, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)

for ax in [ax_corr, ax_shift, ax_ampl] :
    ax.grid()
    ax.set_xticks([1, 2, 3, 4], ["N", "P", "Z", "D"])

ax_param.grid()
ax_param.set_xticks(torch.arange(1, 11, 1), ["$K_N$", "$R_m$", "$g$", r"$\lambda$", r"$\epsilon$", r"$\alpha$", r"$\beta$", "$r$", r"$\phi$", '$S_w$'])
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14) #fontsize of the x and y labels
plt.rc('xtick', labelsize=14) #fontsize of the x tick labels
plt.rc('ytick', labelsize=14) #fontsize of the y tick labels

plt.subplots_adjust(bottom=0.15, wspace=0.15)
plt.tight_layout()
plt.savefig(name_file+f"version_{nb_version}/Plot_validation_metrics", dpi = 200)

print("End of the script")