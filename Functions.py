"""
Last update on June 2024

@author: jlittaye
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn.functional as F
import pandas as pd
import time
import os
from os import listdir
from os.path import isfile, join
import shutil
import random
from math import *
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

global Q0
Q0 = 4+2.5+1.5+0


###########################################################################
######################### Data generation/analysis #########################
###########################################################################

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
    ampl = 2*(torch.max(data_pred)-torch.min(data_pred))/((torch.max(data_pred)-torch.min(data_pred))+torch.max(data_obs)-torch.min(data_obs))
    corr = torch.corrcoef(torch.moveaxis(torch.cat((data_obs[:, None], data_pred[:, None]), dim = 1), 0, 1))[0, 1].item()
    return [corr, shift, ampl]

############################################

def clear_before_violinplot(corr, shift, ampl, crit = [1, 360, 100]) :
    df = pd.DataFrame(data = torch.cat((corr, shift, ampl), axis = 1), columns = ["Corr_N", "Corr_P", "Corr_Z", "Corr_D", "Shift_N", "Shift_P", "Shift_Z", "Shift_D", "Ampl_N", "Ampl_P", "Ampl_Z", "Ampl_D"])
    df_val = df.iloc[:, :][(1-df["Corr_N"] <= crit[0])&(1-df["Corr_P"] <= crit[0])&(1-df["Corr_Z"] <= crit[0])&(1-df["Corr_D"] <= crit[0])]

    df_val = df_val.iloc[:, :][(abs(df_val["Shift_N"]) <= crit[1])&(abs(df_val["Shift_P"]) <= crit[1])&(abs(df_val["Shift_Z"]) <= crit[1])&(abs(df_val["Shift_D"]) <= crit[1])]

    df_val = df_val.iloc[:, :][(abs(1-df_val["Ampl_N"]) <= crit[2])&(abs(1-df_val["Ampl_P"]) <= crit[2])&(abs(1-df_val["Ampl_Z"]) <= crit[2])&(abs(1-df_val["Ampl_D"]) <= crit[2])]

    return torch.tensor(df_val[["Corr_N", "Corr_P", "Corr_Z", "Corr_D"]].values), torch.tensor(df_val[["Shift_N", "Shift_P", "Shift_Z", "Shift_D"]].values), torch.tensor(df_val[["Ampl_N", "Ampl_P", "Ampl_Z", "Ampl_D"]].values)

############################################

def norm(tensor) :
    std = torch.std(tensor)
    mean = torch.mean(tensor)
    return (tensor-mean)/std
    
############################################

def DA(case, nb_version, epochs, lr, train_loader, global_f, global_delta, mask_obs_error, sampling_patt, theta_target, device = 'cpu', dt = 1/24, lr_x0 = 1e-4, w_bg = 1) :

    name_file = f"Res/DA_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/"
    n_sample = theta_target.shape[0]  
    ti = time.time()
    tepoch = time.time()

## If it does not exist, creates a file with the initial/prior parameters.
    if not os.path.isdir("Generated_Datasets/DA/Params_start_global.pt") :
        theta_biais = torch.tensor([[random.uniform(0.8, 1.2) for theta_i in range(10)] for i_sample in range(n_sample)])
        torch.save(theta_biais, "Generated_Datasets/DA/Params_start_global.pt")
    else : 
        theta_biais = torch.load("Generated_Datasets/DA/Params_start_global.pt", map_location = device)
    torch.save(theta_biais, name_file+f"version_{nb_version}/Tensor_initialtheta.pt")
    
    model_stud = Model_NPZD(theta_biais) ## Model that tends to estimate the parameters
    model_ref = Model_NPZD(theta_target) ## Model that already has the correct parameters to compare variational costs
    
    # mask_obs_error = torch.load(f"Generated_Datasets/DA/Case_{case}/Obs_matrix.pt", map_location = device)
    for ch in range(4) :
        if sampling_patt[ch] == 0 :
            mask_obs_error[:, :, ch, :] = 0
    torch.save(mask_obs_error, name_file+f"version_{nb_version}/Tensor_obsmatrix.pt")
    std_obs = torch.std(mask_obs_error, dim = (0, 1, 3))
## Initialization of the background X0    
    with torch.no_grad() :
        for x, y in train_loader :
            for ch in range(4) :
                if sampling_patt[ch] :
                    model_stud.X0[:, :, ch] = torch.clone(x[:, :, 1+ch, 0]+mask_obs_error[:, :, ch, 0])
                else :
                    model_stud.X0[:, :, ch] = torch.mean(x[:, :, 1+ch, 0], dim = 0).repeat(x.shape[0], 1)
            break

    model_stud.to(device)
    model_ref.to(device)
    model_ref.eval()
    optim = torch.optim.Adam(model_stud.parameters(), lr=lr)
    optim_x0 = torch.optim.Adam([model_stud.X0], lr=lr_x0)
    
    params_visu = torch.cat([param.clone().detach()[:, None] for param in model_stud.parameters()], dim = 1)[:, :, None]
    x0_visu = torch.clone(model_stud.X0[:, :, :, None].detach())
    
    costs, costs_ref = torch.tensor([[0., 0.]]), torch.tensor([[0., 0.]])
    mean_simu_time = torch.zeros([5]) ## For time estimation of the optimization process and the storage 

    print("Start of the optimization process")
    for i in range(epochs):
        # Iterates through training dataloader
        cost_mean, cost_ref_mean = torch.tensor([[0., 0.]]), torch.tensor([[0., 0.]])

        for x, y in train_loader :
            # Simulates states
            preds_param = model_stud(torch.cat((x[:, :, 0], model_stud.X0), dim = 2)[:, :, :, None], global_f, global_delta, dt = dt)
            preds_ref = model_ref(torch.cat((x[:, :, 0], model_stud.X0), dim = 2)[:, :, :, None], global_f, global_delta, dt = dt)
            
            # Computes variational cost upon estimated parameters
            cost_npzd_estim, cost_bg_estim = variational_cost(preds_param, y + mask_obs_error, sampling_patt, std_mod = torch.std(y, dim = (1, 3)), std_obs = std_obs)
            cost_npzd_ref, cost_bg_ref = variational_cost(preds_ref, y + mask_obs_error, sampling_patt, std_mod = torch.std(y, dim = (1, 3)), std_obs = std_obs)
            
            cost_mean = torch.cat((cost_mean, torch.tensor([[cost_npzd_estim.item(), cost_bg_estim.item()]])), dim = 0)
            cost_ref_mean = torch.cat((cost_ref_mean, torch.tensor([[cost_npzd_ref.item(), cost_bg_ref.item()]])), dim = 0)

            # Initializes gradient, computes it and backpropagates it
            optim.zero_grad()
            (cost_npzd_estim+cost_bg_estim*w_bg).backward(retain_graph = True)
            optim.step()
            
            optim_x0.zero_grad()
            (cost_npzd_estim+cost_bg_estim*w_bg).backward()
            optim_x0.step()

            with torch.no_grad() :
                model_stud.X0[:] = model_stud.X0.clamp(0) # To prevent the background having negative values
                for param in model_stud.parameters() :
                    param[:] = param.clamp(0.8, 1.2)

        if (i+1)%10 == 0 :
            x0_visu = torch.cat((x0_visu, model_stud.X0[:, :, :, None].detach()), axis = 3)
        params_visu = torch.cat((params_visu, torch.cat([param.clone().detach()[:, None] for param in model_stud.parameters()], dim = 1)[:, :, None]), dim = 2)

        costs = torch.cat((costs, torch.mean(cost_mean[1:, :], dim = 0)[None, :]), dim = 0)
        costs_ref = torch.cat((costs_ref, torch.mean(cost_ref_mean[1:, :], dim = 0)[None, :]), dim = 0)
        
        ## Selects last measured epoch duration to estimate the whole duration
        if i == 0 : 
            mean_simu_time = torch.ones(mean_simu_time.shape)*(time.time()-tepoch)
        else :
            mean_simu_time = torch.roll(mean_simu_time, 1)
            mean_simu_time[0] = time.time()-tepoch
        remaining_time = (mean_simu_time.mean()*(epochs-i-1)).item()
        if (i+1)%10 == 0 :
            sys.stdout.write(f"\rEpoch {1+i}/{epochs} ["+int(np.floor(20*(i+1)/epochs))*"#"+int(20-np.floor(20*(i+1)/epochs))*'_'+f"] : Var_cost: {(cost_npzd_estim+w_bg*cost_bg_estim).item():.4f} in {mean_simu_time.mean():.2f}s. Estimation: {int(remaining_time//3600)}h:{int(remaining_time%3600)//60}min:{remaining_time%60:.1f}s remaining.")
        tepoch = time.time()

    print(f"\nThe optimization took {int((tepoch-ti)//3600)}h:{int((tepoch-ti)%3600)//60}min:{(tepoch-ti)%60:.1f}s.")
    return(costs[1:, :], costs_ref[1:, :], params_visu, x0_visu)

############################################

def data_sampling(DS_in, sampling_patt = [1, 1, 1, 1]) :
    DL = DataLoader(DS_in, batch_size = 10000)
    for x, y in DL :
        x_replace = torch.zeros(x.shape)
        y_replace = torch.clone(y)
        for i in range(len(sampling_patt)) :
            if sampling_patt[i] == 1 :
                x_replace[:, i, :] = x[:, i, :]
            elif sampling_patt[i] > 1 :
                for day in range(x_replace.shape[-1]) :
                    if day%sampling_patt[i] == 0 :
                        x_replace[:, i, day] = x[:, i, day]
    x_replace[:, len(sampling_patt):, :] = x[:, len(sampling_patt):, :]
    return TensorDataset(x_replace, y_replace)

############################################

def selection_criteria(corr, shift, ampl, theta, crit = [1, 100, [0.3, 5.], 2]) :
    df = pd.DataFrame(data = torch.cat((corr, shift, ampl, torch.max(theta, axis = 1).values[:, None]), axis = 1), columns = ["Corr_N", "Corr_P", "Corr_Z", "Corr_D", "Shift_N", "Shift_P", "Shift_Z", "Shift_D", "Ampl_N", "Ampl_P", "Ampl_Z", "Ampl_D", "Theta_err_mean"])
    nb = [len(df.index)]
    df_val = df.iloc[:, :][(1-df["Corr_N"] <= crit[0])&(1-df["Corr_P"] <= crit[0])&(1-df["Corr_Z"] <= crit[0])&(1-df["Corr_D"] <= crit[0])]
    nb.append(len(df_val))
    df_val = df.iloc[:, :][(abs(df["Shift_N"]) <= crit[1])&(abs(df["Shift_P"]) <= crit[1])&(abs(df["Shift_Z"]) <= crit[1])&(abs(df["Shift_D"]) <= crit[1])]
    nb.append(len(df_val))
    df_val = df.iloc[:, :][(df["Ampl_N"] >= crit[2][0])&(df["Ampl_N"] <= crit[2][1])&(df["Ampl_P"] >= crit[2][0])&(df["Ampl_P"] <= crit[2][1])&(df["Ampl_Z"] >= crit[2][0])&(df["Ampl_Z"] <= crit[2][1])&(df["Ampl_D"] >= crit[2][0])&(df["Ampl_D"] <= crit[2][1])]
    nb.append(len(df_val))
    df_val = df.iloc[:, :][(df["Theta_err_mean"] <= crit[3])]
    nb.append(len(df_val))
    df_val = df.iloc[:, :][(1-df["Corr_N"] <= crit[0])&(1-df["Corr_P"] <= crit[0])&(1-df["Corr_Z"] <= crit[0])&(1-df["Corr_D"] < crit[0]) & (abs(df["Shift_N"]) <= crit[1])&(abs(df["Shift_P"]) <= crit[1])&(abs(df["Shift_Z"]) <= crit[1])&(abs(df["Shift_D"]) <= crit[1]) & (df["Ampl_N"] >= crit[2][0])&(df["Ampl_N"] <= crit[2][1])&(df["Ampl_P"] >= crit[2][0])&(df["Ampl_P"] <= crit[2][1])&(df["Ampl_Z"] >= crit[2][0])&(df["Ampl_Z"] <= crit[2][1])&(df["Ampl_D"] >= crit[2][0])&(df["Ampl_D"] <= crit[2][1]) & (df["Theta_err_mean"] <= crit[3])]
    nb.append(len(df_val))
    print(f"We removed {nb[0]-nb[-1]} outliers - Corr {nb[0]-nb[1]}/100, Shift {nb[0]-nb[2]}/100, Amplitude {nb[0]-nb[3]}/100, Theta {nb[0]-nb[4]}/100.")
    return torch.tensor(df_val[["Corr_N", "Corr_P", "Corr_Z", "Corr_D"]].values), torch.tensor(df_val[["Shift_N", "Shift_P", "Shift_Z", "Shift_D"]].values), torch.tensor(df_val[["Ampl_N", "Ampl_P", "Ampl_Z", "Ampl_D"]].values), torch.tensor(df["Theta_err_mean"].values)

############################################

def variational_cost(predictions, targets, sampling_pattern, std_mod, std_obs):
    mask_NPZD = torch.zeros(predictions.shape)
    for ch in range(4) :
        if sampling_pattern[ch] != 0 :
            for i_t in range(1, predictions.shape[3]) :
                if (i_t+1)%(sampling_pattern[ch]) == 0 :
                    mask_NPZD[:, :, ch, i_t] = 1. # Focuses on the weeks 18 (May) to 43 (October) where the dynamic is more important
            mask_NPZD[:, :, ch, :] /= torch.sum(mask_NPZD[0, 0, ch, :])

    error_obs = torch.movedim((predictions - targets)*mask_NPZD/torch.sum(mask_NPZD), (0, 2), (2, 3))
    erreur_bg = torch.movedim(predictions[:, :-1, :, -1]-predictions[:, 1:, :, 0], (0), (1))
    return torch.sum((error_obs/std_obs)**2), torch.mean((erreur_bg/std_mod)**2)

############################################

def select_version(method, case, sampling_patt) :
    nb_version = -1
    for file in os.listdir(f"Res/{method}_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/") :
        if len(file) > 8 :
            if file[:8] == "version_" :
                nb_version+= 1
    return nb_version


###################################################################
####################     Plot functions     ########################
###################################################################

def get_topcorner(ax, lim = 0.9) :
    """To get the y position of the axis."""
    return ax.get_ybound()[0]+(ax.get_ybound()[1]-ax.get_ybound()[0])*lim

############################################

def get_rightcorner(ax, lim = 0.9) :
    """To get the x position of the axis."""
    return ax.get_xbound()[0]+(ax.get_xbound()[1]-ax.get_xbound()[0])*lim

############################################

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)




###################################################################
########################     MODELS     ############################
###################################################################

class ModelNN(pl.LightningModule):
    def __init__(self, mean, std, lr, sampling_patt):
        super().__init__()
        self.meanTheta = mean
        self.stdTheta = std
        self.lr = lr
        self.sampling_patt = sampling_patt
        self.conv1 = nn.Conv1d(6, 16, kernel_size = 5, padding = 0)
        self.conv2 = nn.Conv1d(16, 32, kernel_size = 5, padding = 0)
        self.conv3 = nn.Conv1d(32, 64, kernel_size = 5, padding = 0)
        self.conv4 = nn.Conv1d(64, 32, kernel_size = 5, padding = 0)
        self.avgpool1 = nn.AvgPool1d(349)
        self.dense1 = nn.Linear(32, 10)
        self.dropout1 = nn.Dropout(p=0.2)

    def forward(self, x):   
        for n_sample in range(len(self.sampling_patt)) :
            if self.sampling_patt[n_sample] > 1 :
                ch_to_sample = (x[:, n_sample, :][:, None, :].unfold(2, 1, 7)[:, :, :, 0]).clone()
                x[:, n_sample, :] = nn.functional.interpolate(input = ch_to_sample, size = [ch_to_sample.shape[-1]*self.sampling_patt[n_sample]], mode = 'linear')[:, 0, 3:-3]
            elif self.sampling_patt[n_sample] == 0 :
                x[:, n_sample] = 0.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(self.dropout1(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(self.dropout1(x)))
        x = torch.flatten(self.avgpool1(x), 1) # flatten all dimensions except the batch dimension
        x = self.dense1(x)
        return x
    
    def mse_loss(self, predictions, targets):
        return torch.mean((predictions-targets)**2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, x, y):
        ti = time.time()
        x_train = x[0]
        y_train = x[1]
        loss = self.mse_loss(self(x_train), y_train)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, x, y):
        global device
        x_valid = x[0]
        y_valid = x[1]
        loss = self.mse_loss(self(x_valid), y_valid)
        self.log('valid_loss', loss, on_epoch=True)
        return loss

############################################################

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
        
    def forward(self, X_in, global_f, global_delta, dt, range_pred = 7) :
        Vm = 1.
        threshold = torch.tensor([0.01])
        param_values = torch.tensor([1, 2, 0.1, 0.05, 0.1, 0.3, 0.6, 0.15, 0.4, 0.1])
        # Kn 0, Rm 1, g 2, lambd 3, epsilon 4, alpha 5, beta 6, r 7, phi 8, Sw 9
        N = X_in[:, :, 1, 0][:, :, None]
        P = X_in[:, :, 2, 0][:, :, None]
        Z = X_in[:, :, 3, 0][:, :, None]
        D = X_in[:, :, 4, 0][:, :, None]
        
        f = torch.zeros([X_in.shape[0], X_in.shape[1], int(range_pred/dt)+1])
        delta = torch.zeros([X_in.shape[0], X_in.shape[1], int(range_pred/dt)+1])
        for week in range(X_in.shape[1]) :
            X_time = torch.arange(X_in[0, week, 0, 0].item(), X_in[0, week, 0, 0].item()+range_pred+dt, dt)
            f[:, week, :] = torch.index_select(global_f, 1, torch.round(X_time*24).int())
            delta[:, week, :] = torch.index_select(global_delta, 1, torch.round(X_time*24).int())
        
        gamma_N   = (N / ((self.Kn*param_values[0]).repeat(X_in.shape[1], 1).moveaxis(0, 1)[:, :, None] + N))
        zoo_graze = (self.Rm.repeat(X_in.shape[1], 1).moveaxis(0, 1)[:, :, None]*param_values[1] * (1 - torch.exp(-self.lambd.repeat(X_in.shape[1], 1).moveaxis(0, 1)[:, :, None]*param_values[3] * torch.max(threshold, P))) * torch.max(threshold, Z))
        
        for i_t in range(int(range_pred/dt)) :
            N = torch.cat((N, (dt * (-Vm*gamma_N[:, :, i_t]*f[:, :, i_t]*torch.max(threshold, P[:, :, i_t]) + self.alpha.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[5]*zoo_graze[:, :, i_t] + self.epsilon.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[4]*P[:, :, i_t] + self.g.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[2]*Z[:, :, i_t] + self.phi.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[8]*D[:, :, i_t] + delta[:, :, i_t]*(Q0 - N[:, :, i_t])) + N[:, :, i_t])[:, :, None]), dim = 2)
            P = torch.cat((P, (dt * (Vm*gamma_N[:, :, i_t]*f[:, :, i_t]*torch.max(threshold, P[:, :, i_t]) - zoo_graze[:, :, i_t] - self.epsilon.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[4]*P[:, :, i_t] - self.r.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[7]*P[:, :, i_t] - delta[:, :, i_t]*P[:, :, i_t]) + P[:, :, i_t])[:, :, None]), dim = 2)
            Z = torch.cat((Z, (dt * (self.beta.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[6]*zoo_graze[:, :, i_t] - self.g.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[2]*Z[:,:, i_t] - delta[:, :, i_t]*Z[:, :, i_t]) + Z[:, :, i_t])[:, :, None]), dim = 2)
            D = torch.cat((D, (dt * (self.r.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[7]*P[:, :, i_t] + (1-self.alpha.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[5]-self.beta.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[6])*zoo_graze[:, :, i_t] - self.phi.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[8]*D[:, :, i_t] - self.Sw.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[9]*D[:, :, i_t] - delta[:, :, i_t]*D[:, :, i_t]) + D[:, :, i_t])[:, :, None]), dim = 2)

            gamma_N = torch.cat((gamma_N, (N[:, :, i_t+1] / (self.Kn.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[0] + N[:, :, i_t+1]))[:, :, None]), dim = 2)
            zoo_graze = torch.cat((zoo_graze, (self.Rm.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[1] * (1 - torch.exp(-self.lambd.repeat(X_in.shape[1], 1).moveaxis(0, 1)*param_values[3] * torch.max(threshold, P[:, :, i_t+1]))) * torch.max(threshold, Z[:, :, i_t+1]))[:, :, None]), dim = 2)

        return torch.cat((N[:, :, None, :], P[:, :, None, :], Z[:, :, None, :], D[:, :, None, :]), dim = 2).unfold(3, 1, int(1/dt))[:, :, :, :, 0]
