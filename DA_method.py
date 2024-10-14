"""
Last update on June 2024

@author: jlittaye
"""
import sys

nb_sample = 100
dt = 1/2

case = int(sys.argv[1])
pat_1 = int(sys.argv[2])
pat_2 = int(sys.argv[3])

sampling_patt = [pat_1, pat_1, pat_2, pat_2]

dt_NPZD = [int(sampling_patt[0]/dt), int(sampling_patt[1]/dt), int(sampling_patt[2]/dt), int(sampling_patt[3]/dt)]
obs_noise_perc = 1.
w_bg = 1e-2
nb_epochs = 100

name_file = f"Res/DA_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/"

import matplotlib.pyplot as plt
import numpy as np
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
from Functions import DA, norm, validation_params, function_NPZD, clear_before_violinplot, get_rightcorner, get_topcorner

if not os.path.isdir(name_file) :
    os.makedirs(name_file)

nb_version = 0
for file in listdir(name_file) :
    if len(file) > 8 :
        if file[:8] == "version_" : 
            nb_version += 1

os.makedirs(name_file+f"version_{nb_version}/")
file_name =  os.path.basename(sys.argv[0])
fin = open(file_name, "r")
fout = open(name_file+f"version_{nb_version}/Script.py", "x")
fout.write(fin.read())
fin.close()
fout.close()

print("Start for the experiment: ", name_file+f"version_{nb_version}/")

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

global Q0
Q0 = 4+2.5+1.5+0




##############################################################################################################################
#####################################################      LEARNING      #####################################################
##############################################################################################################################
t_tensor = torch.arange(0, 365*5, 1/24)

global_f_0, global_delta_0 = torch.load(f"Generated_Datasets/DA/Case_{case}/True_forcings.pt", map_location = device).moveaxis(2, 0)
global_f_1, global_delta_1 = torch.load(f"Generated_Datasets/DA/Case_{case}/False_forcings.pt", map_location = device).moveaxis(2, 0)

train_loader = torch.load(f"Generated_Datasets/DA/Case_{case}/TrainLoader.pt", map_location = device)
theta_target = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", map_location = device)
mask_obs_err = torch.load(f"Generated_Datasets/DA/Case_{case}/Obs_matrix.pt", map_location = device)
print("Forcings, states and parameters loaded")

for x, y in train_loader :
    print("Input shape: ", x.shape)
    print("Output shape: ", y.shape)
    break

print("######################################\n######################################\n        Start Learning        \n######################################\n######################################")
(costs, costs_ref, params_visu, x0) = DA(case, nb_version, epochs=nb_epochs, lr=1e-3, sampling_patt=sampling_patt, theta_target=theta_target, device = device, train_loader=train_loader, global_f=global_f_1, global_delta=global_delta_1, mask_obs_error=mask_obs_err, dt=dt, w_bg=w_bg)

## Stores all the use/obtained variables/data into the same file
torch.save(theta_target, name_file+f"version_{nb_version}/Tensor_thetatarget.pt")
torch.save(torch.cat((global_f_0[None, :], global_f_1[None, :], global_delta_0[None, :], global_delta_1[None, :]), dim = 0), name_file+f"version_{nb_version}/Tensor_forcings.pt")
torch.save(torch.cat((costs[None, :, :], costs_ref[None, :, :]), dim = 0), name_file+f"version_{nb_version}/Tensor_cost.pt")
torch.save(params_visu, name_file+f"version_{nb_version}/Tensor_params.pt")
torch.save(train_loader, name_file+f"version_{nb_version}/Tensor_trainloader.pt")
torch.save(x0, name_file+f"version_{nb_version}/Tensor_x0.pt")



####################################################################
#######################    VALIDATION    ###########################
####################################################################
device = 'cpu'
torch.set_default_device(device)


print("Validation and plot of the metrics")
mean_y = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/mean_y.pt", map_location = device)
std_y = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/std_y.pt", map_location = device)

params_visu = params_visu.to(device)
costs = costs.to(device)

best_index = torch.argmin(costs[:, :1] + w_bg*costs[:, 1:])
theta_got = torch.clone(params_visu[:, :, best_index])
theta_init = torch.clone(params_visu[:, :, 0])
theta_target = theta_target.to(device)
global_f_1 = global_f_1.to(device)
global_delta_1 = global_delta_1.to(device)

corr_tensor, corr_tensor_init = torch.zeros([2, global_f_0.shape[0], 4])
shift_tensor, shift_tensor_init = torch.zeros([2, global_f_0.shape[0], 4])
ampl_tensor, ampl_tensor_init = torch.zeros([2, global_f_0.shape[0], 4])

(x_ref, N_ref, P_ref, Z_ref, D_ref) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_delta = global_delta_1, theta_values = theta_target)
(x_pred, N_pred, P_pred, Z_pred, D_pred) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_delta = global_delta_1, theta_values = theta_got)
(x_init, N_init, P_init, Z_init, D_init) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_delta = global_delta_1, theta_values = theta_init)

ti = time.time()
for i_val in range(global_f_0.shape[0]) :
    tepoch = time.time()
    (corr_tensor[i_val, 0], shift_tensor[i_val, 0], ampl_tensor[i_val, 0]) = validation_params(N_ref[i_val, int(365*4/dt):], N_pred[i_val, int(365*4/dt):])
    (corr_tensor[i_val, 1], shift_tensor[i_val, 1], ampl_tensor[i_val, 1]) = validation_params(P_ref[i_val, int(365*4/dt):], P_pred[i_val, int(365*4/dt):])
    (corr_tensor[i_val, 2], shift_tensor[i_val, 2], ampl_tensor[i_val, 2]) = validation_params(Z_ref[i_val, int(365*4/dt):], Z_pred[i_val, int(365*4/dt):])
    (corr_tensor[i_val, 3], shift_tensor[i_val, 3], ampl_tensor[i_val, 3]) = validation_params(D_ref[i_val, int(365*4/dt):], D_pred[i_val, int(365*4/dt):])
    
    (corr_tensor_init[i_val, 0], shift_tensor_init[i_val, 0], ampl_tensor_init[i_val, 0]) = validation_params(N_ref[i_val, int(365*4/dt):], N_init[i_val, int(365*4/dt):])
    (corr_tensor_init[i_val, 1], shift_tensor_init[i_val, 1], ampl_tensor_init[i_val, 1]) = validation_params(P_ref[i_val, int(365*4/dt):], P_init[i_val, int(365*4/dt):])
    (corr_tensor_init[i_val, 2], shift_tensor_init[i_val, 2], ampl_tensor_init[i_val, 2]) = validation_params(Z_ref[i_val, int(365*4/dt):], Z_init[i_val, int(365*4/dt):])
    (corr_tensor_init[i_val, 3], shift_tensor_init[i_val, 3], ampl_tensor_init[i_val, 3]) = validation_params(D_ref[i_val, int(365*4/dt):], D_init[i_val, int(365*4/dt):])
    sys.stdout.write("\rValidation n°"+str(1+i_val)+"/"+str(global_f_0.shape[0])+" within %.2f" % (time.time()-tepoch)+"s, still %.2f" %((time.time()-tepoch)*(global_f_0.shape[0]-1-i_val))+"s.")

print("\n Validation ended in %.2f" %(time.time()-ti) + "s !\n")
torch.save(corr_tensor, name_file+f"version_{nb_version}/Validation_tensor_corr.pt")
torch.save(shift_tensor, name_file+f"version_{nb_version}/Validation_tensor_shift.pt")
torch.save(ampl_tensor, name_file+f"version_{nb_version}/Validation_tensor_ampl.pt")


corr_tensor, shift_tensor, ampl_tensor = clear_before_violinplot(corr_tensor, shift_tensor, ampl_tensor)
corr_tensor_init, shift_tensor_init, ampl_tensor_init = clear_before_violinplot(corr_tensor_init, shift_tensor_init, ampl_tensor_init)



####################################################################
########################      PLOT      ############################
####################################################################
print("Plot of the variational cost evolution")
plt.figure(figsize = [6.4*2, 4.8*0.7])
plt.figure(figsize = [6.4*2, 4.8*0.7])
plt.semilogy(costs[1:, 0], label = 'Cost NPZD')
plt.semilogy(costs[1:, 1], linestyle = '--', label = 'Cost Background')
# plt.ylim(1e-4, 1)
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
        ax[i%5, i//5].semilogy(torch.mean(abs(theta_target[:, i].repeat(params_visu.shape[2], 1).moveaxis(0, 1)-params_visu[:, i, :]), dim = 0))
        ax[i%5, i//5].set_title("Evolution of "+params_list[i])
        ax[i%5, i//5].set_xlabel("Epoch")
        ax[i%5, i//5].grid()
        ax[i%5, i//5].set_ylim(1e-3, 1e0)
plt.tight_layout()
plt.savefig(name_file+f"version_{nb_version}/Plot_parameters", dpi = 200)


fig, ax = plt.subplots(figsize = [6.4*1.5, 4.8*0.7])
with torch.no_grad() :
    ax.semilogy(torch.mean(abs(theta_target.to('cpu').repeat(params_visu.shape[2], 1, 1).moveaxis(0, 2)-params_visu)**2, dim = (0, 1)))
    ax.set_title("Evolution of the averaged parameters")
    ax.set_xlabel("Epoch")
    ax.grid()
    ax.set_ylim(1e-3, 1e0)
plt.tight_layout()
plt.savefig(name_file+f"version_{nb_version}/Plot_avg_parameters", dpi = 200)

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
ax_shift.set_ylim(-1, 15)
ax_ampl.set_title("Amplitude ratio")
ax_ampl.set_ylim(0., 2.)
ax_param.set_title("Biogeochemical parameter error")
ax_param.set_ylim(-0.2, .5)
c_DA, s_DA, r_DA, p_DA, i_DA = [], [], [], [], []


for i in range(1, 5) :
    corr_to_plot = torch.ones(corr_tensor.transpose(0, 1).shape)*1000 # value of 1000 used as a default value
    corr_to_plot[i-1, :] = corr_tensor.transpose(0, 1)[i-1, :]
    shift_to_plot = torch.ones(shift_tensor.transpose(0, 1).shape)*1000
    shift_to_plot[i-1, :] = abs(shift_tensor.transpose(0, 1))[i-1, :]
    ampl_to_plot = torch.ones(ampl_tensor.transpose(0, 1).shape)*1000
    ampl_to_plot[i-1, :] = ampl_tensor.transpose(0, 1)[i-1, :]
    c_DA.append(ax_corr.violinplot(corr_to_plot, showextrema=False))
    s_DA.append(ax_shift.violinplot(shift_to_plot, showextrema=False))
    r_DA.append(ax_ampl.violinplot(ampl_to_plot, showextrema=False))
p_DA = ax_param.violinplot((torch.sqrt((theta_got-theta_target)**2)/mean_y).transpose(0, 1), showextrema=False)

i_DA.append(ax_corr.violinplot(corr_tensor_init.transpose(0, 1), showextrema=False))
i_DA.append(ax_shift.violinplot(shift_tensor_init.transpose(0, 1), showextrema=False))
i_DA.append(ax_ampl.violinplot(ampl_tensor_init.transpose(0, 1), showextrema=False))
i_DA.append(ax_param.violinplot((torch.sqrt((theta_init-theta_target)**2)/mean_y).transpose(0, 1), showextrema=False))

for i in range(4) :
    for pc in i_DA[i]['bodies']: # initial metrics for corr, shift, ampl, param
        pc.set_facecolor('grey')
        pc.set_edgecolor('grey')
        pc.set_linewidth(3)

for pc in p_DA['bodies']: # params
    pc.set_facecolor(color_params[0][0])
    pc.set_edgecolor(color_params[0][1])
    pc.set_linewidth(3)
for i in range(10) :
    ax_param.vlines(i+1, (torch.sqrt((theta_got-theta_target)**2)/mean_y)[:, i].min().item(), (torch.sqrt((theta_got-theta_target)**2)/mean_y)[:, i].max().item(), color=color_params[0][1], linestyle='-.', lw=2)
    ax_param.hlines((torch.sqrt((theta_got-theta_target)**2)/mean_y)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=color_params[0][1], linestyle='-', lw=2)
    ax_param.hlines((torch.sqrt((theta_got-theta_target)**2)/mean_y)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=color_params[0][1], linestyle='-', lw=2)

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
    ax_shift.vlines(i+1, abs(shift_tensor)[:, i].min().item(), abs(shift_tensor)[:, i].max().item(), color=color_NPZD[i][1], linestyle='-.', lw=2)
    ax_shift.hlines(abs(shift_tensor)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=color_NPZD[i][1], lw=1)
    ax_shift.hlines(abs(shift_tensor)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=color_NPZD[i][1], lw=1)
    for pc in r_DA[i]['bodies']: # ampl
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