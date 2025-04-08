"""
Last update on April 2025

@author: jlittaye
"""
global case
global Q0
import sys

case = int(sys.argv[1])
pat_1 = int(sys.argv[2])
pat_2 = int(sys.argv[3])

sampling_patt = [pat_1, pat_1, pat_2, pat_2]

lr = 1e-4
nb_epochs = 10000
dt = 1/2
model = str(sys.argv[4])

name_file = f"Res/NN_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}_{model}/"
btch_size = 512

Q0 = 4+2.5+1.5+0
####################################################################

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
from lightning.pytorch.loggers import CSVLogger


from Functions import validation_params, clear_before_violinplot, function_NPZD, data_sampling, get_rightcorner, get_topcorner, Model_MLP, ModelNN, Model_UNet

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


####################################################################
########################    TRAINING    ############################
####################################################################
    
global device
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu") #########################

torch.set_default_device(device)
print("Par défaut, device set to : ", device)

training_dataset = torch.load(f"Generated_Datasets/NN/Case_{case}/DS_train", map_location = device, weights_only=False)
validation_dataset = torch.load(f"Generated_Datasets/NN/Case_{case}/DS_valid", map_location = device, weights_only=False)

sampled_training_dataset = data_sampling(training_dataset, sampling_patt = sampling_patt)
sampled_validation_dataset = data_sampling(validation_dataset, sampling_patt = sampling_patt)

mean_x = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_x.pt", map_location = 'cpu', weights_only=False)
std_x = torch.load(f"Generated_Datasets/NN/Case_{case}/std_x.pt", map_location = 'cpu', weights_only=False)
mean_y = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_y.pt", map_location = 'cpu', weights_only=False)
std_y = torch.load(f"Generated_Datasets/NN/Case_{case}/std_y.pt", map_location = 'cpu', weights_only=False)

training_DL = DataLoader(sampled_training_dataset, batch_size=btch_size, shuffle=True, generator = torch.Generator(device))
validation_DL = DataLoader(sampled_validation_dataset, batch_size=btch_size, shuffle=True, generator = torch.Generator(device))

epoch_i = 0

if model == "CNN" :
    MyNet = ModelNN(mean = mean_y, std = std_y, lr = lr, sampling_patt = sampling_patt)
elif model == 'MLP' :
    MyNet = Model_MLP(mean = mean_y, std = std_y, lr = lr, sampling_patt = sampling_patt)
elif model == 'UNET' :
    MyNet = Model_UNet(mean = mean_y, std = std_y, lr = lr, sampling_patt = sampling_patt)
MyNet.to(device)


checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="valid_loss",
    mode="min",
    dirpath=name_file+f"version_{nb_version}/top_10/",
    filename="chkpt_{epoch:02d}")
checkpoint_callback_2 = ModelCheckpoint(
    every_n_epochs = 200,
    save_top_k = -1,
    monitor="valid_loss",
    dirpath=name_file+f"version_{nb_version}/every_n_epochs/",
    filename="chkpt_{epoch:02d}")

logger = CSVLogger(save_dir = name_file, name = f"version_{nb_version}/lightning_logs", version = None)

###################################################

trainer = pl.Trainer(check_val_every_n_epoch=10, default_root_dir = name_file+f"version_{nb_version}/", min_epochs = nb_epochs, max_epochs=nb_epochs, callbacks=[checkpoint_callback, checkpoint_callback_2], log_every_n_steps=None, logger = logger)#, log_every_n_steps=50, check_val_every_n_epoch=2
ti_train = time.time()
trainer.fit(model=MyNet, train_dataloaders=training_DL, val_dataloaders=validation_DL)

print("End of the training after %.2f " %(time.time()-ti_train), "seconds.")

trainer.save_checkpoint(name_file+f"version_{nb_version}/final_chkpt.ckpt")
torch.save(MyNet.state_dict(), name_file+f"version_{nb_version}/final_state_dict")
del MyNet

####################################################################
#######################    VALIDATION    ###########################
####################################################################
device = 'cpu'
torch.set_default_device(device)
mean_y = mean_y.to(device); std_y = std_y.to(device)

DS = torch.load(f"Generated_Datasets/DA/Case_{case}/DS_test_NN", map_location = device, weights_only=False)
DL = DataLoader(DS, batch_size = 100)
global_f_0, global_m_0 = torch.load(f"Generated_Datasets/DA/Case_{case}/True_forcings.pt", map_location = device, weights_only=False).moveaxis(2, 0)
global_f_1, global_m_1 = torch.load(f"Generated_Datasets/DA/Case_{case}/False_forcings.pt", map_location = device, weights_only=False).moveaxis(2, 0)
theta_target = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", map_location = device, weights_only=False)


selection_model = pd.DataFrame(listdir(name_file+f"version_{nb_version}/top_10/"), columns = ["file_name"])
selection_model["epoch"] = [int(path_model.lstrip("chkpt_epoch=").rstrip(".ckpt")) for path_model in selection_model.file_name]

logs = pd.read_csv(name_file+f"version_{nb_version}/lightning_logs/version_0/metrics.csv", header = 0)
tensor_logs = torch.from_numpy(np.array([logs.epoch.unique()]))
for name_col in ["train_loss_epoch", "valid_loss"] :
    new_col = torch.from_numpy(np.array([[np.nanmin(logs[logs.epoch == i_epoch][name_col]) for i_epoch in logs.epoch.unique()]]))
    tensor_logs = torch.cat((tensor_logs, new_col), dim = 0)
logs_clean = pd.DataFrame(tensor_logs.transpose(0, 1), columns = ["epoch", "train_loss", "valid_loss"])

selection_model["valid_loss"] = [logs_clean[logs_clean.epoch == top_model_epoch].valid_loss.item() for top_model_epoch in selection_model.epoch]


chckpt_path = name_file+f"version_{nb_version}/top_10/"+selection_model[selection_model.valid_loss == min(selection_model.valid_loss)].file_name.item()

if model == "CNN" :
    MyNet_val = ModelNN.load_from_checkpoint(chckpt_path, mean = mean_y, std = std_y, lr = lr, sampling_patt = sampling_patt)
elif model == 'MLP' :
    MyNet_val = Model_MLP.load_from_checkpoint(chckpt_path, mean = mean_y, std = std_y, lr = lr, sampling_patt = sampling_patt)
elif model == 'UNET' :
    MyNet_val = Model_UNet.load_from_checkpoint(chckpt_path, mean = mean_y, std = std_y, lr = lr, sampling_patt = sampling_patt)
MyNet_val.eval().to(device)

for x, y in DL :
    theta_got = MyNet_val(x).detach()*std_y+mean_y
torch.save(theta_got, name_file+f"version_{nb_version}/Validation_tensor_theta_pred_DA.pt")
theta_rdm = torch.tensor([random.random()*0.4+0.8 for i in range(len(theta_got.flatten()))]).reshape(theta_got.shape)
torch.save(theta_rdm, name_file+f"version_{nb_version}/Validation_tensor_theta_pred_DA_rdm.pt")

corr_tensor, corr_tensor_init = torch.zeros([2, global_f_0.shape[0], 4])
shift_tensor, shift_tensor_init = torch.zeros([2, global_f_0.shape[0], 4])
ampl_tensor, ampl_tensor_init = torch.zeros([2, global_f_0.shape[0], 4])

(x_ref, N_ref, P_ref, Z_ref, D_ref) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_m = global_m_1, theta_values = theta_target)
(x_pred, N_pred, P_pred, Z_pred, D_pred) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_m = global_m_1, theta_values = theta_got)
(x_init, N_init, P_init, Z_init, D_init) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_m = global_m_1, theta_values = theta_rdm)



ti = time.time()
for i_val in range(global_f_0.shape[0]) :
    tepoch = time.time()
    (corr_tensor[i_val, 0], shift_tensor[i_val, 0], ampl_tensor[i_val, 0]) = validation_params(N_ref[i_val, int(365*4/dt):].detach(), N_pred[i_val, int(365*4/dt):].detach())
    (corr_tensor[i_val, 1], shift_tensor[i_val, 1], ampl_tensor[i_val, 1]) = validation_params(P_ref[i_val, int(365*4/dt):].detach(), P_pred[i_val, int(365*4/dt):].detach())
    (corr_tensor[i_val, 2], shift_tensor[i_val, 2], ampl_tensor[i_val, 2]) = validation_params(Z_ref[i_val, int(365*4/dt):].detach(), Z_pred[i_val, int(365*4/dt):].detach())
    (corr_tensor[i_val, 3], shift_tensor[i_val, 3], ampl_tensor[i_val, 3]) = validation_params(D_ref[i_val, int(365*4/dt):].detach(), D_pred[i_val, int(365*4/dt):].detach())
    
    (corr_tensor_init[i_val, 0], shift_tensor_init[i_val, 0], ampl_tensor_init[i_val, 0]) = validation_params(N_ref[i_val, int(365*4/dt):].detach(), N_init[i_val, int(365*4/dt):].detach())
    (corr_tensor_init[i_val, 1], shift_tensor_init[i_val, 1], ampl_tensor_init[i_val, 1]) = validation_params(P_ref[i_val, int(365*4/dt):].detach(), P_init[i_val, int(365*4/dt):].detach())
    (corr_tensor_init[i_val, 2], shift_tensor_init[i_val, 2], ampl_tensor_init[i_val, 2]) = validation_params(Z_ref[i_val, int(365*4/dt):].detach(), Z_init[i_val, int(365*4/dt):].detach())
    (corr_tensor_init[i_val, 3], shift_tensor_init[i_val, 3], ampl_tensor_init[i_val, 3]) = validation_params(D_ref[i_val, int(365*4/dt):].detach(), D_init[i_val, int(365*4/dt):].detach())
    sys.stdout.write("\rValidation n°"+str(1+i_val)+"/"+str(global_f_0.shape[0])+" within %.2f" % (time.time()-tepoch)+"s, still %.2f" %((time.time()-tepoch)*(global_f_0.shape[0]-1-i_val))+"s.")

print("\n Validation ended in %.2f" %(time.time()-ti) + "s !\n")
torch.save(corr_tensor, name_file+f"version_{nb_version}/Validation_tensor_corr.pt")
torch.save(shift_tensor, name_file+f"version_{nb_version}/Validation_tensor_shift.pt")
torch.save(ampl_tensor, name_file+f"version_{nb_version}/Validation_tensor_ampl.pt")

torch.save(corr_tensor_init, name_file+f"version_{nb_version}/Validation_tensor_corr_rdm.pt")
torch.save(shift_tensor_init, name_file+f"version_{nb_version}/Validation_tensor_shift_rdm.pt")
torch.save(ampl_tensor_init, name_file+f"version_{nb_version}/Validation_tensor_ampl_rdm.pt")

corr_tensor, shift_tensor, ampl_tensor = clear_before_violinplot(corr_tensor, shift_tensor, ampl_tensor)
corr_tensor_init, shift_tensor_init, ampl_tensor_init = clear_before_violinplot(corr_tensor_init, shift_tensor_init, ampl_tensor_init)


####################################################################
########################      PLOT      ############################
####################################################################
logs = pd.read_csv(name_file+f"version_{nb_version}/lightning_logs/version_0/metrics.csv", header = 0, index_col = 'epoch')

valid_loss = logs[np.isnan(logs["valid_loss"]) == False]["valid_loss"]
training_loss = logs[np.isnan(logs["train_loss_epoch"]) == False]["train_loss_epoch"]

plt.figure(figsize = [6.4*1.5, 4.8*0.7])
plt.semilogy(valid_loss, label = 'valid')
plt.semilogy(training_loss, label = 'train')
plt.xlabel("epoch")
plt.title("Training metrics")
plt.legend()
plt.grid()
plt.savefig(name_file+f"version_{nb_version}/Plot_loss", dpi = 200)

#############################################
## Individual violin plots
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
ax_param.set_ylim(-0.5, .5)
c_DA, s_DA, r_DA, p_DA, i_DA = [], [], [], [], []

for i in range(1, 5) :
    corr_to_plot = torch.ones(corr_tensor.shape)*1000
    corr_to_plot[:, i-1] = corr_tensor[:, i-1]
    shift_to_plot = torch.ones(shift_tensor.shape)*1000
    shift_to_plot[:, i-1] = abs(shift_tensor)[:, i-1]
    ampl_to_plot = torch.ones(ampl_tensor.shape)*1000
    ampl_to_plot[:, i-1] = ampl_tensor[:, i-1]
    c_DA.append(ax_corr.violinplot(corr_to_plot, showextrema=False))
    s_DA.append(ax_shift.violinplot(shift_to_plot, showextrema=False))
    r_DA.append(ax_ampl.violinplot(ampl_to_plot, showextrema=False))
p_DA = ax_param.violinplot(((theta_got.detach()-theta_target)/mean_y), showextrema=False)

i_DA.append(ax_corr.violinplot(corr_tensor_init, showextrema=False))
i_DA.append(ax_shift.violinplot(abs(shift_tensor_init), showextrema=False))
i_DA.append(ax_ampl.violinplot(ampl_tensor_init, showextrema=False))
i_DA.append(ax_param.violinplot(((theta_rdm-theta_target)/mean_y), showextrema=False))

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
    ax_param.vlines(i+1, ((theta_got.detach()-theta_target)/mean_y)[:, i].min().item(), ((theta_got.detach()-theta_target)/mean_y)[:, i].max().item(), color=color_params[0][1], linestyle='-.', lw=2)
    ax_param.hlines(((theta_got.detach()-theta_target)/mean_y)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=color_params[0][1], linestyle='-', lw=2)
    ax_param.hlines(((theta_got.detach()-theta_target)/mean_y)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=color_params[0][1], linestyle='-', lw=2)

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
ax_param.set_xticks(torch.arange(1, 11, 1), [r"$\chi$", r"$\rho$", r"$\gamma$", r"$\lambda$", r"$\epsilon$", r"$\alpha$", r"$\beta$", r"$\eta$", r"$\varphi$", r'$\zeta$'])
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14) #fontsize of the x and y labels
plt.rc('xtick', labelsize=14) #fontsize of the x tick labels
plt.rc('ytick', labelsize=14) #fontsize of the y tick labels

plt.subplots_adjust(bottom=0.15, wspace=0.15)
plt.tight_layout()
plt.savefig(name_file+f"version_{nb_version}/Plot_validation_metrics", dpi = 200)


####################################################################
######################      ENSEMBLE      ##########################
####################################################################

DS_test_ensemble = torch.load(f"Generated_Datasets/DA/Case_{case}/DS_test_NN_ensemble_20", map_location = device, weights_only=False)
DL_ensemble = DataLoader(DS_test_ensemble, batch_size = 100)
    
theta_target = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", map_location = device, weights_only=False)
    

selection_model = pd.DataFrame(listdir(name_file+f"version_{nb_version}/top_10/"), columns = ["file_name"])
selection_model["epoch"] = [int(path_model.lstrip("chkpt_epoch=").rstrip(".ckpt")) for path_model in selection_model.file_name]

logs = pd.read_csv(name_file+f"version_{nb_version}/lightning_logs/version_0/metrics.csv", header = 0)
tensor_logs = torch.from_numpy(np.array([logs.epoch.unique()]))
for name_col in ["train_loss_epoch", "valid_loss"] :
    new_col = torch.from_numpy(np.array([[np.nanmin(logs[logs.epoch == i_epoch][name_col]) for i_epoch in logs.epoch.unique()]]))
    tensor_logs = torch.cat((tensor_logs, new_col), dim = 0)
logs_clean = pd.DataFrame(tensor_logs.transpose(0, 1), columns = ["epoch", "train_loss", "valid_loss"])

selection_model["valid_loss"] = [logs_clean[logs_clean.epoch == top_model_epoch].valid_loss.item() for top_model_epoch in selection_model.epoch]

chckpt_path = name_file+f"version_{nb_version}/top_10/"+selection_model[selection_model.valid_loss == min(selection_model.valid_loss)].file_name.item()


if model == "CNN" :
    MyNet_val = ModelNN.load_from_checkpoint(chckpt_path, mean = mean_y, std = std_y, lr = lr, sampling_patt = sampling_patt)
elif model == 'MLP' :
    MyNet_val = Model_MLP.load_from_checkpoint(chckpt_path, mean = mean_y, std = std_y, lr = lr, sampling_patt = sampling_patt)
elif model == 'UNET' :
    MyNet_val = Model_UNet.load_from_checkpoint(chckpt_path, mean = mean_y, std = std_y, lr = lr, sampling_patt = sampling_patt)

MyNet_val.eval().to(device)
print(f"start infering with version {chckpt_path}")
for x, y in DL_ensemble :
    for samp in range(x.shape[-1]) :
        if not samp :
            theta_NN = (MyNet_val(x[:, :, :, samp]).detach()*std_y+mean_y)[None]
        else :
            theta_NN = torch.cat((theta_NN, (MyNet_val(x[:, :, :, samp]).detach()*std_y+mean_y)[None]), dim = 0)
    # print(f"Loss {torch.mean((theta_NN.mean(dim=0)-(theta_target-mean_y)/std_y)**2)}")
    # print(f"NRMSE {torch.mean(abs((theta_NN.mean(dim=0)-(theta_target-mean_y)/std_y)))}")

theta_ens = torch.mean(theta_NN, dim = 0)
torch.save(theta_ens, name_file+f"version_{nb_version}/Validation_tensor_theta_pred_DA_ens.pt")

print("End of the script")