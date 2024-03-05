global case
global global_LR
global graph_flag
global Q0

case = 2
sampling_patt = [1, 1, 7, 7]
global_LR = 1e-4
nb_epochs = 20
dt = 1/2

name_file = "NN_"+str(sampling_patt[0])+"d_"+str(sampling_patt[1])+"d_"+str(sampling_patt[2])+"d_"+str(sampling_patt[3])+"d_case"+str(case)+"/"
btch_size = 512

Q0 = 4+2.5+1.5+0
####################################################################

import matplotlib.pyplot as plt
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


print("Start of the script for the file: ", name_file)

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



####################################################################
####################################################################

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
    """Function that calculates the metrics upon the state dynamics."""
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
    """Function that remove the outliers that prevent a correct display of the violin plots."""
    df = pd.DataFrame(data = torch.cat((corr, shift, ampl), axis = 1), columns = ["Corr_N", "Corr_P", "Corr_Z", "Corr_D", "Shift_N", "Shift_P", "Shift_Z", "Shift_D", "Ampl_N", "Ampl_P", "Ampl_Z", "Ampl_D"])
    df_val = df.iloc[:, :][(1-df["Corr_N"] <= crit[0])&(1-df["Corr_P"] <= crit[0])&(1-df["Corr_Z"] <= crit[0])&(1-df["Corr_D"] <= crit[0])]
    print(f"Après corr : {df_val.shape[0]}")
    df_val = df_val.iloc[:, :][(abs(df["Shift_N"]) <= crit[1])&(abs(df["Shift_P"]) <= crit[1])&(abs(df["Shift_Z"]) <= crit[1])&(abs(df["Shift_D"]) <= crit[1])]
    print(f"Après shift : {df_val.shape[0]}")
    df_val = df_val.iloc[:, :][(abs(1-df["Ampl_N"]) <= crit[2])&(abs(1-df["Ampl_P"]) <= crit[2])&(abs(1-df["Ampl_Z"]) <= crit[2])&(abs(1-df["Ampl_D"]) <= crit[2])]
    print(f"En tout, on a enlevé {corr.shape[0]-len(df_val.index)} outliers")

    return torch.tensor(df_val[["Corr_N", "Corr_P", "Corr_Z", "Corr_D"]].values), torch.tensor(df_val[["Shift_N", "Shift_P", "Shift_Z", "Shift_D"]].values), torch.tensor(df_val[["Ampl_N", "Ampl_P", "Ampl_Z", "Ampl_D"]].values)

############################################

def data_sampling(DS_in, sampling_patt = [1, 1, 1, 1]) :
    """Function that sets the observed states to 0 according to the sampling pattern."""
    cat_x, cat_y = None, None
    for x, y in DS_in :
        if cat_x == None :
            cat_x = torch.clone(x[None, :, :])
            for day in range(x.shape[1]) :
                for ch in range(4) :
                    if sampling_patt[ch] != 0 :
                        if day%sampling_patt[ch] != 0 :
                            cat_x[:, ch, day] = 0
                    else :
                        cat_x[:, ch, day] = 0
            cat_y = torch.clone(y[None, :])
        else :
            cat_x = torch.cat((cat_x, x[None, :, :]))
            for day in range(x.shape[1]) :
                for ch in range(4) :
                    if sampling_patt[ch] != 0 :
                        if day%sampling_patt[ch] != 0 :
                            cat_x[-1, ch, day] = 0
                    else :
                        cat_x[-1, ch, day] = 0
            cat_y = torch.cat((cat_y, y[None, :]))
    return TensorDataset(cat_x, cat_y)

############################################

def get_topcorner(ax, lim = 0.9) :
    """To get the y position of the axis."""
    return ax.get_ybound()[0]+(ax.get_ybound()[1]-ax.get_ybound()[0])*lim
def get_rightcorner(ax, lim = 0.9) :
    """To get the x position of the axis."""
    return ax.get_xbound()[0]+(ax.get_xbound()[1]-ax.get_xbound()[0])*lim



###################################################################
########################     MODEL     ############################
###################################################################

class MyModel(pl.LightningModule):
    def __init__(self, mean, std):
        super().__init__()
        self.meanTheta = mean
        self.stdTheta = std
        global sampling_patt
        self.conv1 = nn.Conv1d(6, 16, kernel_size = 5, padding = 0)
        self.conv2 = nn.Conv1d(16, 32, kernel_size = 5, padding = 0)
        self.conv3 = nn.Conv1d(32, 64, kernel_size = 5, padding = 0)
        self.conv4 = nn.Conv1d(64, 32, kernel_size = 5, padding = 0)
        self.avgpool1 = nn.AvgPool1d(349)
        self.dense1 = nn.Linear(32, 10)
        self.dropout1 = nn.Dropout(p=0.2)

    def forward(self, x):   
        for n_sample in range(len(sampling_patt)) :
            if sampling_patt[n_sample] > 1 :
                ch_to_sample = (x[:, n_sample, :][:, None, :].unfold(2, 1, 7)[:, :, :, 0]).clone()
                x[:, n_sample, :] = nn.functional.interpolate(input = ch_to_sample, size = [ch_to_sample.shape[-1]*sampling_patt[n_sample]], mode = 'linear')[:, 0, 3:-3]
            elif sampling_patt[n_sample] == 0 :
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
        global global_LR
        optimizer = torch.optim.Adam(self.parameters(), lr = global_LR)
        return optimizer

    def training_step(self, x, y):
        ti = time.time()
        x_train = x[0].to(device)
        y_train = x[1].to(device)
        loss = self.mse_loss(self(x_train), y_train)
        self.log('train_loss', loss, on_epoch=True)
        # sys.stdout.write("\rFin d'une training step en %.2f" %(time.time()-ti) + "s.")
        return loss

    def validation_step(self, x, y):
        global device
        x_valid = x[0].to(device)
        y_valid = x[1].to(device)
        loss = self.mse_loss(self(x_valid), y_valid)
        self.log('valid_loss', loss, on_epoch=True)
        # sys.stdout.write("\rFin d'une validation step en %.2f" %(time.time()-ti) + "s.")
        return loss



####################################################################
########################    TRAINING    ############################
####################################################################
    
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
print("Par défaut, device set to : ", device)

training_dataset = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/DS_train", map_location = device)
validation_dataset = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/DS_Valid", map_location = device)

sampled_training_dataset = data_sampling(training_dataset, sampling_patt = sampling_patt)
sampled_validation_dataset = data_sampling(validation_dataset, sampling_patt = sampling_patt)

mean_x = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/mean_x.pt", map_location = 'cpu')
std_x = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/std_x.pt", map_location = 'cpu')
mean_y = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/mean_y.pt", map_location = 'cpu')
std_y = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/std_y.pt", map_location = 'cpu')	

training_DL = DataLoader(sampled_training_dataset, batch_size=btch_size, shuffle=True, generator = torch.Generator(device))
validation_DL = DataLoader(sampled_validation_dataset, batch_size=btch_size, shuffle=True)

epoch_i = 0

MyNet = MyModel(mean = mean_y, std = std_y)
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


###################################################

trainer = pl.Trainer(check_val_every_n_epoch=10, default_root_dir = name_file+f"version_{nb_version}/", min_epochs = nb_epochs, max_epochs=nb_epochs, callbacks=[checkpoint_callback, checkpoint_callback_2], log_every_n_steps=None)#, log_every_n_steps=50, check_val_every_n_epoch=2
ti_train = time.time()
trainer.fit(model=MyNet, train_dataloaders=training_DL, val_dataloaders=validation_DL)

print("End of the training after %.2f " %(time.time()-ti_train), "seconds.")

trainer.save_checkpoint(name_file+f"version_{nb_version}/final_chkpt.ckpt")
torch.save(MyNet.state_dict(), name_file+f"version_{nb_version}/final_state_dict")
del MyNet

####################################################################
#######################    VALIDATION    ###########################
####################################################################

DS = torch.load(f"Generated_Datasets/DA/Case_{case}/DS_test_NN")
DL = DataLoader(DS, batch_size = 100)
global_f_0, global_delta_0 = torch.load(f"Generated_Datasets/DA/Case_{case}/True_forcings.pt")[:, :, 0], torch.load(f"Generated_Datasets/DA/Case_{case}/True_forcings.pt")[:, :, 1]
global_f_1, global_delta_1 = torch.load(f"Generated_Datasets/DA/Case_{case}/False_forcings.pt")[:, :, 0], torch.load(f"Generated_Datasets/DA/Case_{case}/False_forcings.pt")[:, :, 1]
theta_target = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt")


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

MyNet_val = MyModel.load_from_checkpoint(chckpt_path, mean = mean_y, std = std_y)
MyNet_val.eval()

for x, y in DL :
    theta_got = MyNet_val(x).detach()*std_y+mean_y
torch.save(theta_got, name_file+f"version_{nb_version}/Validation_tensor_theta_pred_DA.pt")

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