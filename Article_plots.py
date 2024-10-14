"""
Last update on June 2024

@author: jlittaye
"""
import sys

dt = 1/2
w_bg = 1e-2
global Q0
Q0 = 4+2.5+1.5+0
####################################################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import pandas as pd
import time
import os

from Functions import Model_NPZD, variational_cost, selection_criteria, norm, get_topcorner, get_rightcorner, set_axis_style, select_version, clear_before_violinplot


#############################################
## Bar plot with criterion
fig, ax = plt.subplots(figsize = [6.4*1.5, 4.8*0.7])
scenarios = ("1/7", "7/7\nCase 0", "1/0", "1/7", "7/7\nCase 1", "1/0", "1/7", "7/7\nCase 2", "1/0", "1/7", "7/7\nCase 3", "1/0")
x = np.array([0, 0.75, 1.5, 3, 3.75, 4.5, 6, 6.75, 7.5, 9, 9.75, 10.5])  # the label locations

width = 0.25  # the width of the bars
bbox = dict(boxstyle="round", fc="#e7effa", ec = '#cfd7e1', alpha = 0.5)# color = '#e7effa', ec = '#cfd7e1'

count = 0
for case in [0, 1, 2, 3] : # 1, 2, 3
    for sampling_patt in [[1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]] : #[1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]
        nb_version_DA = select_version("DA", case = case, sampling_patt = sampling_patt) #len(os.listdir(f"Res/DA_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/"))-1
        nb_version_NN = select_version("NN", case = case, sampling_patt = sampling_patt) #len(os.listdir(f"Res/NN_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/"))-1
        name_file_DA = f"Res/DA_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/version_{nb_version_DA}/"
        name_file_NN = f"Res/NN_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/version_{nb_version_NN}/"
        
        mean_y = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/mean_y.pt", map_location = 'cpu')
        std_y = torch.load("Generated_Datasets/NN/Case_"+str(case)+"/std_y.pt", map_location = 'cpu')
        
        ampl_tensor_DA = torch.load(name_file_DA+"Validation_tensor_ampl.pt", map_location = 'cpu')
        corr_tensor_DA = torch.load(name_file_DA+"Validation_tensor_corr.pt", map_location = 'cpu')
        shift_tensor_DA = torch.load(name_file_DA+"Validation_tensor_shift.pt", map_location = 'cpu')
        theta_ref = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", map_location = 'cpu')
        ## Select the best epoch (where the cost was the lower)
        best_index_DA = torch.argmin(torch.load(name_file_DA+"Tensor_cost.pt")[0, :, 0] + w_bg*torch.load(name_file_DA+f"/Tensor_cost.pt")[0, :, 1])
        theta_DA = torch.load(name_file_DA+"Tensor_params.pt", map_location = 'cpu')[:, :, best_index_DA]

        corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA, theta_error_DA = selection_criteria(corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA, torch.sqrt((theta_ref-theta_DA)**2)/mean_y, crit = [0.01, 5, [0.8, 1.2], 0.15])


        ampl_tensor_NN = torch.load(name_file_NN+"Validation_tensor_ampl.pt", map_location = 'cpu')
        max_corr_tensor_NN = torch.load(name_file_NN+"Validation_tensor_corr.pt", map_location = 'cpu')
        shift_tensor_NN = torch.load(name_file_NN+"Validation_tensor_shift.pt", map_location = 'cpu')
        theta_NN = torch.load(name_file_NN+"Validation_tensor_theta_pred_DA.pt", map_location = 'cpu').detach()
        max_corr_tensor_NN, shift_tensor_NN, ampl_tensor_NN, theta_error_NN = selection_criteria(max_corr_tensor_NN, shift_tensor_NN, ampl_tensor_NN, torch.sqrt((theta_ref-theta_NN)**2)/mean_y, crit = [0.01, 5, [0.8, 1.2], 0.15])

        if not count :
            rects_DA = [ax.bar(x[count], corr_tensor_DA.shape[0], width, color='#6ea0f5', ec = '#155ad1')]
            rects_NN = [ax.bar(x[count] + width, max_corr_tensor_NN.shape[0], width, color='#f5c84c', ec = '#db8607')]
        else :
            rects_DA.append(ax.bar(x[count], corr_tensor_DA.shape[0], width, color='#6ea0f5', ec = '#155ad1'))
            rects_NN.append(ax.bar(x[count] + width, max_corr_tensor_NN.shape[0], width, color='#f5c84c', ec = '#db8607'))

        if len(str(corr_tensor_DA.shape[0])) == 2:
            ax.annotate(text = str(corr_tensor_DA.shape[0]), xy = (x[count]-width/2, corr_tensor_DA.shape[0]+1.5 -(corr_tensor_DA.shape[0] > 93)*7))
        else :
            ax.annotate(text = str(corr_tensor_DA.shape[0]), xy = (x[count]-width/2+0.05, corr_tensor_DA.shape[0]+1.5 -(corr_tensor_DA.shape[0] > 93)*7))
            
        if len(str(max_corr_tensor_NN.shape[0])) == 2:
            ax.annotate(text = str(max_corr_tensor_NN.shape[0]), xy = (x[count]+width/2+0.02, max_corr_tensor_NN.shape[0]+1.5 -(max_corr_tensor_NN.shape[0] > 93)*7))
        else : 
            ax.annotate(text = str(max_corr_tensor_NN.shape[0]), xy = (x[count]+width/2+0.05, max_corr_tensor_NN.shape[0]+1.5 -(max_corr_tensor_NN.shape[0] > 93)*7))
        count += 1

ax.set_ylabel('Calibration success (%)')
ax.set_xticks(x + width, scenarios)
ax.legend([rects_DA[0], rects_NN[0]], ["DA-based", "NN-based"], loc='upper right')
ax.set_ylim(0, 100)
plt.grid(linestyle = ':')
fig.tight_layout()
plt.savefig("Barplot_DA_NN", dpi = 500)
print("\n############ Bar plot done ############\n")

#############################################
## Scatter plot variational cost/nrmse(theta)
cost_tensor = torch.zeros([2, 4, 3])
Error_theta_tensor = torch.zeros([2, 4, 3])


for i in range(4) : # 1, 2, 3
    for j in range(3) :  # [1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]
        case = [0, 1, 2, 3][i]
        sampling_patt = [[1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]][j]
        nb_version_DA = select_version("DA", case = case, sampling_patt = sampling_patt)
        nb_version_NN = select_version("NN", case = case, sampling_patt = sampling_patt)
        name_file_DA = f"Res/DA_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/version_{nb_version_DA}/"
        name_file_NN = f"Res/NN_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/version_{nb_version_NN}/"
        
        best_index_DA = torch.argmin(torch.load(name_file_DA+"Tensor_cost.pt")[0, :, 0] + w_bg*torch.load(name_file_DA+f"/Tensor_cost.pt")[0, :, 1])
        theta_DA = torch.load(name_file_DA+"Tensor_params.pt", map_location = 'cpu')[:, :, 1+best_index_DA]
        x0 = torch.load(name_file_DA+"Tensor_x0.pt", map_location = 'cpu')[:, :, :, 1+(best_index_DA)//10] # /!\ For the provided results x0 has been sample every 10 steps to lighter the file but the acutal DA method samples it every step (so remove //10 to select the optimal x0)

        train_loader = torch.load(f"Generated_Datasets/DA/Case_{case}/TrainLoader.pt", map_location = 'cpu')
        f0, d0 = torch.load(f"Generated_Datasets/DA/Case_{case}/True_forcings.pt", map_location = 'cpu').moveaxis(2, 0)
        f1, d1 = torch.load(f"Generated_Datasets/DA/Case_{case}/False_forcings.pt", map_location = 'cpu').moveaxis(2, 0)
        mask_bruit_obs = torch.load(f"Generated_Datasets/DA/Case_{case}/Obs_matrix.pt", map_location = 'cpu')
        std_obs = torch.std(mask_bruit_obs, dim = (0, 1, 3))

        theta_ref = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", map_location = 'cpu').detach()

        mean_y = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_y.pt", map_location = 'cpu')
        std_y = torch.load(f"Generated_Datasets/NN/Case_{case}/std_y.pt", map_location = 'cpu')
        theta_NN = torch.load(name_file_NN+"Validation_tensor_theta_pred_DA.pt", map_location = 'cpu').detach()
        
        for x, y in train_loader :
            std_mod = torch.std(y, dim = (1, 3))

        model_NN = Model_NPZD(theta_NN)
        model_NN.X0 = x[:, :, 1:, 0].detach()
        model_DA = Model_NPZD(theta_DA)
        model_DA.X0 = x0
        model_NN.eval()
        model_DA.eval()

        with torch.no_grad() :
            for x, y in train_loader : 
                preds_NN = model_NN(x.detach(), f1, d1, dt = dt)
                preds_DA = model_DA(torch.cat((x[:, :, 0], model_DA.X0), dim = 2)[:, :, :, None], f1, d1, dt = dt)

            Error_theta_tensor[0, i, j] = torch.mean(torch.sqrt((theta_ref-theta_DA)**2)/mean_y)
            cost_tensor[0, i, j] = torch.sum(variational_cost(preds_DA.detach(), y.cpu().detach() + mask_bruit_obs, sampling_patt, std_mod = std_mod, std_obs = std_obs)[0].detach() + w_bg*variational_cost(preds_DA.detach(), y.cpu().detach() + mask_bruit_obs, sampling_patt, std_mod = std_mod, std_obs = std_obs)[1].detach())
            Error_theta_tensor[1, i, j] = torch.mean(torch.sqrt((theta_ref-theta_NN)**2)/mean_y)
            cost_tensor[1, i, j] = torch.sum(variational_cost(preds_NN.detach(), y.cpu().detach() + mask_bruit_obs, sampling_patt, std_mod = std_mod, std_obs = std_obs)[0].detach() + w_bg*variational_cost(preds_NN.detach(), y.cpu().detach() + mask_bruit_obs, sampling_patt, std_mod = std_mod, std_obs = std_obs)[1].detach())
        print(f"Calcul fait pour cas {case} - {sampling_patt[0]}/{sampling_patt[1]}/{sampling_patt[2]}/{sampling_patt[3]}")

markers = ['h', 's', '^']
marker_size = [6, 10, 14, 18]
axleg = []
fig, ax = plt.subplots(figsize = [4.8*1.25, 4.8*1])
axleg.append(plt.fill_between(x = [], y1 = 0, y2 = 0, color = '#6c80f5', ec = 'k'))
for i in range(3, -1, -1) :
    for j in range(2, -1, -1) :
        plt.plot([Error_theta_tensor[0, i, j]], [cost_tensor[0, i, j]], color = '#6c80f5', label = "DA-based", marker = markers[j], linestyle = ' ', markersize = marker_size[i], markeredgecolor = 'k')
        plt.plot([Error_theta_tensor[1, i, j]], [cost_tensor[1, i, j]], color = 'orange', label = "NN-based", marker = markers[j], linestyle = ' ', markersize = marker_size[i], markeredgecolor = 'k')
        if (Error_theta_tensor[0, i, j]-Error_theta_tensor[1, i, j]) > 0 :
            plt.quiver(Error_theta_tensor[1, i, j], cost_tensor[1, i, j], (Error_theta_tensor[0, i, j]-Error_theta_tensor[1, i, j]), (cost_tensor[0, i, j]-cost_tensor[1, i, j]), angles='xy', scale_units='xy',scale=1, width=.005, color='k', alpha = 0.5,zorder=2)
        else :
            plt.quiver(Error_theta_tensor[1, i, j], cost_tensor[1, i, j], (Error_theta_tensor[0, i, j]-Error_theta_tensor[1, i, j]), (cost_tensor[0, i, j]-cost_tensor[1, i, j]), angles='xy', scale_units='xy',scale=1, width=.005, color='red', alpha = 0.5,zorder=2)

for i in range(3):
    axleg.append(plt.plot([], [], marker = markers[i], markersize = marker_size[1], color = 'w', markeredgecolor = 'k')[0])
axleg.append(plt.fill_between(x = [], y1 = 0, y2 = 0, color = 'orange', ec = 'k'))
for i in range(4) :
    axleg.append(plt.plot([], [], marker = 'o', color = 'k', markersize = marker_size[i], linestyle = ' ')[0])
    
plt.legend(axleg, ['DA-based', 'Case 0', 'Case 1', 'Case 2', 'Case 3', 'NN-based', '1/7', '7/7', ' 1/0'], ncol = 2, labelspacing = 0.7)
plt.grid()
ax.set_xlabel(r"NRMSE($\theta$)")
ax.set_ylabel(r"Variational cost")
fig.tight_layout()
plt.savefig("Scatterplot_cost_theta", dpi = 500)
print("\n############ Scatter plot done ############\n")


#############################################
## Violin plot for comparing DA to NN 1 by 1
#############################################
## Violin plot for comparing DA to NN 1 by 1
for case in [0, 1, 2, 3] : # 0, 1, 2, 3
    for sampling_patt in [[1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]] : # [1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]
        nb_version_DA = select_version("DA", case = case, sampling_patt = sampling_patt)
        nb_version_NN = select_version("NN", case = case, sampling_patt = sampling_patt)
        name_file_DA = f"Res/DA_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/version_{nb_version_DA}/"
        name_file_NN = f"Res/NN_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/version_{nb_version_NN}/"

        
        ampl_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_ampl.pt", map_location = 'cpu')
        corr_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_corr.pt", map_location = 'cpu')
        shift_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_shift.pt", map_location = 'cpu')
        corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA = clear_before_violinplot(corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA, crit = [1, 1000, 100])
        ampl_tensor_NN = torch.load(name_file_NN+"/Validation_tensor_ampl.pt", map_location = 'cpu')
        corr_tensor_NN = torch.load(name_file_NN+"/Validation_tensor_corr.pt", map_location = 'cpu')
        shift_tensor_NN = torch.load(name_file_NN+"/Validation_tensor_shift.pt", map_location = 'cpu')
        corr_tensor_NN, shift_tensor_NN, ampl_tensor_NN = clear_before_violinplot(corr_tensor_NN, shift_tensor_NN, ampl_tensor_NN, crit = [1, 1000, 100])

        mean_y = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_y.pt", map_location = 'cpu')
        std_y = torch.load(f"Generated_Datasets/NN/Case_{case}/std_y.pt", map_location = 'cpu')
        theta_ref = torch.load(name_file_DA+"/Tensor_thetatarget.pt", map_location = 'cpu')
        best_index = torch.argmin(torch.load(name_file_DA+"Tensor_cost.pt")[0, :, 0] + w_bg*torch.load(name_file_DA+f"/Tensor_cost.pt")[0, :, 1])
        
        theta_DA = torch.load(name_file_DA+"/Tensor_params.pt", map_location = 'cpu')[:, :, best_index+1]
        theta_NN = (torch.load(name_file_NN+"/Validation_tensor_theta_pred_DA.pt", map_location = 'cpu')).detach().to('cpu')


        fig = plt.figure(figsize=(6.4*2.1, 6.4*0.7*2))
        fig.suptitle(f"Scenario: Case {case} - {sampling_patt[0]}/{sampling_patt[-1]}")
        grid = fig.add_gridspec(ncols=3, nrows = 2)
        ax_corr = fig.add_subplot(grid[0, 0])
        ax_shift = fig.add_subplot(grid[0, 1])
        ax_ampl = fig.add_subplot(grid[0, 2])
        ax_param = fig.add_subplot(grid[1, :])

        
        ax_corr.set_title('Correlation')
        # ax_corr.set_ylim(0.965, 1.0035)
        ax_corr.set_ylim(0.9, 1.01)
        ax_corr_Z = ax_corr.twinx()
        ax_corr_Z.set_ylim(0.5, 1.05)
        
        ax_shift.set_title('Shift (day)')
        ax_shift.set_ylim(-0.2, 8)
        ax_shift_Z = ax_shift.twinx()
        ax_shift_Z.set_ylim(-1.2, 48)
        
        ax_ampl.set_title("Amplitude ratio")
        ax_ampl.set_ylim(0.8, 1.2)
        ax_ampl_Z = ax_ampl.twinx()
        ax_ampl_Z.set_ylim(0, 2)
        
        ax_param.set_title("Biogeochemical parameter error")
        ax_param.set_ylim(-0.05, .5)


        corr_tensor_DA_Z = torch.cat((torch.ones([100, 3])*1000, corr_tensor_DA[:, 2:3]), dim = 1)
        corr_tensor_DA = torch.cat((corr_tensor_DA[:, 0:1], corr_tensor_DA[:, 1:2], corr_tensor_DA[:, 3:4], torch.ones([100, 1])*1000), dim = 1)
        corr_tensor_NN_Z = torch.cat((torch.ones([100, 3])*1000, corr_tensor_NN[:, 2:3]), dim = 1)
        corr_tensor_NN = torch.cat((corr_tensor_NN[:, 0:1], corr_tensor_NN[:, 1:2], corr_tensor_NN[:, 3:4], torch.ones([100, 1])*1000), dim = 1)
    
        shift_tensor_DA_Z = torch.cat((torch.ones([100, 3])*1000, shift_tensor_DA[:, 2:3]), dim = 1)
        shift_tensor_DA = torch.cat((shift_tensor_DA[:, 0:1], shift_tensor_DA[:, 1:2], shift_tensor_DA[:, 3:4], torch.ones([100, 1])*1000), dim = 1)
        shift_tensor_NN_Z = torch.cat((torch.ones([100, 3])*1000, shift_tensor_NN[:, 2:3]), dim = 1)
        shift_tensor_NN = torch.cat((shift_tensor_DA[:, 0:1], shift_tensor_NN[:, 1:2], shift_tensor_NN[:, 3:4], torch.ones([100, 1])*1000), dim = 1)
    
        ampl_tensor_DA_Z = torch.cat((torch.ones([100, 3])*1000, ampl_tensor_DA[:, 2:3]), dim = 1)
        ampl_tensor_DA = torch.cat((ampl_tensor_DA[:, 0:1], ampl_tensor_DA[:, 1:2], ampl_tensor_DA[:, 3:4], torch.ones([100, 1])*1000), dim = 1)
        ampl_tensor_NN_Z = torch.cat((torch.ones([100, 3])*1000, ampl_tensor_NN[:, 2:3]), dim = 1)
        ampl_tensor_NN = torch.cat((ampl_tensor_NN[:, 0:1], ampl_tensor_NN[:, 1:2], ampl_tensor_NN[:, 3:4], torch.ones([100, 1])*1000), dim = 1)
        
        
        r_DA_Z = ax_ampl_Z.violinplot(ampl_tensor_DA_Z.transpose(0, 1))
        r_DA = ax_ampl.violinplot(ampl_tensor_DA.transpose(0, 1))
        r_NN_Z = ax_ampl_Z.violinplot(ampl_tensor_NN_Z.transpose(0, 1))
        r_NN = ax_ampl.violinplot(ampl_tensor_NN.transpose(0, 1))
        
        c_DA_Z = ax_corr_Z.violinplot(corr_tensor_DA_Z.transpose(0, 1))
        c_DA = ax_corr.violinplot(corr_tensor_DA.transpose(0, 1))
        c_NN_Z = ax_corr_Z.violinplot(corr_tensor_NN_Z.transpose(0, 1))
        c_NN = ax_corr.violinplot(corr_tensor_NN.transpose(0, 1))
        
        s_DA_Z = ax_shift_Z.violinplot(abs(shift_tensor_DA_Z.transpose(0, 1)))
        s_DA = ax_shift.violinplot(abs(shift_tensor_DA.transpose(0, 1)))
        s_NN_Z = ax_shift_Z.violinplot(abs(shift_tensor_NN_Z.transpose(0, 1)))
        s_NN = ax_shift.violinplot(abs(shift_tensor_NN.transpose(0, 1)))
        
        p_DA = ax_param.violinplot((abs(theta_DA-theta_ref)/mean_y).transpose(0, 1))
        p_NN = ax_param.violinplot((abs(theta_NN-theta_ref)/mean_y).transpose(0, 1))

        ax_param.set_xticks(torch.arange(1, 11, 1), ["$K_N$", "$R_m$", "$g$", r"$\lambda$", r"$\epsilon$", r"$\alpha$", r"$\beta$", "$r$", r"$\phi$", '$S_w$'])
        for v_base_DA in [r_DA, r_DA_Z, c_DA, c_DA_Z, s_DA, s_DA_Z, p_DA] :
            for pc in v_base_DA['bodies']:
                pc.set_facecolor('#6ea0f5')
                pc.set_edgecolor('#155ad1')
                pc.set_linewidth(2)
        for v_base_NN in [r_NN, r_NN_Z, c_NN, c_NN_Z, s_NN, s_NN_Z, p_NN] :
            for pc in v_base_NN['bodies']:
                pc.set_facecolor('#f5c84c')
                pc.set_edgecolor('#db8607')
                pc.set_linewidth(2)
                pc.set_alpha(0.5)
                
        ax_corr.set_ylabel("N, P, D metric")
        ax_corr_Z.set_ylabel("Z metric")
        ax_shift.set_ylabel("N, P, D metric")
        ax_shift_Z.set_ylabel("Z metric")
        ax_ampl.set_ylabel("N, P, D metric")
        ax_ampl_Z.set_ylabel("Z metric")
        
        ax_corr_Z.text(s = "a)", x = get_rightcorner(ax_corr_Z, 0.9), y = get_topcorner(ax_corr_Z, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
        ax_shift_Z.text(s = "b)", x = get_rightcorner(ax_shift_Z, 0.9), y = get_topcorner(ax_shift_Z, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
        ax_ampl_Z.text(s = "c)", x = get_rightcorner(ax_ampl_Z, 0.9), y = get_topcorner(ax_ampl_Z, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
        ax_param.text(s = "d)", x = get_rightcorner(ax_param, 0.96), y = get_topcorner(ax_param, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
        
        ax_shift_Z.set_yticks(ticks = [0, 6, 12, 18, 24, 30, 36, 42, 48], labels = ["0", "6", "12", "18", "24", "30", "36", "42", "48"])
        
        DA_legend = mpatches.Patch(color='#6ea0f5', ec = '#155ad1', label='DA-based')
        NN_legend = mpatches.Patch(color='#f5c84c', ec = '#db8607', label='NN-based')
        ax_ampl.legend(handles=[DA_legend, NN_legend], loc = 'upper left')
        for ax in [ax_corr, ax_shift, ax_ampl] :
            ax.grid()
            set_axis_style(ax, ["N", "P", "D", "Z"])
        ax_param.grid()
        plt.subplots_adjust(bottom=0.15, wspace=0.15)
        fig.tight_layout()
        

        plt.savefig(f"Violinplot_comp_DANN_case{case}_{sampling_patt[0]}_{sampling_patt[-1]}", dpi = 500)
    
