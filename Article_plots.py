"""
Last update on April 2025

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

from Functions import function_NPZD, Model_NPZD, variational_cost, selection_criteria, norm, get_topcorner, get_rightcorner, set_axis_style, select_version, clear_before_violinplot, validation_params


#############################################
########## Bar plot with criterion ##########
#############################################
NN_method = 'CNN'
fig, ax = plt.subplots(figsize = [6.4*1.5, 4.8*0.7])
scenarios = ("1/7", "7/7\nCase 0", "1/0", "1/7", "7/7\nCase 1", "1/0", "1/7", "7/7\nCase 2", "1/0", "1/7", "7/7\nCase 3", "1/0")
x = np.array([0, 0.75, 1.5, 3, 3.75, 4.5, 6, 6.75, 7.5, 9, 9.75, 10.5])  # the label locations

width = 0.25  # the width of the bars
bbox = dict(boxstyle="round", fc="#e7effa", ec = '#cfd7e1', alpha = 0.5)# color = '#e7effa', ec = '#cfd7e1'

count = 0
for case in [0, 1, 2, 3] : # 1, 2, 3
    for sampling_patt in [[1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]] : #[1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]
        nb_version_DA = select_version("DA", case = case, sampling_patt = sampling_patt)
        nb_version_NN = select_version("NN", case = case, sampling_patt = sampling_patt, architecture=NN_method)
        name_file_DA = f"Res/DA_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/version_{nb_version_DA}/"
        name_file_NN = f"Res/NN_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}_{NN_method}/version_{nb_version_NN}/"
        
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
#############################################
cost_tensor = torch.zeros([2, 4, 3])
Error_theta_tensor = torch.zeros([2, 4, 3])


for i in range(4) : # 1, 2, 3
    for j in range(3) :  # [1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]
        case = [0, 1, 2, 3][i]
        sampling_patt = [[1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]][j]
        nb_version_DA = select_version("DA", case = case, sampling_patt = sampling_patt)
        nb_version_NN = select_version("NN", case = case, sampling_patt = sampling_patt, architecture=NN_method)
        name_file_DA = f"Res/DA_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/version_{nb_version_DA}/"
        name_file_NN = f"Res/NN_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}_{NN_method}/version_{nb_version_NN}/"
        
        best_index_DA = torch.argmin(torch.load(name_file_DA+"Tensor_cost.pt", weights_only=False)[0, :, 0] + w_bg*torch.load(name_file_DA+f"/Tensor_cost.pt", weights_only=False)[0, :, 1])
        theta_DA = torch.load(name_file_DA+"Tensor_params.pt", map_location = 'cpu', weights_only=False)[:, :, 1+best_index_DA]
        x0 = torch.load(name_file_DA+"Tensor_x0.pt", map_location = 'cpu', weights_only=False)[:, :, :, 1+(best_index_DA)//10] # /!\ For the provided results x0 has been sample every 10 steps to lighter the file but the acutal DA method samples it every step (so remove //10 to select the optimal x0)

        train_loader = torch.load(f"Generated_Datasets/DA/Case_{case}/TrainLoader.pt", map_location = 'cpu', weights_only=False)
        f0, d0 = torch.load(f"Generated_Datasets/DA/Case_{case}/True_forcings.pt", map_location = 'cpu', weights_only=False).moveaxis(2, 0)
        f1, d1 = torch.load(f"Generated_Datasets/DA/Case_{case}/False_forcings.pt", map_location = 'cpu', weights_only=False).moveaxis(2, 0)
        mask_bruit_obs = torch.load(f"Generated_Datasets/DA/Case_{case}/Obs_matrix.pt", map_location = 'cpu', weights_only=False)
        std_obs = torch.std(mask_bruit_obs, dim = (0, 1, 3))

        theta_ref = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", map_location = 'cpu', weights_only=False).detach()

        mean_y = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_y.pt", map_location = 'cpu', weights_only=False)
        std_y = torch.load(f"Generated_Datasets/NN/Case_{case}/std_y.pt", map_location = 'cpu', weights_only=False)
        theta_NN = torch.load(name_file_NN+"Validation_tensor_theta_pred_DA.pt", map_location = 'cpu', weights_only=False).detach()
        
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
        print(f"Done for case {case} - {sampling_patt[0]}/{sampling_patt[1]}/{sampling_patt[2]}/{sampling_patt[3]}")

markers = ['h', 's', '^']
marker_size = [6, 10, 14, 18]
axleg = []
fig, ax = plt.subplots(figsize = [4.8*1.25, 4.8*1])
for i in range(3, -1, -1) :
    for j in range(2, -1, -1) :
        plt.plot([Error_theta_tensor[0, i, j]], [cost_tensor[0, i, j]], color = '#6c80f5', label = "DA-based", marker = markers[j], linestyle = ' ', markersize = marker_size[i], markeredgecolor = 'k')
        plt.plot([Error_theta_tensor[1, i, j]], [cost_tensor[1, i, j]], color = 'orange', label = "NN-based", marker = markers[j], linestyle = ' ', markersize = marker_size[i], markeredgecolor = 'k')
        if (Error_theta_tensor[0, i, j]-Error_theta_tensor[1, i, j]) > 0 :
            plt.quiver(Error_theta_tensor[1, i, j], cost_tensor[1, i, j], (Error_theta_tensor[0, i, j]-Error_theta_tensor[1, i, j]), (cost_tensor[0, i, j]-cost_tensor[1, i, j]), angles='xy', scale_units='xy',scale=1, width=.005, color='k', alpha = 0.5,zorder=2)
        else :
            plt.quiver(Error_theta_tensor[1, i, j], cost_tensor[1, i, j], (Error_theta_tensor[0, i, j]-Error_theta_tensor[1, i, j]), (cost_tensor[0, i, j]-cost_tensor[1, i, j]), angles='xy', scale_units='xy',scale=1, width=.005, color='red', alpha = 0.5,zorder=2)


axleg.append(plt.fill_between(x = [], y1 = 0, y2 = 0, color = '#6c80f5', ec = 'k'))
for i in range(4) :
    axleg.append(plt.plot([], [], marker = 'o', color = 'k', markersize = marker_size[i], linestyle = ' ')[0])
axleg.append(plt.fill_between(x = [], y1 = 0, y2 = 0, color = 'orange', ec = 'k'))
for i in range(3):
    axleg.append(plt.plot([], [], marker = markers[i], markersize = marker_size[1], color = 'w', markeredgecolor = 'k')[0])
    
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
for case in [0, 1, 2, 3] : # 0, 1, 2, 3
    for sampling_patt in [[1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]] : # [1, 1, 7, 7], [7, 7, 7, 7], [1, 1, 0, 0]
        nb_version_DA = select_version("DA", case = case, sampling_patt = sampling_patt)
        nb_version_NN = select_version("NN", case = case, sampling_patt = sampling_patt, architecture=NN_method)
        name_file_DA = f"Res/DA_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}/version_{nb_version_DA}/"
        name_file_NN = f"Res/NN_{sampling_patt[0]}d_{sampling_patt[1]}d_{sampling_patt[2]}d_{sampling_patt[3]}d_case{case}_{NN_method}/version_{nb_version_NN}/"

        
        ampl_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_ampl.pt", map_location = 'cpu', weights_only=False)
        corr_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_corr.pt", map_location = 'cpu', weights_only=False)
        shift_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_shift.pt", map_location = 'cpu', weights_only=False)
        corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA = clear_before_violinplot(corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA, crit = [1, 1000, 100])
        ampl_tensor_NN = torch.load(name_file_NN+"/Validation_tensor_ampl.pt", map_location = 'cpu', weights_only=False)
        corr_tensor_NN = torch.load(name_file_NN+"/Validation_tensor_corr.pt", map_location = 'cpu', weights_only=False)
        shift_tensor_NN = torch.load(name_file_NN+"/Validation_tensor_shift.pt", map_location = 'cpu', weights_only=False)
        corr_tensor_NN, shift_tensor_NN, ampl_tensor_NN = clear_before_violinplot(corr_tensor_NN, shift_tensor_NN, ampl_tensor_NN, crit = [1, 1000, 100])

        mean_y = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_y.pt", map_location = 'cpu', weights_only=False)
        std_y = torch.load(f"Generated_Datasets/NN/Case_{case}/std_y.pt", map_location = 'cpu', weights_only=False)
        theta_ref = torch.load(name_file_DA+"/Tensor_thetatarget.pt", map_location = 'cpu', weights_only=False)
        best_index = torch.argmin(torch.load(name_file_DA+"Tensor_cost.pt", weights_only=False)[0, :, 0] + w_bg*torch.load(name_file_DA+f"/Tensor_cost.pt", weights_only=False)[0, :, 1])
        
        theta_DA = torch.load(name_file_DA+"/Tensor_params.pt", map_location = 'cpu', weights_only=False)[:, :, best_index+1]
        theta_NN = (torch.load(name_file_NN+"/Validation_tensor_theta_pred_DA.pt", map_location = 'cpu', weights_only=False)).detach().to('cpu')


        fig = plt.figure(figsize=(6.4*2.1, 6.4*0.7*2))
        fig.suptitle(f"Scenario: Case {case} - {sampling_patt[0]}/{sampling_patt[-1]}")
        grid = fig.add_gridspec(ncols=3, nrows = 2)
        ax_corr = fig.add_subplot(grid[0, 0])
        ax_shift = fig.add_subplot(grid[0, 1])
        ax_ampl = fig.add_subplot(grid[0, 2])
        ax_param = fig.add_subplot(grid[1, :])

        
        ax_corr.set_title('Correlation')
        ax_corr_Z = ax_corr.twinx()
        
        ax_shift.set_title('Shift (day)')
        ax_shift_Z = ax_shift.twinx()
        
        ax_ampl.set_title("Amplitude ratio")
        ax_ampl_Z = ax_ampl.twinx()
        
        ax_param.set_title("Biogeochemical parameter error")

        corr_tensor_DA_Z = torch.cat((torch.ones([100, 3])*torch.nan, corr_tensor_DA[:, 2:3]), dim = 1)
        corr_tensor_DA = torch.cat((corr_tensor_DA[:, 0:1], corr_tensor_DA[:, 1:2], corr_tensor_DA[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)
        corr_tensor_NN_Z = torch.cat((torch.ones([100, 3])*torch.nan, corr_tensor_NN[:, 2:3]), dim = 1)
        corr_tensor_NN = torch.cat((corr_tensor_NN[:, 0:1], corr_tensor_NN[:, 1:2], corr_tensor_NN[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)
    
        shift_tensor_DA_Z = torch.cat((torch.ones([100, 3])*torch.nan, shift_tensor_DA[:, 2:3]), dim = 1)
        shift_tensor_DA = torch.cat((shift_tensor_DA[:, 0:1], shift_tensor_DA[:, 1:2], shift_tensor_DA[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)
        shift_tensor_NN_Z = torch.cat((torch.ones([100, 3])*torch.nan, shift_tensor_NN[:, 2:3]), dim = 1)
        shift_tensor_NN = torch.cat((shift_tensor_DA[:, 0:1], shift_tensor_NN[:, 1:2], shift_tensor_NN[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)
    
        ampl_tensor_DA_Z = torch.cat((torch.ones([100, 3])*torch.nan, ampl_tensor_DA[:, 2:3]), dim = 1)
        ampl_tensor_DA = torch.cat((ampl_tensor_DA[:, 0:1], ampl_tensor_DA[:, 1:2], ampl_tensor_DA[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)
        ampl_tensor_NN_Z = torch.cat((torch.ones([100, 3])*torch.nan, ampl_tensor_NN[:, 2:3]), dim = 1)
        ampl_tensor_NN = torch.cat((ampl_tensor_NN[:, 0:1], ampl_tensor_NN[:, 1:2], ampl_tensor_NN[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)
        
        
        r_DA_Z = ax_ampl_Z.violinplot(ampl_tensor_DA_Z)
        r_DA = ax_ampl.violinplot(ampl_tensor_DA)
        r_NN_Z = ax_ampl_Z.violinplot(ampl_tensor_NN_Z)
        r_NN = ax_ampl.violinplot(ampl_tensor_NN)
        
        c_DA_Z = ax_corr_Z.violinplot(corr_tensor_DA_Z)
        c_DA = ax_corr.violinplot(corr_tensor_DA)
        c_NN_Z = ax_corr_Z.violinplot(corr_tensor_NN_Z)
        c_NN = ax_corr.violinplot(corr_tensor_NN)
        
        s_DA_Z = ax_shift_Z.violinplot(abs(shift_tensor_DA_Z))
        s_DA = ax_shift.violinplot(abs(shift_tensor_DA))
        s_NN_Z = ax_shift_Z.violinplot(abs(shift_tensor_NN_Z))
        s_NN = ax_shift.violinplot(abs(shift_tensor_NN))
        
        p_DA = ax_param.violinplot(((theta_DA-theta_ref)/mean_y))
        p_NN = ax_param.violinplot(((theta_NN-theta_ref)/mean_y))

        ax_param.set_xticks(torch.arange(1, 11, 1), [r"$\chi$", r"$\rho$", r"$\gamma$", r"$\lambda$", r"$\epsilon$", r"$\alpha$", r"$\beta$", r"$\eta$", r"$\varphi$", r'$\zeta$'])
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


#############################################
## Violin plot for comparing DA - case 0,1,2,3
#############################################

torch.set_default_device('cpu')
pattern = [1, 1, 7, 7]
colors = [['#85d1d6', '#0ccaa5'], ['#1d17d6', '#0a0769'], ["#a055b5", "#8f04b5"], ["#ff6200", "#ad4605"]] # fc, ec : light blue, dark blue, purple

fig = plt.figure(figsize=(6.4*2.1, 6.4*0.7*2))
grid = fig.add_gridspec(nrows = 2, ncols=3)
ax_corr = fig.add_subplot(grid[0, 0])
ax_shift = fig.add_subplot(grid[0, 1])
ax_ampl = fig.add_subplot(grid[0, 2])
ax_param = fig.add_subplot(grid[1, :])

ax_corr.set_title('Correlation')
ax_corr_Z = ax_corr.twinx()

ax_shift.set_title('Shift (day)')
ax_shift_Z = ax_shift.twinx()

ax_ampl.set_title("Amplitude ratio")
ax_ampl_Z = ax_ampl.twinx()

ax_param.set_title("Biogeochemical parameter error")
ax_param.set_ylim(-0.5, .5)
c_DA, s_DA, r_DA, v_DA = [], [], [], []
handles_leg = []
compteur = 0
w = 0.01
for case in [3, 2, 1, 0] :
    name_file_DA = f"Res/DA_{pattern[0]}d_{pattern[1]}d_{pattern[2]}d_{pattern[3]}d_case{case}"
    nb_version_DA = select_version("DA", case = case, sampling_patt = pattern)
    name_file_DA += f"/version_{nb_version_DA}"
    ## Load the data
    corr_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_corr.pt", map_location = 'cpu')
    shift_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_shift.pt", map_location = 'cpu')
    ampl_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_ampl"+".pt", map_location = 'cpu')
    
    best_index = torch.argmin(torch.load(name_file_DA+f"/Tensor_cost.pt")[0, :, 0] + w*torch.load(name_file_DA+f"/Tensor_cost.pt")[0, :, 1])
    theta_DA = torch.load(name_file_DA+"/Tensor_params.pt", map_location = 'cpu')[:, :, best_index+1]
    theta_ref = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", map_location = 'cpu')
    ## Pre-treat the data
    corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA = clear_before_violinplot(corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA)

    corr_tensor_DA_Z = torch.cat((torch.ones([100, 3])*torch.nan, corr_tensor_DA[:, 2:3]), dim = 1)
    corr_tensor_DA = torch.cat((corr_tensor_DA[:, 0:1], corr_tensor_DA[:, 1:2], corr_tensor_DA[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)

    shift_tensor_DA_Z = torch.cat((torch.ones([100, 3])*torch.nan, shift_tensor_DA[:, 2:3]), dim = 1)
    shift_tensor_DA = torch.cat((shift_tensor_DA[:, 0:1], shift_tensor_DA[:, 1:2], shift_tensor_DA[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)

    ampl_tensor_DA_Z = torch.cat((torch.ones([100, 3])*torch.nan, ampl_tensor_DA[:, 2:3]), dim = 1)
    ampl_tensor_DA = torch.cat((ampl_tensor_DA[:, 0:1], ampl_tensor_DA[:, 1:2], ampl_tensor_DA[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)

    ## Display the violin plots
    r_DA.append(ax_ampl_Z.violinplot(ampl_tensor_DA_Z, showextrema=False))
    r_DA.append(ax_ampl.violinplot(ampl_tensor_DA, showextrema=False))
    
    c_DA.append(ax_corr_Z.violinplot(corr_tensor_DA_Z, showextrema=False))
    c_DA.append(ax_corr.violinplot(corr_tensor_DA, showextrema=False))
    
    s_DA.append(ax_shift_Z.violinplot(abs(shift_tensor_DA_Z), showextrema=False))
    s_DA.append(ax_shift.violinplot(abs(shift_tensor_DA), showextrema=False))
    
    v_DA.append(ax_param.violinplot(((theta_DA-theta_ref)/mean_y), showextrema=False))

    ## Manage the violin characteristics and edges
    for pc in v_DA[compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for i in range(10) :
        ax_param.vlines(i+1, ((theta_DA-theta_ref)/mean_y)[:, i].min().item(), ((theta_DA-theta_ref)/mean_y)[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
        ax_param.hlines(((theta_DA-theta_ref)/mean_y)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
        ax_param.hlines(((theta_DA-theta_ref)/mean_y)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)

    for pc in c_DA[2*compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for pc in c_DA[2*compteur+1]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for i in range(3) :
        ax_corr.vlines(i+1, corr_tensor_DA[:, i].min().item(), corr_tensor_DA[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
        ax_corr.hlines(corr_tensor_DA[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
        ax_corr.hlines(corr_tensor_DA[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    i = 3
    ax_corr_Z.vlines(i+1, corr_tensor_DA_Z[:, i].min().item(), corr_tensor_DA_Z[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
    ax_corr_Z.hlines(corr_tensor_DA_Z[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    ax_corr_Z.hlines(corr_tensor_DA_Z[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    
    for pc in s_DA[2*compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for pc in s_DA[2*compteur+1]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for i in range(3) :
        ax_shift.vlines(i+1, abs(shift_tensor_DA)[:, i].min().item(), abs(shift_tensor_DA)[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
        ax_shift.hlines(abs(shift_tensor_DA)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], lw=1)
        ax_shift.hlines(abs(shift_tensor_DA)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], lw=1)
    i = 3
    ax_shift_Z.vlines(i+1, abs(shift_tensor_DA_Z)[:, i].min().item(), abs(shift_tensor_DA_Z)[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
    ax_shift_Z.hlines(abs(shift_tensor_DA_Z)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], lw=1)
    ax_shift_Z.hlines(abs(shift_tensor_DA_Z)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], lw=1)
        
    for pc in r_DA[2*compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for pc in r_DA[2*compteur+1]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for i in range(3) :
        ax_ampl.vlines(i+1, ampl_tensor_DA[:, i].min().item(), ampl_tensor_DA[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
        ax_ampl.hlines(ampl_tensor_DA[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
        ax_ampl.hlines(ampl_tensor_DA[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    i = 3
    ax_ampl_Z.vlines(i+1, ampl_tensor_DA_Z[:, i].min().item(), ampl_tensor_DA_Z[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
    ax_ampl_Z.hlines(ampl_tensor_DA_Z[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    ax_ampl_Z.hlines(ampl_tensor_DA_Z[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)

    handles_leg.append(mpatches.Patch(color=colors[compteur][0], ec = colors[compteur][1], alpha=(6-compteur)/6, linewidth=3, label=f'Case {case}'))

    compteur += 1

ax_corr_Z.text(s = "a)", x = get_rightcorner(ax_corr, 0.9), y = get_topcorner(ax_corr_Z, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
ax_shift_Z.text(s = "b)", x = get_rightcorner(ax_shift_Z, 0.9), y = get_topcorner(ax_shift_Z, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
ax_ampl_Z.text(s = "c)", x = get_rightcorner(ax_ampl_Z, 0.9), y = get_topcorner(ax_ampl_Z, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
ax_param.text(s = "d)", x = get_rightcorner(ax_param, 0.96), y = get_topcorner(ax_param, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)

ax_corr_Z.set_axis_off()
# ax_shift_Z.set_axis_off()
# ax_corr.set_ylabel("N, P, D metric")
# ax_corr_Z.set_ylabel("Z metric")
ax_shift_Z.set_axis_off()
# ax_shift.set_ylabel("N, P, D metric")
# ax_shift_Z.set_ylabel("Z metric")
ax_ampl.set_ylabel("N, P, D metric")
ax_ampl_Z.set_ylabel("Z metric")
# ax_shift_Z.set_yticks([0, 2, 4, 6, 8], ["0", "2", "4", "6", "8"])
# ax_shift.set_yticks([0, 2, 4, 6, 8], ["0", "2", "4", "6", "8"])

ax_param.legend(handles=handles_leg, loc = 'upper center', fontsize = 14, ncols = 2)
for ax in [ax_corr, ax_shift, ax_ampl] :
    ax.grid()
    ax.set_xticks([1, 2, 3, 4], ["N", "P", "D", "Z"])

ax_param.grid()
ax_param.set_xticks(torch.arange(1, 11, 1), ["$\\chi$", "$\\rho$", "$\\gamma$", "$\\lambda$", "$\\epsilon$", "$\\alpha$", "$\\beta$", "$\\eta$", "$\\varphi$", '$\\zeta$'])

plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14) #fontsize of the x and y labels
plt.rc('xtick', labelsize=14) #fontsize of the x tick labels
plt.rc('ytick', labelsize=14) #fontsize of the y tick labels

plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig(f"Violinplot_DA_cas0123_{pattern[0]}{pattern[-1]}", dpi = 500)


################################################
## Violin plot for comparing DA - 1/7, 7/7, 1/0
################################################

t_tensor = torch.arange(0, 365*5, 1/24)

case = 1
w = 1e-2
colors = [['#85d1d6', '#0ccaa5'], ['#1d17d6', '#0a0769'], ["#bf0817", "#78050e"]]

fig = plt.figure(figsize=(6.4*2.1, 6.4*0.7*2))
grid = fig.add_gridspec(nrows = 2, ncols=3)
ax_corr = fig.add_subplot(grid[0, 0])
ax_shift = fig.add_subplot(grid[0, 1])
ax_ampl = fig.add_subplot(grid[0, 2])
ax_param = fig.add_subplot(grid[1, :])

ax_corr.set_title('Correlation')
ax_corr_Z = ax_corr.twinx()

ax_shift.set_title('Shift (day)')
ax_shift_Z = ax_shift.twinx()

ax_ampl.set_title("Amplitude ratio")
ax_ampl_Z = ax_ampl.twinx()

ax_param.set_title("Biogeochemical parameter error")
ax_param.set_ylim(-0.5, .5)
c_DA, s_DA, r_DA, v_DA = [], [], [], []
nmse = []
handles_leg = []
compteur = 0
for pattern in [[1, 1, 0, 0], [7, 7, 7, 7], [1, 1, 7, 7]] : # [1, 1, 0, 0], [7, 7, 7, 7], [1, 1, 7, 7] , [0, 1, 0, 0], [0, 7, 0, 0], [0, 7, 7, 7] [0, 7, 0, 7], [7, 7, 0, 7], [7, 7]
    
    std_y = torch.load(f"Generated_Datasets/NN/Case_{case}/std_y.pt", map_location = 'cpu', weights_only=False)
    mean_y = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_y.pt", map_location = 'cpu', weights_only=False)
    
    name_file_DA = f"Res/DA_{pattern[0]}d_{pattern[1]}d_{pattern[2]}d_{pattern[3]}d_case{case}"
    nb_version_DA = select_version("DA", case = case, sampling_patt = pattern)
    name_file_DA += f"/version_{nb_version_DA}"
    
    corr_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_corr.pt", map_location = 'cpu', weights_only=False)
    shift_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_shift.pt", map_location = 'cpu', weights_only=False)
    ampl_tensor_DA = torch.load(name_file_DA+"/Validation_tensor_ampl.pt", map_location = 'cpu', weights_only=False)
    best_index = torch.argmin(torch.load(name_file_DA+f"/Tensor_cost.pt", weights_only=False)[0, :, 0] + w*torch.load(name_file_DA+f"/Tensor_cost.pt", weights_only=False)[0, :, 1])
    theta_DA = torch.load(name_file_DA+"/Tensor_params.pt", map_location = 'cpu', weights_only=False)[:, :, best_index+1]
    # theta_DA = torch.load(name_file_DA+"/Validation_tensor_theta_pred_DA.pt", map_location = 'cpu')

    theta_ref = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", map_location = 'cpu', weights_only=False)
    
    corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA = clear_before_violinplot(corr_tensor_DA, shift_tensor_DA, ampl_tensor_DA)

    corr_tensor_DA_Z = torch.cat((torch.ones([100, 3])*torch.nan, corr_tensor_DA[:, 2:3]), dim = 1)
    corr_tensor_DA = torch.cat((corr_tensor_DA[:, 0:1], corr_tensor_DA[:, 1:2], corr_tensor_DA[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)

    shift_tensor_DA_Z = torch.cat((torch.ones([100, 3])*torch.nan, shift_tensor_DA[:, 2:3]), dim = 1)
    shift_tensor_DA = torch.cat((shift_tensor_DA[:, 0:1], shift_tensor_DA[:, 1:2], shift_tensor_DA[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)

    ampl_tensor_DA_Z = torch.cat((torch.ones([100, 3])*torch.nan, ampl_tensor_DA[:, 2:3]), dim = 1)
    ampl_tensor_DA = torch.cat((ampl_tensor_DA[:, 0:1], ampl_tensor_DA[:, 1:2], ampl_tensor_DA[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)

    
    r_DA.append(ax_ampl_Z.violinplot(ampl_tensor_DA_Z, showextrema=False))
    r_DA.append(ax_ampl.violinplot(ampl_tensor_DA, showextrema=False))
    
    c_DA.append(ax_corr_Z.violinplot(corr_tensor_DA_Z, showextrema=False))
    c_DA.append(ax_corr.violinplot(corr_tensor_DA, showextrema=False))
    
    s_DA.append(ax_shift_Z.violinplot(abs(shift_tensor_DA_Z), showextrema=False))
    s_DA.append(ax_shift.violinplot(abs(shift_tensor_DA), showextrema=False))
    
    v_DA.append(ax_param.violinplot(((theta_DA-theta_ref)/mean_y), showextrema=False))

    for pc in v_DA[compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for i in range(10) :
        ax_param.vlines(i+1, ((theta_DA-theta_ref)/mean_y)[:, i].min().item(), ((theta_DA-theta_ref)/mean_y)[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
        ax_param.hlines(((theta_DA-theta_ref)/mean_y)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
        ax_param.hlines(((theta_DA-theta_ref)/mean_y)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)

    for pc in c_DA[2*compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for pc in c_DA[2*compteur+1]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for i in range(3) :
        ax_corr.vlines(i+1, corr_tensor_DA[:, i].min().item(), corr_tensor_DA[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
        ax_corr.hlines(corr_tensor_DA[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
        ax_corr.hlines(corr_tensor_DA[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    i = 3
    ax_corr_Z.vlines(i+1, corr_tensor_DA_Z[:, i].min().item(), corr_tensor_DA_Z[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
    ax_corr_Z.hlines(corr_tensor_DA_Z[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    ax_corr_Z.hlines(corr_tensor_DA_Z[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    
    for pc in s_DA[2*compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for pc in s_DA[2*compteur+1]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for i in range(3) :
        ax_shift.vlines(i+1, abs(shift_tensor_DA)[:, i].min().item(), abs(shift_tensor_DA)[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
        ax_shift.hlines(abs(shift_tensor_DA)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], lw=1)
        ax_shift.hlines(abs(shift_tensor_DA)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], lw=1)
    i = 3
    ax_shift_Z.vlines(i+1, abs(shift_tensor_DA_Z)[:, i].min().item(), abs(shift_tensor_DA_Z)[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
    ax_shift_Z.hlines(abs(shift_tensor_DA_Z)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], lw=1)
    ax_shift_Z.hlines(abs(shift_tensor_DA_Z)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], lw=1)
        
    for pc in r_DA[2*compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for pc in r_DA[2*compteur+1]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
        pc.set_alpha=(6-compteur)/6
    for i in range(3) :
        ax_ampl.vlines(i+1, ampl_tensor_DA[:, i].min().item(), ampl_tensor_DA[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
        ax_ampl.hlines(ampl_tensor_DA[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
        ax_ampl.hlines(ampl_tensor_DA[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    i = 3
    ax_ampl_Z.vlines(i+1, ampl_tensor_DA_Z[:, i].min().item(), ampl_tensor_DA_Z[:, i].max().item(), color=colors[compteur][1], linestyle='-.', lw=2)
    ax_ampl_Z.hlines(ampl_tensor_DA_Z[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)
    ax_ampl_Z.hlines(ampl_tensor_DA_Z[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[compteur][1], linestyle='-', lw=2)

    handles_leg.append(mpatches.Patch(color=colors[compteur][0], ec = colors[compteur][1], linewidth = 3, alpha = (6-compteur)/6, label=f'{pattern[0]}/{pattern[1]}'))
    compteur += 1

ax_corr_Z.text(s = "a)", x = get_rightcorner(ax_corr_Z, 0.9), y = get_topcorner(ax_corr_Z, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
ax_shift_Z.text(s = "b)", x = get_rightcorner(ax_shift_Z, 0.9), y = get_topcorner(ax_shift_Z, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
ax_ampl_Z.text(s = "c)", x = get_rightcorner(ax_ampl_Z, 0.9), y = get_topcorner(ax_ampl_Z, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)
ax_param.text(s = "d)", x = get_rightcorner(ax_param, 0.96), y = get_topcorner(ax_param, 0.9), bbox = dict(boxstyle="round", fc="#eee8d7", alpha = 1), fontsize = 14)


# ax_corr_Z.set_axis_off()
# ax_shift_Z.set_axis_off()
ax_corr.set_ylabel("N, P, D metric")
ax_corr_Z.set_ylabel("Z metric")
ax_shift.set_ylabel("N, P, D metric")
ax_shift_Z.set_ylabel("Z metric")
ax_ampl.set_ylabel("N, P, D metric")
ax_ampl_Z.set_ylabel("Z metric")

ax_shift.legend(handles=handles_leg, loc = 'upper left', fontsize = 14)
for ax in [ax_corr, ax_shift, ax_ampl] :
    ax.grid()
    ax.set_xticks([1, 2, 3, 4], ["N", "P", "D", "Z"])

ax_param.grid()
# ax_param.set_xticks(torch.arange(1, 11, 1), ["$K_N$", "$R_m$", "$g$", r"$\lambda$", r"$\epsilon$", r"$\alpha$", r"$\beta$", "$r$", r"$\phi$", '$S_w$'])
ax_param.set_xticks(torch.arange(1, 11, 1), ["$\\chi$", "$\\rho$", "$\\gamma$", "$\\lambda$", "$\\epsilon$", "$\\alpha$", "$\\beta$", "$\\eta$", "$\\varphi$", '$\\zeta$'])

plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14) #fontsize of the x and y labels
plt.rc('xtick', labelsize=14) #fontsize of the x tick labels
plt.rc('ytick', labelsize=14) #fontsize of the y tick labels

plt.subplots_adjust(bottom=0.15, wspace=0.15)
plt.tight_layout(w_pad = 1.0)
plt.savefig(f"Violinplit_DA_case{case}_177710", dpi = 500)


###################################################
## Violin plot for comparing DA single vs ensemble
###################################################

case = 3
pattern = [1, 1, 0, 0]
w = 1e-2

name_file = f"Res/DAens_{pattern[0]}d_{pattern[1]}d_{pattern[2]}d_{pattern[3]}d_case{case}"
nb_version = select_version("DAens", case = case, sampling_patt = pattern)
name_file += f"/version_{nb_version}"
    

costs, costs_ref = torch.load(name_file+f"/Tensor_cost.pt", weights_only=False, map_location = 'cpu')

theta_single = torch.load(name_file+f"/Tensor_params.pt", weights_only=False, map_location = 'cpu')[:, :, 0, torch.argmin(costs[:, :1] + w*costs[:, 1:2])]
theta_ens = torch.load(name_file+f"/Tensor_params.pt", weights_only=False, map_location = 'cpu').mean(dim = 2)[:, :, torch.argmin(costs[:, :1] + w*costs[:, 1:2])]
theta_target = torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", weights_only=False, map_location='cpu')

global_f_1, global_m_1 = torch.load(f"Generated_Datasets/DA/Case_{case}/False_forcings.pt", weights_only=False, map_location = 'cpu').moveaxis(2, 0)

dt = 1/2
w_bg = 1e-2
device = 'cpu'
torch.set_default_device(device)

mean_y = torch.load(f"Generated_Datasets/NN/Case_{case}/mean_y.pt", map_location = device, weights_only=False)
std_y = torch.load(f"Generated_Datasets/NN/Case_{case}/std_y.pt", map_location = device, weights_only=False)


corr_tensor_ens, corr_tensor_solo = torch.zeros([2, global_f_1.shape[0], 4])
shift_tensor_ens, shift_tensor_solo = torch.zeros([2, global_f_1.shape[0], 4])
ampl_tensor_ens, ampl_tensor_solo = torch.zeros([2, global_f_1.shape[0], 4])

(x_ref, N_ref, P_ref, Z_ref, D_ref) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_m = global_m_1, theta_values = theta_target)
(x_ens, N_ens, P_ens, Z_ens, D_ens) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_m = global_m_1, theta_values = theta_ens)
(x_solo, N_solo, P_solo, Z_solo, D_solo) = function_NPZD(t_range = torch.arange(0*365, 5*365, dt), global_f = global_f_1, global_m = global_m_1, theta_values = theta_single)

ti = time.time()
for i_val in range(global_f_1.shape[0]) :
    tepoch = time.time()
    (corr_tensor_ens[i_val, 0], shift_tensor_ens[i_val, 0], ampl_tensor_ens[i_val, 0]) = validation_params(N_ref[i_val, int(365*4/dt):], N_ens[i_val, int(365*4/dt):])
    (corr_tensor_ens[i_val, 1], shift_tensor_ens[i_val, 1], ampl_tensor_ens[i_val, 1]) = validation_params(P_ref[i_val, int(365*4/dt):], P_ens[i_val, int(365*4/dt):])
    (corr_tensor_ens[i_val, 2], shift_tensor_ens[i_val, 2], ampl_tensor_ens[i_val, 2]) = validation_params(Z_ref[i_val, int(365*4/dt):], Z_ens[i_val, int(365*4/dt):])
    (corr_tensor_ens[i_val, 3], shift_tensor_ens[i_val, 3], ampl_tensor_ens[i_val, 3]) = validation_params(D_ref[i_val, int(365*4/dt):], D_ens[i_val, int(365*4/dt):])
    
    (corr_tensor_solo[i_val, 0], shift_tensor_solo[i_val, 0], ampl_tensor_solo[i_val, 0]) = validation_params(N_ref[i_val, int(365*4/dt):], N_solo[i_val, int(365*4/dt):])
    (corr_tensor_solo[i_val, 1], shift_tensor_solo[i_val, 1], ampl_tensor_solo[i_val, 1]) = validation_params(P_ref[i_val, int(365*4/dt):], P_solo[i_val, int(365*4/dt):])
    (corr_tensor_solo[i_val, 2], shift_tensor_solo[i_val, 2], ampl_tensor_solo[i_val, 2]) = validation_params(Z_ref[i_val, int(365*4/dt):], Z_solo[i_val, int(365*4/dt):])
    (corr_tensor_solo[i_val, 3], shift_tensor_solo[i_val, 3], ampl_tensor_solo[i_val, 3]) = validation_params(D_ref[i_val, int(365*4/dt):], D_solo[i_val, int(365*4/dt):])
    
    sys.stdout.write("\rValidation n°"+str(1+i_val)+"/"+str(global_f_1.shape[0])+" within %.2f" % (time.time()-tepoch)+"s, still %.2f" %((time.time()-tepoch)*(global_f_1.shape[0]-1-i_val))+"s.")

print("\n Validation ended in %.2f" %(time.time()-ti) + "s !\n")


corr_tensor_ens, shift_tensor_ens, ampl_tensor_ens = clear_before_violinplot(corr_tensor_ens, shift_tensor_ens, ampl_tensor_ens)
corr_tensor_solo, shift_tensor_solo, ampl_tensor_solo = clear_before_violinplot(corr_tensor_solo, shift_tensor_solo, ampl_tensor_solo)

ref_value = torch.tensor([1., 2., 0.1, 0.05, 0.1, 0.3, 0.6, 0.15, 0.4, 0.1])
dt = 1/2
poids = 1e-2



corr_tensor_ens_Z = torch.cat((torch.ones([100, 3])*torch.nan, corr_tensor_ens[:, 2:3]), dim = 1)
corr_tensor_ens = torch.cat((corr_tensor_ens[:, 0:1], corr_tensor_ens[:, 1:2], corr_tensor_ens[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)
corr_tensor_solo_Z = torch.cat((torch.ones([100, 3])*torch.nan, corr_tensor_solo[:, 2:3]), dim = 1)
corr_tensor_solo = torch.cat((corr_tensor_solo[:, 0:1], corr_tensor_solo[:, 1:2], corr_tensor_solo[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)

shift_tensor_ens_Z = torch.cat((torch.ones([100, 3])*torch.nan, shift_tensor_ens[:, 2:3]), dim = 1)
shift_tensor_ens = torch.cat((shift_tensor_ens[:, 0:1], shift_tensor_ens[:, 1:2], shift_tensor_ens[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)
shift_tensor_solo_Z = torch.cat((torch.ones([100, 3])*torch.nan, shift_tensor_solo[:, 2:3]), dim = 1)
shift_tensor_solo = torch.cat((shift_tensor_solo[:, 0:1], shift_tensor_solo[:, 1:2], shift_tensor_solo[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)

ampl_tensor_ens_Z = torch.cat((torch.ones([100, 3])*torch.nan, ampl_tensor_ens[:, 2:3]), dim = 1)
ampl_tensor_ens = torch.cat((ampl_tensor_ens[:, 0:1], ampl_tensor_ens[:, 1:2], ampl_tensor_ens[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)
ampl_tensor_solo_Z = torch.cat((torch.ones([100, 3])*torch.nan, ampl_tensor_solo[:, 2:3]), dim = 1)
ampl_tensor_solo = torch.cat((ampl_tensor_solo[:, 0:1], ampl_tensor_solo[:, 1:2], ampl_tensor_solo[:, 3:4], torch.ones([100, 1])*torch.nan), dim = 1)



fig = plt.figure(figsize=(6.4*2.1, 6.4*0.7*2))
fig.suptitle(f"Scenario: Case {case} - {pattern[0]}/{pattern[-1]}")
grid = fig.add_gridspec(ncols=3, nrows = 2)
ax_corr = fig.add_subplot(grid[0, 0])
ax_shift = fig.add_subplot(grid[0, 1])
ax_ampl = fig.add_subplot(grid[0, 2])
ax_param = fig.add_subplot(grid[1, :])


ax_corr.set_title('Correlation')
ax_corr_Z = ax_corr.twinx()

ax_shift.set_title('Shift (day)')
ax_shift_Z = ax_shift.twinx()

ax_ampl.set_title("Amplitude ratio")
ax_ampl_Z = ax_ampl.twinx()

ax_param.set_title("Biogeochemical parameter error")
ax_param.set_ylim(-.5, .5)

r_solo_Z = ax_ampl_Z.violinplot(ampl_tensor_solo_Z)
r_solo = ax_ampl.violinplot(ampl_tensor_solo)
r_ens_Z = ax_ampl_Z.violinplot(ampl_tensor_ens_Z)
r_ens = ax_ampl.violinplot(ampl_tensor_ens)

c_solo_Z = ax_corr_Z.violinplot(corr_tensor_solo_Z)
c_solo = ax_corr.violinplot(corr_tensor_solo)
c_ens_Z = ax_corr_Z.violinplot(corr_tensor_ens_Z)
c_ens = ax_corr.violinplot(corr_tensor_ens)

s_solo_Z = ax_shift_Z.violinplot(abs(shift_tensor_solo_Z))
s_solo = ax_shift.violinplot(abs(shift_tensor_solo))
s_ens_Z = ax_shift_Z.violinplot(abs(shift_tensor_ens_Z))
s_ens = ax_shift.violinplot(abs(shift_tensor_ens))

p_solo = ax_param.violinplot(((theta_single-theta_target)/mean_y))
p_ens = ax_param.violinplot(((theta_ens-theta_target)/mean_y))

ax_param.set_xticks(torch.arange(1, 11, 1), ["$\\chi$", "$\\rho$", "$\\gamma$", "$\\lambda$", "$\\epsilon$", "$\\alpha$", "$\\beta$", "$\\eta$", "$\\varphi$", '$\\zeta$'])
for v_base_solo in [r_solo, r_solo_Z, c_solo, c_solo_Z, s_solo, s_solo_Z, p_solo] :
    for sc in v_base_solo['bodies']:
        sc.set_facecolor('#6ea0f5')
        sc.set_edgecolor('#155ad1')
        sc.set_linewidth(2)
for v_base_ens in [r_ens, r_ens_Z, c_ens, c_ens_Z, s_ens, s_ens_Z, p_ens] :
    for ec in v_base_ens['bodies']:
        ec.set_facecolor('#f5c84c')
        ec.set_edgecolor('#db8607')
        ec.set_linewidth(2)
        ec.set_alpha(0.5)


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

ens_legend = mpatches.Patch(color='#f5c84c', ec = '#db8607', linewidth=2, alpha=0.5, label='4DVar + Ensemble')
solo_legend = mpatches.Patch(color='#6ea0f5', ec = '#155ad1', linewidth=2, alpha=0.5, label='4DVar')
ax_ampl.legend(handles=[solo_legend, ens_legend], loc = 'upper left')
for ax in [ax_corr, ax_shift, ax_ampl] :
    ax.grid()
    set_axis_style(ax, ["N", "P", "D", "Z"])
ax_param.grid()
plt.subplots_adjust(bottom=0.15, wspace=0.15)
fig.tight_layout()
plt.savefig(f"Plots/Violinplot_DAvsDAens_case{case}_{pattern[0]}{pattern[-1]}", dpi = 500)


#################################
## Violin plot for comparing NNs
#################################

case = 3
pattern = [1, 1, 0, 0]
list_method = ["MLP", "CNN", "UNET"]
nb_case_stud = len(list_method)
folder_name = "Res/"

for i in range(nb_case_stud) :
    file_name = f"NN_{pattern[0]}d_{pattern[1]}d_{pattern[2]}d_{pattern[3]}d_case{case}_{list_method[i]}/"
    nb_version = select_version("NN", case = case, sampling_patt = pattern, architecture = list_method[i])

    print(f"Load file {folder_name+file_name+f"version_{nb_version}/"}")
    try :
        if i == 0 :
            Theta_stud = torch.load(folder_name+file_name+f"version_{nb_version}/"+"Validation_tensor_theta_pred_DA.pt", weights_only=False, map_location='cpu')[None]
            Corr_stud = torch.load(folder_name+file_name+f"version_{nb_version}/"+"Validation_tensor_corr.pt", weights_only=False, map_location='cpu')[None]
            Shift_stud = torch.load(folder_name+file_name+f"version_{nb_version}/"+"Validation_tensor_shift.pt", weights_only=False, map_location='cpu')[None]
            Ampl_stud = torch.load(folder_name+file_name+f"version_{nb_version}/"+"Validation_tensor_ampl.pt", weights_only=False, map_location='cpu')[None]
        else :
            Theta_stud = torch.cat((Theta_stud, torch.load(folder_name+file_name+f"version_{nb_version}/"+"Validation_tensor_theta_pred_DA.pt", weights_only=False, map_location='cpu')[None]), dim = 0)
            Corr_stud = torch.cat((Corr_stud, torch.load(folder_name+file_name+f"version_{nb_version}/"+"Validation_tensor_corr.pt", weights_only=False, map_location='cpu')[None]), dim = 0)
            Shift_stud = torch.cat((Shift_stud, torch.load(folder_name+file_name+f"version_{nb_version}/"+"Validation_tensor_shift.pt", weights_only=False, map_location='cpu')[None]), dim = 0)
            Ampl_stud = torch.cat((Ampl_stud, torch.load(folder_name+file_name+f"version_{nb_version}/"+"Validation_tensor_ampl.pt", weights_only=False, map_location='cpu')[None]), dim = 0)
    except :
        print("Not possible for file : ", file_name)

Theta_ref=torch.load(f"Generated_Datasets/DA/Case_{case}/Theta.pt", weights_only=False, map_location='cpu')

# from Functions import validation_params, function_NPZD, clear_before_violinplot, get_rightcorner, get_topcorner

colors = [["#852020", "#4f0202"], ["#e83a63", "#eb0239"], ['#1d17d6', '#0a0769']] # red, purple, db, lb, green

fig = plt.figure(figsize=(6.4*2.1, 6.4*0.7*2))
grid = fig.add_gridspec(nrows = 2, ncols=3)
ax_corr = fig.add_subplot(grid[0, 0])
ax_shift = fig.add_subplot(grid[0, 1])
ax_ampl = fig.add_subplot(grid[0, 2])
ax_param = fig.add_subplot(grid[1, :])

ax_corr.set_title('Correlation')
ax_shift.set_title('Shift (day)')
ax_ampl.set_title("Amplitude ratio")
ax_param.set_title("Biogeochemical parameter error")

c_DA, s_DA, r_DA, v_DA = [], [], [], []

compteur = 0
corr, shift, ampl = [], [], []
for i in range(nb_case_stud) :
    # corr_step, shift_step, ampl_step = metrics_BGC(torch.sum(States_stud[i]*z_ref, dim = 3), torch.sum(States_ref*z_ref, dim = 3))
    # corr.append(corr_step.cpu()); shift.append(shift_step.cpu()); ampl.append(ampl_step.cpu())

    c_DA.append(ax_corr.violinplot(Corr_stud[i], showextrema=False))
    s_DA.append(ax_shift.violinplot(abs(Shift_stud[i]), showextrema=False))
    r_DA.append(ax_ampl.violinplot(Ampl_stud[i], showextrema=False))
    v_DA.append(ax_param.violinplot(((Theta_stud[i]-Theta_ref)/Theta_ref), showextrema=False))

for j in range(nb_case_stud) :
    for i in range(10) :
        ax_param.vlines(i+1, ((Theta_stud[j]-Theta_ref)/Theta_ref)[:, i].min().item(), ((Theta_stud[j]-Theta_ref)/Theta_ref)[:, i].max().item(), color=colors[j][1], linestyle='-', alpha = 0.5)
        ax_param.hlines(((Theta_stud[j]-Theta_ref)/Theta_ref)[:, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[j][1], linestyle='-', alpha = 0.5)
        ax_param.hlines(((Theta_stud[j]-Theta_ref)/Theta_ref)[:, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[j][1], linestyle='-', alpha = 0.5)

    for i in range(4) :
        ax_corr.vlines(i+1, Corr_stud[j][ :, i].min().item(), Corr_stud[j][ :, i].max().item(), color=colors[j][1], linestyle='-', alpha = 0.5)
        ax_corr.hlines(Corr_stud[j][ :, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[j][1], linestyle='-', alpha = 0.5)
        ax_corr.hlines(Corr_stud[j][ :, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[j][1], linestyle='-', alpha = 0.5)
    
        ax_shift.vlines(i+1, abs(Shift_stud[j])[ :, i].min().item(), abs(Shift_stud[j])[ :, i].max().item(), color=colors[j][1], linestyle='-', alpha = 0.5)
        ax_shift.hlines(abs(Shift_stud[j])[ :, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[j][1], alpha = 0.5)
        ax_shift.hlines(abs(Shift_stud[j])[ :, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[j][1], alpha = 0.5)
    
        ax_ampl.vlines(i+1, Ampl_stud[j][ :, i].min().item(), Ampl_stud[j][ :, i].max().item(), color=colors[j][1], linestyle='-', alpha = 0.5)
        ax_ampl.hlines(Ampl_stud[j][ :, i].min().item(), xmin = i+0.9, xmax = i+1.1, color=colors[j][1], linestyle='-', alpha = 0.5)
        ax_ampl.hlines(Ampl_stud[j][ :, i].max().item(), xmin = i+0.9, xmax = i+1.1, color=colors[j][1], linestyle='-', alpha = 0.5)
    
for compteur in range(nb_case_stud) :
    for pc in v_DA[compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
    for pc in c_DA[compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
    for pc in s_DA[compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)
    for pc in r_DA[compteur]['bodies']:
        pc.set_facecolor(colors[compteur][0])
        pc.set_edgecolor(colors[compteur][1])
        pc.set_linewidth(3)

comp_legend=[]
for i in range(nb_case_stud) :
    comp_legend.append(mpatches.Patch(color=colors[i][0], ec = colors[i][1], linewidth=3, alpha = 0.3, label=list_method[i]))

ax_param.legend(handles=comp_legend, loc = 'upper center', fontsize = 14, ncol=3)
for ax in [ax_corr, ax_shift, ax_ampl] :
    ax.grid()
    ax.set_xticks([1, 2, 3, 4], ["N", "P", "Z", "D"])

ax_param.grid()
ax_param.set_xticks(torch.arange(1, 11, 1).cpu(), ["$\\chi$", "$\\rho$", "$\\gamma$", "$\\lambda$", "$\\epsilon$", "$\\alpha$", "$\\beta$", "$\\eta$", "$\\varphi$", '$\\zeta$'])
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14) #fontsize of the x and y labels
plt.rc('xtick', labelsize=14) #fontsize of the x tick labels
plt.rc('ytick', labelsize=14) #fontsize of the y tick labels
fig.suptitle(t=f"Scenario: Case {case} - {pattern[0]}/{pattern[1]}", y = 0.99, fontsize = 16)

plt.subplots_adjust(bottom=0.15, wspace=0.15)
fig.tight_layout()
fig.savefig(f"Plots/Violinplot_NNs_case{case}_{pattern[0]}_{pattern[1]}", dpi=300)