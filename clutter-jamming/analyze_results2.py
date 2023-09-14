# File: analyze_results2.py
# Authors: Dane Brouwer, Joshua Citron
# Description: File used to generate
# plots from experiment 2.

# Import relevant modules.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

###### DATA PROCESSING #####
# TODO
pkl_dir = "" # Fill in with the directory your log file is in
pkl_fn = "" # Fill in with the name of the pkl file
pkl_df = pd.read_pickle(pkl_dir + pkl_fn)

for pkl_df in [pkl_df]:

    pkl_df['Control Type'] = pkl_df['Control Type'].astype(str)
    pkl_df['Distance to goal'] = pkl_df['Distance to goal'].astype(float)
    pkl_df['Completion time'] = pkl_df['Completion time'].astype(float)
    pkl_df['Success time'] = pkl_df['Success time'].astype(float)
    pkl_df['Stuck counter'] = pkl_df['Stuck counter'].astype(int)
    pkl_df['Number of obstacles'] = pkl_df['Number of obstacles'].astype(int)
    pkl_df['Random seed'] = pkl_df['Random seed'].astype(int)
    pkl_df['Burrow amplitude'] = pkl_df['Burrow amplitude'].astype(float)
    pkl_df['Burrow frequency'] = pkl_df['Burrow frequency'].astype(float)
    pkl_df['Excavate step thresh'] = pkl_df['Excavate step thresh'].astype(float)
    pkl_df['Trigger excavate step thresh'] = pkl_df['Trigger excavate step thresh'].astype(float)

# TODO
n_trials = 50 # Change to number of trials
num_params = 10 # Change to number of parameters

dist_to_goal_burrow = []
dist_to_goal_excavate = []
comp_time_burrow = []
comp_time_excavate = []

# TODO
bur_amp_list = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90] # Fill in with burrow amplitudes
bur_freq_list = [0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625] # Fill in with burrow frequencies
excavate_step_thresh_list = [450/240, 600/240, 750/240, 900/240, 1050/240, 1200/240, 1350/240, 1500/240, 1650/240, 1800/240] # Fill in with excavate step
trigger_excavate_step_thresh_list = [450/240, 600/240, 750/240, 900/240, 1050/240, 1200/240, 1350/240, 1500/240, 1650/240, 1800/240] # Fill in with excavate trigger

sl_df = pkl_df[(pkl_df["Control Type"] == "Straight Line")]

sl_failure = np.sum(np.isnan(sl_df["Success time"]))
sl_succ_rate = (n_trials-sl_failure)/n_trials

dist_to_goal_straight = np.mean(sl_df["Distance to goal"])
comp_time_straight = np.mean(sl_df["Completion time"])

bur_worst_case = 1.0
excv_worst_case = 1.0

for i in range(len(bur_amp_list)):
    for j in range(len(bur_freq_list)):
        bur_df = pkl_df[(pkl_df["Control Type"] == "Burrow")
                                                                  & (pkl_df["Burrow amplitude"] == bur_amp_list[i]/(1-bur_amp_list[i])) 
                                                                  & (pkl_df["Burrow frequency"] == bur_freq_list[j]/240)]
        dist_to_goal_burrow.append([dist_to_goal_straight/np.mean(bur_df["Distance to goal"])])
        comp_time_burrow.append([comp_time_straight/np.mean(bur_df["Completion time"])])

        excv_df = pkl_df[(pkl_df["Control Type"] == "Excavate")
                                                                  & (pkl_df["Excavate step thresh"] == excavate_step_thresh_list[i]*240) 
                                                                  & (pkl_df["Trigger excavate step thresh"] == trigger_excavate_step_thresh_list[j]*240)]
        dist_to_goal_excavate.append([dist_to_goal_straight/np.mean(excv_df["Distance to goal"])])
        comp_time_excavate.append([comp_time_straight/np.mean(excv_df["Completion time"])])
        
        
        bur_failure = np.sum(np.isnan(bur_df["Success time"]))
        bur_succ_rate = (n_trials-bur_failure)/n_trials

        excv_failure = np.sum(np.isnan(excv_df["Success time"]))
        excv_succ_rate = (n_trials-excv_failure)/n_trials
        
        bur_worst_case = min([bur_worst_case, bur_succ_rate])
        excv_worst_case = min([excv_worst_case, excv_succ_rate])
        
        
XB, YB = np.meshgrid(bur_amp_list, bur_freq_list, indexing='ij')
XE, YE = np.meshgrid(excavate_step_thresh_list, trigger_excavate_step_thresh_list, indexing='ij')
reshaped_dtgb = np.array([dist_to_goal_burrow]).reshape(XB.shape)
reshaped_dtge = np.array([dist_to_goal_excavate]).reshape(XE.shape)
reshaped_dtgb_filt = scipy.ndimage.gaussian_filter(reshaped_dtgb, sigma=1.0, order=0)
reshaped_dtge_filt = scipy.ndimage.gaussian_filter(reshaped_dtge, sigma=1.0, order=0)

reshaped_ctb = np.array([comp_time_burrow]).reshape(XB.shape)
reshaped_cte = np.array([comp_time_excavate]).reshape(XE.shape)
reshaped_ctb_filt = scipy.ndimage.gaussian_filter(reshaped_ctb, sigma=1.0, order=0)
reshaped_cte_filt = scipy.ndimage.gaussian_filter(reshaped_cte, sigma=1.0, order=0)

##### PLOTTING #####
# Feel free to uncomment any graph of interest.

# fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
# ax1.plot_surface(XB,YB, reshaped_dtgb)
# ax1.set_title("Burrow Distance to Goal")
# ax1.set_xlabel("Burrow Amplitude")
# ax1.set_ylabel("Burrow Frequency")

# fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
# ax2.plot_surface(XB,YB, reshaped_ctb)
# ax2.set_title("Burrow Completion Time")
# ax2.set_xlabel("Burrow Amplitude")
# ax2.set_ylabel("Burrow Frequency")

# fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
# ax3.plot_surface(XE,YE, reshaped_dtge)
# ax3.set_title("Excavate Distance to Goal")
# ax3.set_xlabel("Excavate Time")
# ax3.set_ylabel("Trigger Excavate Time")

# fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
# ax4.plot_surface(XE,YE, reshaped_cte)
# ax4.set_title("Excavate Completion Time")
# ax4.set_xlabel("Excavate Time")
# ax4.set_ylabel("Trigger Excavate Time")

plt.rcParams.update({'font.size': 32})

fig5, ax5 = plt.subplots(figsize=(13.5,9.5))
levels = np.arange(3.4, 6.0, 0.3)
plot = ax5.contour(XB, YB, reshaped_dtgb_filt, levels)
ax5.clabel(plot, inline=True, fontsize=32)
ax5.set_ylabel('Burrow frequency (Hz)')
ax5.set_xlabel('Burrow amplitude')
# ax5.set_title('Burrow distance to goal')

fig6, ax6 = plt.subplots(figsize=(13.5,9.5))
levels = np.arange(1.2, 1.70, 0.05)
plot = ax6.contour(XB, YB, reshaped_ctb_filt, levels)
ax6.clabel(plot, inline=True, fontsize=32)
ax6.set_ylabel('Burrow frequency (Hz)')
ax6.set_xlabel('Burrow amplitude')
# ax6.set_title('Burrow completion time')

fig7, ax7 = plt.subplots(figsize=(13.5,9.5))
levels = np.arange(2.2, 6.0, 0.3)
plot = ax7.contour(XE, YE, reshaped_dtge_filt, levels)
ax7.clabel(plot, inline=True, fontsize=32)
ax7.set_ylabel('Trigger excavate time (s)')
ax7.set_xlabel('Excavate duration (s)')
# ax7.set_title('Excavate distance to goal')

fig8, ax8 = plt.subplots(figsize=(13.5,9.5))
levels = np.arange(1.2, 1.60, 0.05)
plot = ax8.contour(XE, YE, reshaped_cte_filt, levels)
ax8.clabel(plot, inline=True, fontsize=32)
ax8.set_ylabel('Trigger excavate time (s)')
ax8.set_xlabel('Excavate duration (s)')
# ax8.set_title('Excavate completion time')

plt.show()