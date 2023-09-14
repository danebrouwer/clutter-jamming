# File: analyze_results1.py
# Author: Dane Brouwer
# Description: File used to generate plots and
# analyze the results of experiment 1.

# Import relevant modules.
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

###### DATA PROCESSING #####
# TODO
pkl_dir1 = "" # Fill in the directory for your log file.
pkl_fn1 = ".pkl" # Fill in with log file name.
pkl_df = pd.read_pickle(pkl_dir1 + pkl_fn1)

for pkl_df in [pkl_df]:

    pkl_df['Control Type'] = pkl_df['Control Type'].astype(str)
    pkl_df['Distance to goal'] = pkl_df['Distance to goal'].astype(float)
    pkl_df['Completion time'] = pkl_df['Completion time'].astype(float)
    pkl_df['Success time'] = pkl_df['Success time'].astype(float)
    pkl_df['Stuck counter'] = pkl_df['Stuck counter'].astype(int)
    pkl_df['Number of obstacles'] = pkl_df['Number of obstacles'].astype(int)
    pkl_df['Random seed'] = pkl_df['Random seed'].astype(int)
    # You can add in any of the values that we collect, as
    # not all are shown here!

pairs1 = [(("Hybrid Clock"), ("Hybrid Event")), (("Excavate"), ("Hybrid Clock")), (("Burrow"), ("Excavate")), (("Straight Line"), ("Burrow"))]
pairs2 = [(("Straight Line"), ("Burrow")), (("Burrow"), ("Excavate")), (("Hybrid Clock"), ("Hybrid Event")), (("Excavate"), ("Hybrid Clock"))]
pairs3 = [(("Straight Line"), ("Burrow")), (("Burrow"), ("Hybrid Event")), (("Straight Line"), ("Hybrid Event"))]
pairs4 = [(("Straight Line"), ("Burrow")), (("Burrow"), ("Excavate")), (("Excavate"), ("Hybrid Clock")), (("Hybrid Clock"), ("Hybrid Event")), (("Excavate"), ("Hybrid Event")), (("Burrow"), ("Hybrid Event")), (("Straight Line"), ("Hybrid Event"))]

SL_df = pkl_df[pkl_df["Control Type"] == "Straight Line"]
BU_df = pkl_df[pkl_df["Control Type"] == "Burrow"]
EX_df = pkl_df[pkl_df["Control Type"] == "Excavate"]
HC_df = pkl_df[pkl_df["Control Type"] == "Hybrid Clock"]
HE_df = pkl_df[pkl_df["Control Type"] == "Hybrid Event"]

##### PLOTTING #####
plt.figure()
ax = sns.boxplot(data=pkl_df,x='Control Type',y='Distance to goal')
annotator = Annotator(ax, data=pkl_df, x='Control Type',y='Distance to goal',pairs=pairs4)
annotator.configure(test="t-test_welch", text_format="star", loc="inside")
annotator.apply_and_annotate()
print(plt.rcParams)
plt.figure()
ax = sns.boxplot(data=pkl_df,x='Control Type',y='Success time')
annotator = Annotator(ax, data=pkl_df, x='Control Type',y='Success time',pairs=pairs4)
annotator.configure(test="t-test_welch", text_format="star", loc="inside")
annotator.apply_and_annotate()

plt.figure()
ax = sns.boxplot(data=pkl_df,x='Control Type',y='Completion time')
annotator = Annotator(ax, data=pkl_df, x='Control Type',y='Completion time',pairs=pairs4)
annotator.configure(test="t-test_welch", text_format="star", loc="inside")
annotator.apply_and_annotate()

n_trials = len(SL_df)

SL_failure = np.sum(np.isnan(SL_df["Success time"]))
BU_failure = np.sum(np.isnan(BU_df["Success time"]))
EX_failure = np.sum(np.isnan(EX_df["Success time"]))
HC_failure = np.sum(np.isnan(HC_df["Success time"]))
HE_failure = np.sum(np.isnan(HE_df["Success time"]))

print("Straight line success rate: ", (n_trials-SL_failure)/n_trials)
print("Burrow success rate: ", (n_trials-BU_failure)/n_trials)
print("Excavate success rate: ", (n_trials-EX_failure)/n_trials)
print("Hybrid clock success rate: ", (n_trials-HC_failure)/n_trials)
print("Hybrid event success rate: ", (n_trials-HE_failure)/n_trials)


plt.show()