import pdb 

# import pybullet_planning as pp
import yaml 
import argparse
import pybullet as pb
import numpy as np 
import pybullet_data
from tqdm import tqdm 
import time
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import OPTICS
from collections import defaultdict

from clutter.controllers.straight_line_controller import StraightLineController
from clutter.controllers.burrow_controller import BurrowController
from clutter.controllers.excavate_controller import ExcavateController
from clutter.controllers.hybrid_clock_controller import HybridClockController
from clutter.controllers.hybrid_event_controller import HybridEventController
from clutter.utils import save_results

from urdfpy import URDF
import urdfpy

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gui', action="store_true", help="Use gui")
parser.add_argument('--log', action="store_true", help="Log results")
args = parser.parse_args()

# Parse config file
with open(os.path.join('config','default_copy.yaml'), 'r') as f: 
	params = yaml.safe_load(f)

max_force = 10
max_torque = 15
seed_sequence = []
results_list = None
map_contacts = False

if args.log:
	cwd = os.getcwd()
	now = datetime.now()
	date_string = now.strftime("%Y_%m_%d")
	time_string = now.strftime("%H%M%S")
	log_dir_name = os.path.join(cwd, "Logging", date_string)
	path_exists = os.path.exists(log_dir_name)
	if not path_exists:
		os.mkdir(log_dir_name)
	log_filename =  time_string + "_" + str(params['num_trials']) + "Trials" + ".pkl"
	log_path = os.path.join(log_dir_name, log_filename)

test_idx = 0 

seed_sequence = [np.random.randint(0,1e6) for _ in range(params['num_trials'])]

# Initialize pybullet physics
if args.gui:
	physics_client = pb.connect(pb.GUI)
else:
	physics_client = pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())

all_controllers = [StraightLineController(physics_client, params), 
		   			BurrowController(physics_client, params), 
				    ExcavateController(physics_client, params),
					# HybridClockController(physics_client, params),
				   	# HybridEventController(physics_client, params)
					]

results_dict = defaultdict(list)

# bur_amp_list = [1, 3, 5, 7, 9] #
# bur_freq_list = [0.5/240, 0.75/240, 1/240, 1.25/240, 1.5/240] #, 1/240, 1.25/240, 1.5/240
num_obs_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30] # , 22, 23, 24, 25, 26, 27, 28, 29, 30
depth_list = [2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0] # 2.75, 3.25, , 4.25, 4.75
obj_size_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65] # , 0.3, 0.4, 0.5, 0.6

y_paddle_filepath = "./resources/ypaddle2D_copy.urdf"
# 3.75 50 0.2
# excavate_step_thresh_list = [600, 900, 1200, 1500, 1800] #, 1200, 1500, 1800
# trigger_excavate_step_thresh_list = [600, 900, 1200, 1500, 1800] #, 1200, 1500, 1800
list_of_param_lists = [depth_list, num_obs_list, obj_size_list]

for controller in tqdm(all_controllers,desc="Test Cases"):
    # if controller.test_case == "Straight Line":
    #     list_of_param_lists = [[1]]
    # else:
    #     list_of_param_lists = [depth_list, num_obs_list, obj_size_list]
    for param_list in tqdm(list_of_param_lists):
        for param in tqdm(param_list, leave=False, desc="param1"):
            if param_list == num_obs_list:
                  controller.num_obs = param
            else:
                  controller.num_obs = 25
            if param_list == depth_list:
                # Credit to Marion for this code
                y_paddle_filepath_new = "./resources/ypaddle2D_copy_{depth}.urdf".format(depth=param)
                with open(y_paddle_filepath, 'r') as file:
                    data = file.read()
                    data = data.replace("$WIDTH$", str(param))
                with open(y_paddle_filepath_new, 'w') as file:
                     file.write(data)
            for trial_idx in tqdm(range(params['num_trials']), leave=False, desc="Trials"):
                seed = seed_sequence[trial_idx]
                np.random.seed(seed) 

                pb.resetSimulation()
                if param_list == num_obs_list:
                    controller.reset(seed, 3.75, 0.4)
                elif param_list == depth_list:
                    controller.reset(seed, param, 0.4)
                elif param_list == obj_size_list:
                    controller.reset(seed, 3.75, param)
                # else:
                #     controller.reset(seed, 3.75, 0.4)

                prev_step = 0
                pbar = tqdm(total=params['total_step_thresh'] + 1, leave=False, desc="Trial Timeout")
                done = False
                while not done:
                    done, curr_step = controller.step()
                    pbar.update(curr_step - prev_step)
                    prev_step = curr_step

                pbar.close()
    if args.log:
        for key, val in controller.results_dict.items():
            results_dict[key] += val #Was tabbed, fixed double logging?


if args.log:	
	save_results(log_path, results_dict) ## Move to above if inside for loop?

pb.disconnect()