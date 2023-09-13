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

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gui', action="store_true", help="Use gui")
parser.add_argument('--log', action="store_true", help="Log results")
args = parser.parse_args()

# Parse config file
with open(os.path.join('config','default.yaml'), 'r') as f: 
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

bur_amp_list = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90] #
bur_freq_list = [0.5/240, 0.625/240, 0.75/240, 0.875/240, 1/240, 1.125/240, 1.25/240, 1.375/240, 1.5/240, 1.625/240] #
# num_obs_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

excavate_step_thresh_list = [450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800] #
trigger_excavate_step_thresh_list = [450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800] #


for controller in tqdm(all_controllers,desc="Test Cases"):
	if controller.test_case == "Straight Line":
		param1_list = [0.5]
		param2_list = [0.5] #bur_freq_list
	elif controller.test_case == "Burrow":
		param1_list = bur_amp_list
		param2_list = bur_freq_list #bur_freq_list
	elif controller.test_case == "Excavate":
		param1_list = excavate_step_thresh_list
		param2_list = trigger_excavate_step_thresh_list
	for param1 in tqdm(param1_list, leave=False, desc="param1"):
		for param2 in tqdm(param2_list, leave=False, desc="param2"):
			controller.bur_amp = param1/(1-param1)
			controller.excavate_step_thresh = param1
			controller.bur_freq = param2
			controller.trigger_excavate_step_thresh = param2
			# controller.num_obs = param2
			for trial_idx in tqdm(range(params['num_trials']), leave=False, desc="Trials"):
				seed = seed_sequence[trial_idx]
				np.random.seed(seed) 

				pb.resetSimulation()
				controller.reset(seed, 3.75, 0.4)

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
			results_dict[key] += val



if args.log:	
	save_results(log_path, results_dict) ## Move to above if inside for loop?

pb.disconnect()