# File: experiment1.py
# Authors: Dane Brouwer, Marion Lepert, Joshua Citron,
# Description: This file serves as the top level
# script that runs PyBullet experiment 1. This
# experiment tests control strategies on a
# set of pseudo-randomly generated scenes. The
# script controls the logging of information as
# well as control strategies that will be tested.

# Import relevant modules and functions.
import yaml 
import argparse
import pybullet as pb
import numpy as np 
import pybullet_data
from tqdm import tqdm 
from datetime import datetime
import os
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

# Parse config file.
with open(os.path.join('config','default.yaml'), 'r') as f: 
	params = yaml.safe_load(f)

# If logging, create empty log file.
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

# Initialize pybullet physics.
if args.gui:
	physics_client = pb.connect(pb.GUI)
else:
	physics_client = pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())

# Initialize control strategies to test. If you don't want to test
# a certian strategy, just comment out the strategy.
all_controllers = [StraightLineController(physics_client, params), 
		   			BurrowController(physics_client, params), 
				    ExcavateController(physics_client, params),
					HybridClockController(physics_client, params),
				   	HybridEventController(physics_client, params)
					]

# Initialize relevant variables for testing
results_dict = defaultdict(list)
test_idx = 0 
seed_sequence = [np.random.randint(0,1e6) for _ in range(params['num_trials'])]

# Run given control strategies for a given number of trials
for controller in tqdm(all_controllers,desc="Test Cases"):
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
	save_results(log_path, results_dict)

pb.disconnect()