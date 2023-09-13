import pdb 

# import pybullet_planning as pp

import pybullet as pb
import numpy as np 
import pybullet_data
from tqdm import tqdm 
import time
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt

from clutter.object2D import *
from clutter.cluttered_scene import * 

# Function to plot contact force vectors
def plot_contact_forces(obj, lifetime):
	"""
	Display contact force vectors on body. 
	"""
	
	list_cps = obj.get_contact_points(external_id=None)

	if list_cps is not None: 
		for cp in list_cps:
			Fn_mag = cp.normal_force
			Fn_dir = np.array(cp.contact_normal_on_B)
			Fn_vec = Fn_mag * Fn_dir 
			contact_pos = np.array(cp.pos_on_A)
			pb.addUserDebugLine(contact_pos, contact_pos + 0.1*Fn_vec, lineColorRGB=[1,0,0], lineWidth=3.0, lifeTime=lifetime)
			Fs1_mag = cp.lateral_friction_mag_1
			Fs1_dir = np.array(cp.lateral_friction_dir_1)
			Fs1_vec = Fs1_mag * Fs1_dir 
			pb.addUserDebugLine(contact_pos, contact_pos + 0.1*Fs1_vec, lineColorRGB=[0,0,1], lineWidth=3.0, lifeTime=lifetime)

def store_contact_locations(obj, contact_map, obstacle_map):
	"""
	Display contact force vectors on body. 
	"""

	list_cps = obj.get_contact_points(external_id=None)

	if list_cps is not None: 
		for cp in list_cps:
			Fn_mag = cp.normal_force
			contact_pos = np.array(cp.pos_on_A)
			if Fn_mag >= 1.0:
				if Fn_mag >= 10.0:
					if obstacle_map is not None:
						obstacle_map = np.vstack((obstacle_map,[contact_pos[0],contact_pos[1]]))
					else:
						obstacle_map = np.array([contact_pos[0],contact_pos[1]])
				else:
					if contact_map is not None:
						contact_map = np.vstack((contact_map,[contact_pos[0],contact_pos[1]]))
					else:
						contact_map = np.array([contact_pos[0],contact_pos[1]])

	# if obstacle_map is not None:
	# 	for i in range(len(obstacle_map[:,0])):
	# 		contact_location = np.array((obstacle_map[:,0],obstacle_map[:,1],0.5))
	# 		if is_inside(puck_obj,contact_location):
	# 			np.delete(obstacle_map,i)

	
	return contact_map, obstacle_map

def plot_contact_map(contact_map, obstacle_map):
	x_contact = contact_map[:,0]
	y_contact = contact_map[:,1]
	x_obstacle = obstacle_map[:,0]
	y_obstacle = obstacle_map[:,1] 
	plt.scatter(-y_contact,x_contact,color=[0,0,1])
	plt.scatter(-y_obstacle,x_obstacle,color=[1,0,0])
	plt.ylim(-1,5)
	plt.xlim(-2,2)
	ax = plt.gca()
	ax.set_aspect('equal')
	plt.show()


		
def get_impeding_contact_loc(obj):
	"""
	Return the location of the contact with highest force
	"""
	
	contact_pos_max = np.array([0.0,0.0])
	F_net_max = 0.0

	list_cps = obj.get_contact_points(external_id=None)

	if list_cps is not None: 
		for cp in list_cps:
			Fn_mag = cp.normal_force
			contact_pos = np.array(cp.pos_on_A)
			Fs1_mag = cp.lateral_friction_mag_1
			F_net = np.linalg.norm([Fn_mag, Fs1_mag])
			if F_net > F_net_max:
				F_net_max = F_net
				contact_pos_max = contact_pos

	return contact_pos_max

def contact_force_exceeds(obj,force_thresh):
	"""
	Return true if any contact exceeds force threshold
	"""

	list_cps = obj.get_contact_points(external_id=None)

	if list_cps is not None: 
		for cp in list_cps:
			Fn_mag = cp.normal_force
			# contact_pos = np.array(cp.pos_on_A)
			Fs1_mag = cp.lateral_friction_mag_1
			F_net = np.linalg.norm([Fn_mag, Fs1_mag])
			if F_net > force_thresh:
				return True
	else:
		return False
                
def run_stop(vel_mag = 0.0, ang_vel_mag = 0.0, max_force = 10, max_torque = 15):

	target_vel = vel_mag * target_vel_direction
	target_ang_vel = ang_vel_mag * target_ang_vel_direction

	puck_obj.apply_velocity(target_vel, max_force=max_force)
	puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)


def run_burrow_control(vel_mag = 0.5, ang_vel_mag = 0.35, max_force = 10, max_torque = 15, lin_sin_amp = 1.0, orn_sin_amp = 0.25, sin_period = 1.5):

	sin_term = np.sin(2*np.pi/sin_period*curr_time)
	
	target_vel_sin = target_vel_direction + perp_vel_direction*lin_sin_amp*sin_term
	target_vel_direction_sin = (target_vel_sin) / np.linalg.norm((target_vel_sin))

	target_vel = vel_mag * target_vel_direction_sin

	if pointed_at_goal(): 
		target_ang_vel = 0 * target_ang_vel_direction
	else:
		target_ang_vel = ang_vel_mag * target_ang_vel_direction

	puck_obj.apply_velocity(target_vel, max_force=max_force)
	puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)

def interrupt_CW_excavate(t_ex, vel_mag = 0.5, ang_vel_mag = 0.5, max_force = 10, max_torque = 15): #goes CCW since it interrupts CW
	global excavate_start_time_tactile
	# global t_ex

	# print("Starting CW interrupt")
	if t_ex - excavate_start_time_tactile < excavate_time_thresh/10: #back
		target_vel = vel_mag * np.array([-0.25, 0])
		target_ang_vel = 0
	elif t_ex - excavate_start_time_tactile < 2*excavate_time_thresh/8: #right/rotate fwd
		target_vel = vel_mag * np.array([0, 1])
		target_ang_vel = -ang_vel_mag*puck_orn
	elif t_ex - excavate_start_time_tactile < 4*excavate_time_thresh/8: #fwd
		target_vel = vel_mag * np.array([1, 0])
		target_ang_vel = 0
	elif t_ex - excavate_start_time_tactile < 5*excavate_time_thresh/8: #left/rotate CCW/fwd
		target_vel = vel_mag * np.array([0.1, -1])
		target_ang_vel = -0.5*ang_vel_mag
	elif t_ex - excavate_start_time_tactile  < 6*excavate_time_thresh/8: #back/left/rotate CCW
		target_vel = vel_mag * np.array([-1, -0.25])
		target_ang_vel = -0.5*ang_vel_mag
	else: # t_ex - excavate_start_time_tactile  <= 11/10*excavate_time_thresh:  #right/rotate CW
		target_vel = vel_mag * np.array([0, 0.25])
		target_ang_vel = 0.3*ang_vel_mag
	# else:
	# 	print("t_ex outside bounds of time cutoff")
	# 	print("t_ex end of interrupt: ", t_ex)
	# 	print("excavate_start_time end of interrupt: ", excavate_start_time)

	puck_obj.apply_velocity(target_vel, max_force=max_force)
	puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)

def interrupt_CCW_excavate(t_ex, vel_mag = 0.5, ang_vel_mag = 0.5, max_force = 10, max_torque = 15): # goes CW since interrupts CCW
	global excavate_start_time_tactile
	# global t_ex

	# print("Starting CCW interrupt")
	if t_ex - excavate_start_time_tactile < excavate_time_thresh/10: #back
		target_vel = vel_mag * np.array([-0.25, 0])
		target_ang_vel = 0
	elif t_ex - excavate_start_time_tactile < 2*excavate_time_thresh/8: #left/rotate fwd, 
		target_vel = vel_mag * np.array([0, -1])
		target_ang_vel = -ang_vel_mag*puck_orn
	elif t_ex - excavate_start_time_tactile < 4*excavate_time_thresh/8: #fwd
		target_vel = vel_mag * np.array([1, 0])
		target_ang_vel = 0
	elif t_ex - excavate_start_time_tactile < 5*excavate_time_thresh/8: #right/rotate CW/fwd
		target_vel = vel_mag * np.array([0.1, 1])
		target_ang_vel = 0.5*ang_vel_mag
	elif t_ex - excavate_start_time_tactile  < 6*excavate_time_thresh/8: #back/right/rotate CW
		target_vel = vel_mag * np.array([-1, 0.25])
		target_ang_vel = 0.5*ang_vel_mag
	else: # t_ex - excavate_start_time_tactile  <= 11/10*excavate_time_thresh:  #left/rotate CCW
		target_vel = vel_mag * np.array([0, -0.25])
		target_ang_vel = -0.3*ang_vel_mag
	# else:
	# 	print("t_ex outside bounds of time cutoff")
	# 	print("t_ex end of interrupt: ", t_ex)
	# 	print("excavate_start_time end of interrupt: ", excavate_start_time_tactile)

	puck_obj.apply_velocity(target_vel, max_force=max_force)
	puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)

def perform_CCW_excavate(t_ex, vel_mag = 0.5, ang_vel_mag = 0.5, max_force = 10, max_torque = 15):
	global excavate_start_time
	# global target_vel
	# global t_ex

	# print("Starting CCW excavate")
	if t_ex - excavate_start_time < excavate_time_thresh/10: #back
		target_vel = vel_mag * np.array([-0.25, 0])
		target_ang_vel = 0
	elif t_ex - excavate_start_time < 2*excavate_time_thresh/8: #right/rotate fwd
		contact_pos_max = get_impeding_contact_loc(puck_obj)
		if tactile and contact_pos_max[1] > tip_pos[1] and is_stuck_tactile(): # higher y value means left
			print("Starting CCW interrupt")
			global excavate_start_time_tactile 
			excavate_start_time_tactile = time.time() - t_init
			t_ex_tactile = time.time() - t_init
			while t_ex_tactile - excavate_start_time_tactile < excavate_time_thresh:
				pb.stepSimulation()
				interrupt_CCW_excavate(t_ex=t_ex_tactile)
				t_ex_tactile = time.time() - t_init
			excavate_start_time = time.time() - t_init # or set excavate_start_time to t_ex - excavate_time_thresh to exit and resume straight/burrow?
			t_ex = time.time() - t_init
			# print("t_ex inside perform: ", t_ex)
			# print("excavate_start_time inside perform: ", excavate_start_time)
			target_vel = vel_mag * np.array([-0.25, 0])
			target_ang_vel = 0
		else:
			target_vel = vel_mag * np.array([0, 1])
			target_ang_vel = -ang_vel_mag*puck_orn
	elif t_ex - excavate_start_time < 4*excavate_time_thresh/8: #fwd
		target_vel = vel_mag * np.array([1, 0])
		target_ang_vel = 0
	elif t_ex - excavate_start_time < 5*excavate_time_thresh/8: #left/rotate CCW/fwd
		target_vel = vel_mag * np.array([0.1, -1])
		target_ang_vel = -0.5*ang_vel_mag
	elif t_ex - excavate_start_time  < 6*excavate_time_thresh/8: #back/left/rotate CCW
		target_vel = vel_mag * np.array([-1, -0.25])
		target_ang_vel = -0.5*ang_vel_mag
	else: # t_ex - excavate_start_time  <= excavate_time_thresh:  #right/rotate CW
		target_vel = vel_mag * np.array([0, 0.25])
		target_ang_vel = 0.3*ang_vel_mag
	# else:
	# 	print("t_ex outside bounds of time cutoff")
	# 	print("t_ex end of perform: ", t_ex)
	# 	print("excavate_start_time end of perform: ", excavate_start_time)

	# print("Ending CCW excavate")

	puck_obj.apply_velocity(target_vel, max_force=max_force)
	puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)

def perform_CW_excavate(t_ex, vel_mag = 0.5, ang_vel_mag = 0.5, max_force = 10, max_torque = 15):
	global excavate_start_time
	# global t_ex

	# print("Starting CW excavate")
	if t_ex - excavate_start_time < excavate_time_thresh/10: #back
		target_vel = vel_mag * np.array([-0.25, 0])
		target_ang_vel = 0
	elif t_ex - excavate_start_time < 2*excavate_time_thresh/8: #left/rotate fwd, 
		contact_pos_max = get_impeding_contact_loc(puck_obj)
		if tactile and contact_pos_max[1] < tip_pos[1] and is_stuck_tactile(): # higher y value means left
			print("Starting CW interrupt")
			global excavate_start_time_tactile 
			excavate_start_time_tactile = time.time() - t_init
			t_ex_tactile = time.time() - t_init
			while t_ex_tactile - excavate_start_time_tactile < excavate_time_thresh:
				pb.stepSimulation()
				interrupt_CW_excavate(t_ex=t_ex_tactile)
				t_ex_tactile = time.time() - t_init
			excavate_start_time = time.time() - t_init # or set excavate_start_time to t_ex - excavate_time_thresh to exit and resume straight/burrow?
			t_ex = time.time() - t_init
			# print("t_ex inside perform: ", t_ex)
			# print("excavate_start_time inside perform: ", excavate_start_time)
			target_vel = vel_mag * np.array([-0.25, 0])
			target_ang_vel = 0
		else:
			target_vel = vel_mag * np.array([0, -1])
			target_ang_vel = -ang_vel_mag*puck_orn
	elif t_ex - excavate_start_time < 4*excavate_time_thresh/8: #fwd
		target_vel = vel_mag * np.array([1, 0])
		target_ang_vel = 0
	elif t_ex - excavate_start_time < 5*excavate_time_thresh/8: #right/rotate CW/fwd
		target_vel = vel_mag * np.array([0.1, 1])
		target_ang_vel = 0.5*ang_vel_mag
	elif t_ex - excavate_start_time  < 6*excavate_time_thresh/8: #back/right/rotate CW
		target_vel = vel_mag * np.array([-1, 0.25])
		target_ang_vel = 0.5*ang_vel_mag
	else: # t_ex - excavate_start_time  <= excavate_time_thresh:  #left/rotate CCW
		target_vel = vel_mag * np.array([0, -0.25])
		target_ang_vel = -0.3*ang_vel_mag
	# else:
	# 	print("t_ex outside bounds of time cutoff")
	# 	print("t_ex end of perform: ", t_ex)
	# 	print("excavate_start_time end of perform: ", excavate_start_time)

	# print("Ending CW excavate")

	puck_obj.apply_velocity(target_vel, max_force=max_force)
	puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)


def run_straight_line_control(vel_mag = 0.5, ang_vel_mag = 0.35, max_force = 10, max_torque = 15):

	if approaching_goal(): ### Oscillation near goal fixed?? ###
		vel_mag = 0.25
		target_vel = vel_mag * target_vel_direction
	else:
		target_vel = vel_mag * target_vel_direction

	if pointed_at_goal(): 
		target_ang_vel = 0.01 * target_ang_vel_direction
	else:
		target_ang_vel = ang_vel_mag * target_ang_vel_direction

	puck_obj.apply_velocity(target_vel, max_force=max_force)
	puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)

def run_lateral_balance_control(vel_mag = 0.5, ang_vel_mag = 0.35):

	total_force, _ = puck_obj.get_external_force_torque() ## Doesnt actually use shear forces too, fix
	lateral_force = np.dot(total_force,perp_vel_direction)
	force_scaling = np.linalg.norm(target_vel_direction)/np.linalg.norm(lateral_force)
	
	if np.linalg.norm((lateral_force)) > 7.5:
		target_vel_balance = target_vel_direction + force_scaling*lateral_force
		if np.linalg.norm((target_vel_balance)) < 0.05:
			target_vel_balance = target_vel_direction
	else:
		target_vel_balance = target_vel_direction
	target_vel_direction_balance = (target_vel_balance) / np.linalg.norm((target_vel_balance))

	if approaching_goal(): 
		target_vel = vel_mag * target_vel_direction
	else:
		target_vel = vel_mag * target_vel_direction_balance

	if pointed_at_goal(): 
		target_ang_vel = 0 * target_ang_vel_direction
	else:
		target_ang_vel = ang_vel_mag * target_ang_vel_direction

	puck_obj.apply_velocity(target_vel, max_force=max_force)
	puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)


def run_react_normal_control(vel_mag = 0.5, ang_vel_mag = 0.35):

	total_force, _ = puck_obj.get_external_force_torque() ## Doesnt actually use shear forces too, fix
	force_scaling = np.linalg.norm(target_vel_direction)/np.linalg.norm(total_force)
	
	if np.linalg.norm((total_force)) > 7.5:
		target_vel_normal = target_vel_direction - force_scaling*total_force
		if np.linalg.norm((target_vel_normal)) < 0.05:
			target_vel_normal = target_vel_direction ####### Instead set to be tangent?? #################
	else:
		target_vel_normal = target_vel_direction
	target_vel_direction_normal = (target_vel_normal) / np.linalg.norm((target_vel_normal))

	if approaching_goal(): 
		target_vel = vel_mag * target_vel_direction
	else:
		target_vel = vel_mag * target_vel_direction_normal

	if pointed_at_goal(): 
		target_ang_vel = 0 * target_ang_vel_direction
	else:
		target_ang_vel = ang_vel_mag * target_ang_vel_direction

	puck_obj.apply_velocity(target_vel, max_force=max_force)
	puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)

def log_trial_result():
	filewriter.writerow([i, dist_to_goal, curr_time, num_obs, stuck_ctr, seed]) # , num_fixed_obs, num_cylinders]
	# print("Trial number: ", i)
	# print("Distance to goal: ", dist_to_goal)
	# print("Time taken: ", curr_time)

def approaching_goal():
	return (dist_to_goal < 0.1)

def pointed_at_goal():
	return (np.abs(orn_diff) < 0.05)

def at_goal():
	return dist_to_goal < 0.01

def is_stuck():
	global stop_time
	if (np.linalg.norm((puck_obj.get_linear_velocity())) > 0.05 or puck_obj.get_angular_velocity() > 0.05) and x_prime_prev - x_prime > 0.025:
		stop_time = curr_time
	return (curr_time > 3.0 and curr_time - stop_time > stop_time_cutoff and puck_obj.is_in_collision() \
		 and puck_obj.get_summed_contact_force_mags(exclusion_ids=[planeId, cluttered_scene.bottom_wall_id]) > 9.5)

def is_stuck_tactile():
	return (curr_time > 3.0 and np.linalg.norm((puck_obj.get_linear_velocity())) < 0.025 and puck_obj.is_in_collision() \
		 and contact_force_exceeds(puck_obj,force_thresh=7.5))
	
def is_stuck_proprio():
	global stop_time
	if np.linalg.norm((puck_obj.get_linear_velocity())) > 0.05 and x_prime_prev - x_prime > 0.025:
		stop_time = curr_time
	return (curr_time > 3.0 and curr_time - stop_time > stop_time_cutoff)
	
def is_being_resisted():
	return (puck_obj.is_in_collision() and puck_obj.get_summed_contact_force_mags(exclusion_ids=[planeId, cluttered_scene.bottom_wall_id]) > 1.5)





# control_type = "burrow" #"straight_line" #"lateral_balance" #"normal" #"excavate" #"hybrid" #
tactile = False
burrow = False
excavate = False
# avoid = True?
num_trials = 3
max_force = 10
max_torque = 15

seed_sequence = [0,1,2]

# seed_sequence = [563290, 599026, 196534, 961079, 507433, 920107, 726030, 291771, 169523, \
# 		956760, 888066, 165065, 476807, 904281, 392596, 180344, 291350, 345828, 491947, \
# 		996700] # Uncomment if repeating scenes with seeds filled in

log_results = True

if log_results:
	cwd = os.getcwd()
	now = datetime.now()
	date_string = now.strftime("%Y%m%d\\")
	time_string = now.strftime("%H%M%S_")
	log_dir_name = cwd + "\\Logging\\" + date_string 
	path_exists = os.path.exists(log_dir_name)

	if not path_exists:
		os.mkdir(log_dir_name)
	# log_filename = "reachTest_" + time_string + control_type + "_" + str(num_trials) + "_" + str(max_force) + "N_" + str(max_torque) + "Nm" + ".csv" # friction too?

	log_filename = "reachTest_" + time_string + "tactile" + str(tactile) + "_" + "burrow" + str(burrow) + "_" + "excavate" + str(excavate) + ".csv" 

	csvfile =  open(log_dir_name + log_filename, 'w')
	filewriter = csv.writer(csvfile, delimiter=',') #, quotechar='|', quoting=csv.QUOTE_MINIMAL)
	filewriter.writerow(['Trial #', 'Distance to goal', 'Completion time', "Number of obstacles", "Stuck counter", "Random seed"]) # , "Number of fixed obstacles", "Number of cylinders"]

# print("Event-driven: " + str(event_driven))
print("Burrow: " + str(burrow))
print("Excavate: " + str(excavate))

for i in range(num_trials): 

	# Replay specific scenario to see what happened
	# seed = 0 # desired seed to be repeated
	
	# Run this to generate new random scene (total of 1 mil)
	# seed = np.random.randint(0,1e6) 


	# To run multiple strategies on the same scenes:

	# # Run this if the first time going through sequence
	# seed = np.random.randint(0,1e6) 

	# # Run this on subsequent strategies to repeat scenes (fill in seed_sequence and uncomment first!)
	seed = seed_sequence[i]

	np.random.seed(seed) 

	# Initialize pybullet physics
	physicsClient = pb.connect(pb.GUI)
	pb.setAdditionalSearchPath(pybullet_data.getDataPath())
	grav_constant = -9.81
	pb.setGravity(0, 0, grav_constant)
	planeId = pb.loadURDF("plane.urdf")

	# Generate random scene 
	num_obs = np.random.randint(20,31) #(20,26) # 
	# num_fixed_obs = 0 # Not reflected inside cluttered_scene.py yet
	# num_cylinders = 0 # Not reflected inside cluttered_scene.py yet
	cluttered_scene = ClutteredScene(pb, planeId, num_obs=num_obs)
	cluttered_scene.generate_clutter()

	# Add puck robot
	puck_height = 1.0
	puck_placement_height = puck_height/2.0 + cluttered_scene.scene_height + 0.01
	puck_x_pos = -2
	puck_y_pos = np.random.uniform(-1.5,1.5)
	puck_start_pos = np.array([puck_x_pos, puck_y_pos, puck_placement_height])
	puck_obj = Object2D(pb, "forearmJoint", puck_start_pos)
	tip_goal_y_pos = np.random.uniform(-1.5,1.5) #(-1,1) # 
	tip_goal_x_pos = 4.5 #5.1 #
	tip_goal_pos = np.array([tip_goal_x_pos, tip_goal_y_pos, puck_placement_height])
	pb.addUserDebugText(text="Target",textPosition=tip_goal_pos+np.array(([-0.25,0.5,-0.5])),textColorRGB=[0,1,0],textSize=1.0)
	# puck_fric_coeff = 0.5
	# pb.changeDynamics(puck_obj.obj_id,linkIndex=-1,lateralFriction=puck_fric_coeff,contactDamping = 10.0,contactStiffness=10.0)
	# puck_dyn = puck_obj.get_mass() #pb.getDynamicsInfo(puck_obj.obj_id,linkIndex=-1)
	puck_length = 0.158*10
	# print("Puck mass:", puck_dyn)

	puck_orn = 0.0
	tip_start_pos = puck_start_pos + np.array([puck_length*np.cos(puck_orn), puck_length*np.sin(puck_orn), 0])

	# Run controller
	init_vel_mag = 0.5 #1.0
	init_ang_vel_mag = 0.35
	original_target_vel_direction = (tip_goal_pos - tip_start_pos) / np.linalg.norm((tip_goal_pos - tip_start_pos))
	target_vel = init_vel_mag * original_target_vel_direction
	target_orn = 0.0
	original_target_ang_vel_direction = 0
	target_ang_vel = init_ang_vel_mag * original_target_ang_vel_direction

	t_init = time.time() 

	stop_time = 0
	stop_time_cutoff = 2.0

	print_time = 0.0
	print_time_cutoff = 2.0
	
	plot_force_time = 0
	plot_force_time_cutoff = 0.25

	store_contact_time = 0
	store_contact_time_cutoff = 1.0
	contact_map = None
	obstacle_map = None

	prog_time = 0
	prog_time_cutoff = 5.0
	x_prime = np.linalg.norm((tip_goal_pos-tip_start_pos))
	x_prime_prev = x_prime + 0.5

	excavate_time = 0
	excavate_time_thresh = 3

	stuck_ctr = 0
	stuck_ctr_thresh = 30

	# replan_ctr = 0
	# replan_thresh = 10

	print("Trial #: ", i)

	still_running = True

	# pb.setRealTimeSimulation(1)
	step_num = 0
	while still_running:
		pb.stepSimulation()
		step_num += 1

		puck_pos = puck_obj.get_position()
		puck_orn = puck_obj.get_orientation()

		tip_pos = puck_pos + np.array([puck_length*np.cos(puck_orn), puck_length*np.sin(puck_orn), 0])

		vec_to_goal = tip_goal_pos - tip_pos
		target_orn = np.arctan2(vec_to_goal[1],vec_to_goal[0])
		orn_diff = target_orn - puck_orn

		dist_to_goal = np.linalg.norm((vec_to_goal))
		target_vel_direction = (vec_to_goal) / dist_to_goal
		target_ang_vel_direction = np.sign(orn_diff)

		z_vec = np.array([0,0,1])
		perp_vel_direction = np.cross(target_vel_direction, z_vec)

		curr_time = time.time() - t_init

		x_prime = np.linalg.norm((tip_goal_pos-tip_pos))
		if curr_time - prog_time > prog_time_cutoff:
			x_prime_prev = x_prime
			prog_time = curr_time

		# if curr_time - plot_force_time > plot_force_time_cutoff:
		# 	plot_contact_forces(puck_obj,lifetime=0)
		# 	plot_force_ctr = 0

		if curr_time - store_contact_time > store_contact_time_cutoff:
			contact_map, obstacle_map = store_contact_locations(puck_obj, contact_map, obstacle_map)
			plot_force_ctr = 0

		if at_goal(): #Finished trial, log result and shut down trial
			run_stop()
			print("Success!")
			if log_results:
				log_trial_result()
			pb.disconnect()
			still_running = False
			# plot_contact_map(contact_map, obstacle_map)
		
		elif curr_time > 180.0:
			run_stop()
			print("Failure: max time exceeded.")
			if log_results:
				log_trial_result()
			pb.disconnect()
			still_running = False
			# plot_contact_map(contact_map, obstacle_map)
		
		elif (tactile == False and is_stuck_proprio()) or (tactile and is_stuck_tactile()): #Stuck, either excavate or log result and shut down trial
			print("Stuck! #: ", stuck_ctr)
			stuck_ctr += 1
			if stuck_ctr > stuck_ctr_thresh or excavate == False:
				run_stop()
				print("Failure: max excavation manuevers used.")
				if log_results:
					log_trial_result()
				pb.disconnect()
				still_running = False
				# plot_contact_map(contact_map, obstacle_map)
			else:
				######## ONLY DO THIS IF YOU'VE GOT 3+ SECONDS LEFT ########
				if tactile == True: 
					contact_pos_max = get_impeding_contact_loc(puck_obj)
					if contact_pos_max[1] < tip_pos[1]: # higher y value means left
						excavate_direction_flag = "CCW"
					else:
						excavate_direction_flag = "CW"
				else:
					if stuck_ctr%2 == 0:
						excavate_direction_flag = "CCW"
					else:
						excavate_direction_flag = "CW"
				
				if excavate_direction_flag == "CCW":
					print("Starting CCW excavate")
				else:
					print("Starting CW excavate")

				excavate_start_time = time.time() - t_init
				t_ex = time.time() - t_init
				while t_ex - excavate_start_time < excavate_time_thresh:
					pb.stepSimulation()
					if excavate_direction_flag == "CCW":
						perform_CCW_excavate(t_ex=t_ex)
					else:
						perform_CW_excavate(t_ex=t_ex)
					# pb.stepSimulation()
					t_ex = time.time() - t_init
				stop_time = t_ex
		
		## Replanning?

		else: # Not stuck or there yet, proceed straight or while burrowing
			if burrow:
				if tactile:
					if puck_obj.is_in_collision(): #is_being_resisted():
						run_burrow_control()
					else: 
						run_straight_line_control()
				else:
					run_burrow_control()
			else:
				run_straight_line_control()
		
		# else:
		# 	run_straight_line_control()

			# if control_type == "straight_line": 
			# 	run_straight_line_control()
			# elif control_type == "burrow":
			# 	run_burrow_control()
			# elif control_type == "lateral_balance":
			# 	run_lateral_balance_control()
			# elif control_type == "normal":
			# 	run_react_normal_control()

# if i == num_trials - 1:
# 	print("Finished trials!")
if log_results:
	csvfile.close()
# print("Results: ", results_log)





