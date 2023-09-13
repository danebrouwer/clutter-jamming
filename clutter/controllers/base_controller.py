# File: base_controller.py
# Authors: Dane Brouwer, Marion Lepert, Joshua Citron
# Description: 
#
import pybullet as pb
import numpy as np 
from collections import defaultdict

from clutter.object2D import Object2D
from clutter.cluttered_scene import ClutteredScene
from clutter.utils import plot_contact_map, store_contact_locations, contact_at_tip

class BaseController():
	def __init__(self, physics_client, params): 
		self.physics_client = physics_client
		self.map_contacts = False
		self.total_step_thresh = params['total_step_thresh']
		self.results_list = None
		self.results_dict = defaultdict(list)
		self.params = params
		self.scene_depth = None

		self.init_vel_mag = self.params['controller']['init_vel_mag']

		self.progress_time_cutoff = self.params['controller']['progress_time_cutoff']
		
		self.position_limit = None
		self.dist_to_goal_initial = None
		self.init_stuck = None
		self.vec_to_goal = None
		self.stuck_clock = None
		self.broken_contact_ctr = 0

		self.entered_excavate = False
		self.excavate_step_thresh = params["controller"]["excavate_step_thresh"]
		self.trigger_excavate_step_thresh = params["controller"]["trigger_excavate_step_thresh"]

		self.bur_amp = params['controller']['bur_amp']
		self.bur_freq = params['controller']['bur_freq'] * np.pi

		self.num_obs = None # None


	def reset(self, seed, scene_depth, avg_size):
		self.seed = seed
		self.scene_depth = scene_depth
		self.avg_size = avg_size
		 
		# Generate random scene 
		grav_constant = self.params['scene']['grav_constant']
		pb.setGravity(0, 0, grav_constant)
		self.planeId = pb.loadURDF("plane.urdf")
		self.num_obs = np.random.randint(20,31) 
		self.cluttered_scene = ClutteredScene(pb, self.planeId, self.scene_depth, self.avg_size, num_obs=self.num_obs)
		self.cluttered_scene.generate_clutter()

		# Add puck robot
		# self.scene_depth = self.params['scene']['depth'], used to be uncommented only commented for experiment3
		puck_placement_height = self.params['puck']['height']/2.0 + self.cluttered_scene.scene_height + 0.01
		puck_x_pos = self.params['puck']['init_x_pos']
		puck_y_pos = np.random.uniform(-1.5,1.5)
		puck_start_pos = np.array([puck_x_pos, puck_y_pos, puck_placement_height])
		self.puck_obj = Object2D(pb, "forearmJoint", puck_start_pos)
		# recently added this line
		self.puck_obj.add_friction()
		self.puck_length = self.params['puck']['length']
		tip_goal_y_pos = np.random.uniform(-1.5,1.5) #(-1,1)
		tip_goal_x_pos = self.scene_depth + puck_x_pos + self.puck_length + 0.25
		self.tip_goal_pos = np.array([tip_goal_x_pos, tip_goal_y_pos, puck_placement_height])
		# pb.addUserDebugText(text="Target",
		#       textPosition=self.tip_goal_pos+np.array(([-0.25,0.5,-0.5])),
		# 	  textColorRGB=[0,1,0],textSize=1.0)
		
		self.puck_orn = self.params['puck']['orn']
		tip_start_pos = puck_start_pos + np.array([self.puck_length*np.cos(self.puck_orn), 
					     self.puck_length*np.sin(self.puck_orn), 0])
		x_prime = np.linalg.norm((self.tip_goal_pos-tip_start_pos))
		self.x_prime_prev = x_prime #+ 0.5

		dx_prime = x_prime - self.x_prime_prev
		self.dx_prime_prev = dx_prime

		self.stop_step = 0
		self.tip_contact_step = 0
		self.stop_step_cutoff_clock = self.params['controller']['stop_step_cutoff_clock']
		self.stop_step_cutoff_event = self.params['controller']['stop_step_cutoff_event']
		self.tip_contact_step_thresh = self.params['controller']['tip_pos_contact_thresh']
		self.v_prime_thresh = self.params['controller']['v_prime_thresh']
		self.contact_map = None
		self.force_map = None
		self.gamma = self.params['controller']['gamma']
		self.stuck_ctr = 0
		self.stuck_ctr_thresh = 75
		self.curr_step = 0
		self.init_stuck = 0

		self.tip_pos = puck_start_pos + np.array([self.puck_length*np.cos(self.puck_orn), self.puck_length*np.sin(self.puck_orn), 0]) #+ np.array([self.puck_length*np.cos(self.curr_orn), self.puck_length*np.sin(self.curr_orn), 0]
		self.vec_to_goal = self.tip_goal_pos - self.tip_pos
		target_orn = np.arctan2(self.vec_to_goal[1],self.vec_to_goal[0])
		self.orn_diff = target_orn - self.puck_orn
		# print(self.vec_to_goal)
		self.dist_to_goal_initial = np.linalg.norm((self.vec_to_goal)) + self.init_vel_mag*self.progress_time_cutoff/240

	def update_state(self): 
		self.puck_pos = self.puck_obj.get_position()
		self.puck_orn = self.puck_obj.get_orientation()

		self.tip_pos = self.puck_pos + np.array([self.puck_length*np.cos(self.puck_orn), self.puck_length*np.sin(self.puck_orn), 0])

		self.vec_to_goal = self.tip_goal_pos - self.tip_pos
		target_orn = np.arctan2(self.vec_to_goal[1], self.vec_to_goal[0])
		self.orn_diff = target_orn - self.puck_orn

		self.dist_to_goal = np.linalg.norm((self.vec_to_goal))
		self.target_vel_direction = (self.vec_to_goal) / self.dist_to_goal
		self.target_ang_vel_direction = np.sign(self.orn_diff)

		z_vec = np.array([0,0,1])
		self.perp_vel_direction = np.cross(self.target_vel_direction, z_vec)

		x_prime = np.linalg.norm((self.tip_goal_pos-self.tip_pos))
		dx_prime = x_prime - self.x_prime_prev
		self.x_prime_prev = x_prime
		self.v_prime_filt = dx_prime*self.gamma + (1-self.gamma)*self.dx_prime_prev
		self.dx_prime_prev = dx_prime
		# print(self.curr_step)
		self.position_limit = self.dist_to_goal_initial - self.init_vel_mag * self.stuck_clock/240
		# print("stuck clock: ", self.stuck_clock)
	def execute_action(self): 
		throw("Not implemented")

	def step(self): 
		self.curr_step += 1
		self.stuck_clock = self.curr_step - self.init_stuck
		pb.stepSimulation()
		self.update_state()
		done = self.execute_action()

		# print(contact_at_tip(self.puck_obj, self.tip_pos))
		if not contact_at_tip(self.puck_obj, self.tip_pos):
			self.broken_contact_ctr += 1
			# print("broken contact: ", self.broken_contact_ctr)
			if self.broken_contact_ctr > 5:
				self.tip_contact_step = self.curr_step
		else: 
			self.broken_contact_ctr = 0
		
		if self.v_prime_filt < -self.v_prime_thresh: 
			self.stop_step = self.curr_step
			
		if self.map_contacts:
			self.contact_map, self.force_map = store_contact_locations(self.puck_obj, self.contact_map, self.force_map)
		return done, self.curr_step
	
		

	def run_stop(self, vel_mag = 0.0, ang_vel_mag = 0.0, max_force = 10, max_torque = 15):
		target_vel = vel_mag * self.target_vel_direction
		target_ang_vel = ang_vel_mag * self.target_ang_vel_direction
		self.puck_obj.apply_velocity(target_vel, max_force=max_force)
		self.puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)

	def close_trial(self, test_case, dist_to_goal, curr_step, stuck_ctr, num_obs, seed):
		self.run_stop()
		if self.map_contacts:
			plot_contact_map(self.contact_map,self.force_map)
		if dist_to_goal <= self.params['controller']['at_goal_dist']:
			success_time = curr_step/self.total_step_thresh
		else:
			success_time = "NaN"
		trial_results = {'Control Type':test_case, 'Distance to goal':dist_to_goal/self.scene_depth, 
		    'Completion time':curr_step/self.total_step_thresh, "Success time": success_time, 
			"Stuck counter":stuck_ctr, "Number of obstacles": num_obs, "Random seed":seed, "Burrow amplitude": self.bur_amp, "Burrow frequency": self.bur_freq,
			"Excavate step thresh": self.excavate_step_thresh, "Trigger excavate step thresh": self.trigger_excavate_step_thresh, "Scene depth": self.scene_depth,
			"Average object size": self.avg_size}
		for key, val in trial_results.items(): 
			self.results_dict[key].append(val)
		return self.results_dict

	def approaching_goal(self):
		return (self.dist_to_goal < self.params['controller']['approaching_goal_dist'])

	def pointed_at_goal(self):
		return (np.abs(self.orn_diff) < self.params['controller']['pointed_at_goal_dist'])

	def at_goal(self):
		return self.dist_to_goal < self.params['controller']['at_goal_dist']

	def is_stuck_proprio(self): ## Instead implement separate version inside clock vs event cases?
		# Marion version:
		# return (self.curr_step > self.params['controller']['stuck_step_cutoff'] and (self.curr_step - self.stop_step) > 2*self.stop_step_cutoff) 
		print(self.position_limit <= self.dist_to_goal)
		return self.position_limit <= self.dist_to_goal 

	def is_stuck_tactile(self): # Use contact_force_exceeds() instead?
		# print(self.puck_obj.get_summed_contact_force_mags(exclusion_ids=[self.planeId, self.cluttered_scene.bottom_wall_id]))
		# print(self.puck_obj.is_in_collision() and \
		# 	self.puck_obj.get_summed_contact_force_mags(exclusion_ids=[self.planeId, self.cluttered_scene.bottom_wall_id]) > \
		# 	self.params['controller']['f_thresh_excv'])
		return self.puck_obj.is_in_collision() and \
			self.puck_obj.get_summed_contact_force_mags(exclusion_ids=[self.planeId, self.cluttered_scene.bottom_wall_id]) > \
			self.params['controller']['f_thresh_excv']
	
	def is_being_resisted(self): # Use contact_force_exceeds() instead?
		# print(self.puck_obj.get_summed_contact_force_mags(exclusion_ids=[self.planeId, self.cluttered_scene.bottom_wall_id]))
		# print(self.puck_obj.is_in_collision() and self.puck_obj.get_summed_contact_force_mags(exclusion_ids=[self.planeId, self.cluttered_scene.bottom_wall_id]) > \
		# 		self.params['controller']['f_bur_thresh'])
		return self.puck_obj.is_in_collision() and \
	  		self.puck_obj.get_summed_contact_force_mags(exclusion_ids=[self.planeId, self.cluttered_scene.bottom_wall_id]) > \
				self.params['controller']['f_bur_thresh']
