# File: straight_line_controller.py
# Authors: Dane Brouwer, Marion Lepert
# Description: Contains class that implements
# straight line control strategy.

# Import relevant classes.
from controllers.base_controller import BaseController

# Implements straight line control strategy.
class StraightLineController(BaseController):
	def __init__(self, physics_client, params): 
		super().__init__(physics_client, params)
		self.test_case = "Straight Line"
		
	def is_done(self): 
		return self.at_goal() or (self.curr_step >= self.total_step_thresh)
	
	def execute_action(self): 
		if self.is_done(): 
			done = True
			self.close_trial(self.test_case, self.dist_to_goal, self.curr_step, self.stuck_ctr, self.num_obs, self.seed)
		else:
			done = False
			self.run_straight_line_control()
		return done 

	def run_straight_line_control(self, vel_mag=0.5, ang_vel_mag=0.35, max_force=10, max_torque=15):
		if self.approaching_goal(): 
			vel_mag = 0.25
			target_vel = vel_mag * self.target_vel_direction
		else:
			target_vel = vel_mag * self.target_vel_direction

		if self.pointed_at_goal(): 
			target_ang_vel = 0.01 * self.target_ang_vel_direction
		else:
			target_ang_vel = ang_vel_mag * self.target_ang_vel_direction

		self.puck_obj.apply_velocity(target_vel, max_force=max_force)
		self.puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)
