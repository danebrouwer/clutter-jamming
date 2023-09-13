# File: burrow_controller.py
# Authors: Dane Brouwer, Marion Lepert, Joshua Citron
# Description: 
#

import numpy as np
from clutter.controllers.straight_line_controller import StraightLineController

class BurrowController(StraightLineController): # Did I do the inherit right?
    def __init__(self, physics_client, params):
        super().__init__(physics_client, params)
        self.test_case = "Burrow"

    def is_done(self): 
        return self.at_goal() or (self.curr_step >= self.total_step_thresh) #or self.is_stuck_proprio(self.stop_step_cutoff_clock)

    def execute_action(self): 
        if self.is_done():
            done = True
            self.close_trial(self.test_case, self.dist_to_goal, self.curr_step, self.stuck_ctr, self.num_obs, self.seed)
        else:
            done = False
            self.run_burrow_control(self.curr_step, self.bur_amp, self.bur_freq)
        return done 

    def run_burrow_control(self, curr_step, bur_amp, bur_freq, vel_mag = 0.5, ang_vel_mag = 0.35, max_force = 10, max_torque = 15):
        sin_term = np.sin(bur_freq*curr_step) 
        
        target_vel_sin = self.target_vel_direction + self.perp_vel_direction*bur_amp*sin_term
        target_vel_direction_sin = (target_vel_sin) / np.linalg.norm((target_vel_sin))
        target_vel = vel_mag * target_vel_direction_sin

        if self.pointed_at_goal(): 
            target_ang_vel = 0 * self.target_ang_vel_direction
        else:
            target_ang_vel = ang_vel_mag * self.target_ang_vel_direction

        self.puck_obj.apply_velocity(target_vel, max_force=max_force)
        self.puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)