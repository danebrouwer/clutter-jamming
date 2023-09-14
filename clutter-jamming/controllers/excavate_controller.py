# File: excavate_controller.py
# Authors: Dane Brouwer, Marion Lepert
# Description: Contains the class that
# implements the excavate control strategy.

# Import relevant modules and classes.
import numpy as np
import pybullet as pb
from clutter.controllers.burrow_controller import BurrowController
from clutter.utils import store_contact_locations

# Implements the excavate control strategy.
class ExcavateController(BurrowController):  
    def __init__(self, physics_client, params):
        super().__init__(physics_client, params)
        self.test_case = "Excavate"
        self.stuck_ctr_thresh = params["controller"]["stuck_ctr_thresh"]
        self.excavate_scale = params["controller"]["excavate_scale"]
        self.clock_excv_chance = params["controller"]["clock_excv_chance"]

    # Overriding BaseController reset function.
    def reset(self, seed, scene_depth, avg_size): 
        super().reset(seed, scene_depth, avg_size)
        self.prev_excavate_step = 0 

    # Value returned determines whether or not an excavate should occur.
    def excavate_trigger_timeout(self): 
        return self.curr_step - self.prev_excavate_step > self.trigger_excavate_step_thresh

    # Overriding BurrowController is_done function. Additional check for ending trial
    # is if the strategy gets stuck too often.
    def is_done(self): 
        return self.at_goal() or (self.curr_step >= self.total_step_thresh) or \
            (self.excavate_trigger_timeout() and self.stuck_ctr > self.stuck_ctr_thresh) 

    # If not done with trial, determines whether to excavate or run straight line control.
    def execute_action(self): 
        if self.is_done(): 
            done = True 
            self.close_trial(self.test_case, self.dist_to_goal, self.curr_step, self.stuck_ctr, self.num_obs, self.seed)
        elif self.excavate_trigger_timeout() and self.total_step_thresh - self.curr_step >= self.excavate_step_thresh: 
            done = False
            self.stuck_ctr += 1
            # Equal chance of excavate happening if self.clock_excv_chance set
            # to 0.5
            if np.random.uniform(0,1) < self.clock_excv_chance:
                excavate_direction_flag = "CCW"
            else:
                excavate_direction_flag = "CW"
            excavate_start_step = self.curr_step
            excavate_step = excavate_start_step
            # Perform excavate for a given duration.
            while excavate_step - excavate_start_step < self.excavate_step_thresh:
                pb.stepSimulation()
                self.curr_step += 1

                if self.map_contacts:
                    self.contact_map, self.force_map = store_contact_locations(self.puck_obj, self.contact_map, self.force_map)

                t_ex = excavate_step - excavate_start_step
                if excavate_direction_flag == "CCW":
                    self.perform_CCW_excavate(t_ex, self.excavate_scale, self.excavate_step_thresh, self.puck_orn)
                else:
                    self.perform_CW_excavate(t_ex, self.excavate_scale, self.excavate_step_thresh, self.puck_orn)
                excavate_step = self.curr_step
            self.stop_step = excavate_step
            self.prev_excavate_step = excavate_step
        else:
            done = False
            self.run_straight_line_control()

        return done 
    
    def perform_CCW_excavate(self, t_ex, excavate_scale, excavate_step_thresh, orn, vel_mag = 0.5, ang_vel_mag = 0.5, max_force = 10, max_torque = 15):
        scale = excavate_scale
        A0 = vel_mag/scale
        B0 = ang_vel_mag/scale/scale
        t_tot = excavate_step_thresh
        t = t_ex
        t_frac = t/t_tot

        theta_dot = -B0*(1+(scale-1)*t_frac)*np.sin(t_frac*(2*np.pi))
        vx_prime = -A0*(1+(scale-1)*t_frac)*np.cos(t_frac*(3*np.pi/2))
        vy_prime = -A0*(1+(scale-1)*t_frac)*np.sin(t_frac*(3*np.pi/2)) - self.puck_length*theta_dot

        v_prime = np.array([vx_prime, vy_prime])
        rot = np.array(([np.cos(orn), -np.sin(orn)],[np.sin(orn), np.cos(orn)]))

        target_vel = rot @ v_prime
        target_ang_vel = theta_dot
        
        self.puck_obj.apply_velocity(target_vel, max_force=max_force)
        self.puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)

    def perform_CW_excavate(self, t_ex, excavate_scale, excavate_step_thresh, orn, vel_mag = 0.5, ang_vel_mag = 0.5, max_force = 10, max_torque = 15):
        scale = excavate_scale
        A0 = vel_mag/scale
        B0 = ang_vel_mag/scale/scale
        t_tot = excavate_step_thresh
        t = t_ex
        t_frac = t/t_tot

        theta_dot = B0*(1+(scale-1)*t_frac)*np.sin(t_frac*(2*np.pi))
        vx_prime = -A0*(1+(scale-1)*t_frac)*np.cos(t_frac*(3*np.pi/2))
        vy_prime = A0*(1+(scale-1)*t_frac)*np.sin(t_frac*(3*np.pi/2)) - self.puck_length*theta_dot

        v_prime = np.array([vx_prime, vy_prime])
        rot = np.array(([np.cos(orn), -np.sin(orn)],[np.sin(orn), np.cos(orn)]))

        target_vel = rot @ v_prime
        target_ang_vel = theta_dot

        self.puck_obj.apply_velocity(target_vel, max_force=max_force)
        self.puck_obj.apply_ang_velocity(target_ang_vel, max_torque=max_torque)