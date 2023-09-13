# File: hybrid_event_controller.py
# Authors: Dane Brouwer, Marion Lepert
# Description: Contains the class that
# implements the hybrid event control
# strategy

# Import relevant modules, classes, and functions.
import numpy as np
import pybullet as pb
from clutter.controllers.excavate_controller import ExcavateController
from clutter.utils import store_contact_locations, get_impeding_contact_loc

# Implements hybrid event control strategy.
class HybridEventController(ExcavateController):  
    def __init__(self, physics_client, params):
        super().__init__(physics_client, params)
        self.test_case = "Hybrid Event"
        self.event_excv_chance = self.params['controller']['event_excv_chance']

    # Checks for scenarios that would trigger an excavate. 
    def is_stuck(self):
        a = self.is_stuck_tactile()
        b = self.is_stuck_proprio()
        c = self.is_pushing_object()
        return (a and b) or c
    
    # Returns true if object has been pushed for certain amount of time.
    def is_pushing_object(self):
        return self.curr_step - self.tip_contact_step > self.tip_contact_step_thresh

    # If not done with trial, check to see if there is light contact. If
    # this exceeds a certain threshold, burrow. If there is heavier contact
    # that meets the conditions to trigger an excavate, excavate. If contact
    # does not exceed a threshold, run straight line control strategy.
    def execute_action(self): 
        if self.is_done(): 
            done = True 
            self.close_trial(self.test_case, self.dist_to_goal, self.curr_step, self.stuck_ctr, self.num_obs, self.seed)
        elif self.is_stuck() and self.total_step_thresh - self.curr_step >= self.excavate_step_thresh: 
            done = False
            self.stuck_ctr += 1
            contact_pos_max = get_impeding_contact_loc(self.puck_obj)
            if contact_pos_max[1] > self.tip_pos[1]:
                if np.random.uniform(0,1) < self.event_excv_chance:
                    excavate_direction_flag = "CCW"
                else:
                    excavate_direction_flag = "CW"
            else:
                if np.random.uniform(0,1) < self.event_excv_chance:
                    excavate_direction_flag = "CW"
                else:
                    excavate_direction_flag = "CCW"

            excavate_start_step = self.curr_step
            excavate_step = excavate_start_step
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
            self.prev_excavate_step = excavate_step
            self.init_stuck = self.curr_step
            self.dist_to_goal_initial = np.linalg.norm((self.vec_to_goal)) + self.init_vel_mag*self.progress_time_cutoff/240
            self.stuck_clock = self.curr_step - self.init_stuck
            self.position_limit = self.dist_to_goal_initial - self.init_vel_mag * self.stuck_clock/240
            self.tip_contact_step = self.curr_step
        elif self.is_being_resisted():
            done = False
            self.run_burrow_control(self.curr_step, self.bur_amp, self.bur_freq)
        else:
            done = False
            self.run_straight_line_control()

        return done 
    
