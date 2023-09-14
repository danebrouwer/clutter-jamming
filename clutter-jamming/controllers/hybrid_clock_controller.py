# File; hybrid_clock_controller.py
# Authors: Dane Brouwer, Marion Lepert
# Description: Contains class that implements
# hybrid clock control strategy.

# Import relevant modules, classes, and functions.
import numpy as np
import pybullet as pb
from controllers.excavate_controller import ExcavateController
from utils import store_contact_locations

# Implements the hybrid clock control strategy.
class HybridClockController(ExcavateController):  
    def __init__(self, physics_client, params):
        super().__init__(physics_client, params)
        self.test_case = "Hybrid Clock" 
    # If not done with trial, excavate at set intervals. When
    # not excavating, burrow.
    def execute_action(self): 
        if self.is_done(): 
            done = True 
            self.close_trial(self.test_case, self.dist_to_goal, self.curr_step, self.stuck_ctr, self.num_obs, self.seed)
        elif self.excavate_trigger_timeout() and self.total_step_thresh - self.curr_step >= self.excavate_step_thresh: 
            done = False
            self.stuck_ctr += 1
            if np.random.uniform(0,1) < self.clock_excv_chance:
                excavate_direction_flag = "CCW"
            else:
                excavate_direction_flag = "CW"

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
            self.stop_step = excavate_step
            self.prev_excavate_step = excavate_step
        else:
            done = False
            self.run_burrow_control(self.curr_step, self.bur_amp, self.bur_freq)

        return done 
