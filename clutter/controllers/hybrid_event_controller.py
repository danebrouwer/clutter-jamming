import pdb
import numpy as np
import pybullet as pb

# from clutter.controllers.burrow_controller import BurrowController
from clutter.controllers.excavate_controller import ExcavateController
from clutter.utils import store_contact_locations, get_impeding_contact_loc

class HybridEventController(ExcavateController):  
    def __init__(self, physics_client, params):
        super().__init__(physics_client, params)
        self.test_case = "Hybrid Event"
        self.event_excv_chance = self.params['controller']['event_excv_chance']

    
    # def excavate_trigger_timeout(self): # Use separate param for clock vs event trigger thresh?
    #     return self.curr_step - self.prev_excavate_step > self.trigger_excavate_step_thresh
    
    def is_stuck(self):
        a = self.is_stuck_tactile()  # self.excavate_trigger_timeout() and
        b = self.is_stuck_proprio()
        c = self.is_pushing_object()
        # if self.curr_step % 20 == 0:
        #     print("position limit: ", self.position_limit, "dist to goal: ", self.dist_to_goal)
        # if a and b:
        #     print("\nstuck\n")
            
        # if c:
        #     print("\npushing\n")
        #     # print("force: ", self.get)
        return (a and b) or c #self.excavate_trigger_timeout()
    
    def is_pushing_object(self):
        # print("time: ", np.round((self.curr_step - self.tip_contact_step),4))
        return self.curr_step - self.tip_contact_step > self.tip_contact_step_thresh


    def execute_action(self): 
        if self.is_done(): 
            done = True 
            self.close_trial(self.test_case, self.dist_to_goal, self.curr_step, self.stuck_ctr, self.num_obs, self.seed)
        elif self.is_stuck() and self.total_step_thresh - self.curr_step >= self.excavate_step_thresh: 
            done = False
            self.stuck_ctr += 1
            print("\nStuck ctr: ", self.stuck_ctr)

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
            # print("position limit: ", self.position_limit, "dist to goal: ", self.dist_to_goal)
            # print("Stuck?: ", self.is_stuck())
            # print("Pushing?: ", self.is_pushing_object())
            # print("Performing excavate now")
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
                # pbar.update(1)
            self.stop_step = excavate_step
            self.prev_excavate_step = excavate_step
            self.init_stuck = self.curr_step
            # print("offset: ", self.init_vel_mag*self.progress_time_cutoff/240)
            self.dist_to_goal_initial = np.linalg.norm((self.vec_to_goal)) + self.init_vel_mag*self.progress_time_cutoff/240
            self.stuck_clock = self.curr_step - self.init_stuck
            self.position_limit = self.dist_to_goal_initial - self.init_vel_mag * self.stuck_clock/240
            self.tip_contact_step = self.curr_step
            # print("position limit: ", self.position_limit, "dist to goal: ", self.dist_to_goal)
            # print("Stuck?: ", self.is_stuck())
            # print("Pushing?: ", self.is_pushing_object())
            # print("Finished excavate")
        elif self.is_being_resisted():
            done = False
            self.run_burrow_control(self.curr_step, self.bur_amp, self.bur_freq)
        else:
            done = False
            self.run_straight_line_control()

        return done 
    
