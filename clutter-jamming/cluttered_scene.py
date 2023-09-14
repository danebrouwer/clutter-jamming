# File: cluttered_scene.py
# Author: Marion Lepert
# Description: This file
# contains classes and functions
# that handle the creation of the scene in PyBullet.

# Third-party libraries 
import pybullet as pb
import numpy as np 

# Project libraries
from clutter.contact_point import * 
from clutter.object2D import *

class GhostObj: 
  def __init__(self, id, shape, radius=None, halfExtents=None): 
    self.id = id 
    self.shape = shape 
    self.radius = radius 
    self.halfExtents = halfExtents

# Implements the scene in PyBullet.
class ClutteredScene: 
  def __init__(self, pb, planeId, scene_depth, avg_size,
    paddle_buffer=0.1, initial_scene_size=[10,10],
    min_mass=0.25, max_mass=0.75, min_height=0.6, max_height=0.8, num_obs=15):
    self.pb = pb 
    self.planeId = planeId
    self.scene_depth = scene_depth
    self.scene_area = 3.75 * 5.5
    self.scene_width = self.scene_area/self.scene_depth

    self.paddle_mass = 100
    self.paddle_halfextent = 25
    self.paddle_buffer = paddle_buffer 
    self.initial_scene_size = initial_scene_size
    self.final_scene_size = [self.scene_depth, self.scene_width]
    self.min_radius = avg_size - avg_size/4
    self.max_radius = avg_size + avg_size/4
    self.min_box_size = avg_size - avg_size/4
    self.max_box_size = avg_size + avg_size/4
    self.min_mass = min_mass
    self.max_mass = max_mass
    self.fixed_obs_mass = 10.0
    self.min_height = min_height 
    self.max_height = max_height
    self.num_obs = num_obs

    self.x_pos = self.final_scene_size[0] /2
    self.y_pos = 0 


  def generate_paddles(self): 
    self.paddle_front_obj = Object2D(self.pb, "xpaddle", pos=[self.initial_scene_size[0]/2+self.x_pos+self.paddle_halfextent,0,0])
    self.paddle_back_obj = Object2D(self.pb, "xpaddle", pos=[-(self.initial_scene_size[0]/2+self.x_pos+self.paddle_halfextent),0,0])
    self.paddle_left_obj = Object2D(self.pb, "ypaddle", pos=[self.x_pos,-(self.initial_scene_size[0]/2+self.x_pos+self.paddle_halfextent),0], scene_depth=str(self.scene_depth))
    self.paddle_right_obj = Object2D(self.pb, "ypaddle", pos=[self.x_pos,(self.initial_scene_size[0]/2+self.x_pos+self.paddle_halfextent),0], scene_depth=str(self.scene_depth))

    self.restricted_ids = [self.planeId, self.paddle_front_obj.obj_id, self.paddle_back_obj.obj_id, 
                           self.paddle_left_obj.obj_id, self.paddle_right_obj.obj_id]


  def generate_obstacles(self): 
    max_x = self.x_pos + (self.initial_scene_size[0]/2 - self.max_radius)
    min_x = self.x_pos - (self.initial_scene_size[0]/2 - self.max_radius)
    max_y = self.y_pos + (self.initial_scene_size[1]/2 - self.max_radius)
    min_y = self.y_pos - (self.initial_scene_size[1]/2 - self.max_radius)

    self.obj_dict = {}
    while (self.pb.getNumBodies() < (self.num_obs + 5)):
        collision = True
        while collision: 
          mass = 0.1
          height = 0.2

          x_obs_pos = np.random.uniform(min_x, max_x)
          y_obs_pos = np.random.uniform(min_y, max_y)

          radius = np.random.uniform(self.min_radius, self.max_radius)
          box_x_size = np.random.uniform(self.min_box_size, self.max_box_size)
          box_y_size = np.random.uniform(self.min_box_size, self.max_box_size)
          half_extents = [box_x_size/2, box_y_size/2, height/2]

          if np.random.uniform(0,1) > 0.5: 
            shape = 'cylinder'
            obs_cid = self.pb.createCollisionShape(pb.GEOM_CYLINDER, radius=radius, height=height)
          else: 
            shape = 'rectangle'
            obs_cid = self.pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents)
          
          obs_id = self.pb.createMultiBody(mass, obs_cid, basePosition=[x_obs_pos,y_obs_pos,height/2.0])

          self.obj_dict[obs_id] = GhostObj(id=obs_id, shape=shape, radius=radius, halfExtents=half_extents)

          for _ in range(10): 
            self.pb.stepSimulation()
            collision = False
            cps = self.pb.getContactPoints()
            collision_ids = []
            for cp_data in cps:
              cp = ContactPoint(cp_data)
              if cp.bodyB_id not in self.restricted_ids and cp.bodyA_id not in self.restricted_ids:
                if cp.contact_distance < 1e-5:
                  collision = True
                  removed = False 
                  current_body_ids = [self.pb.getBodyUniqueId(x) for x in range(self.pb.getNumBodies())]

                  if cp.bodyA_id in current_body_ids: 
                    self.pb.removeBody(cp.bodyA_id)
                    del self.obj_dict[cp.bodyA_id]

                  if cp.bodyB_id in current_body_ids: 
                    self.pb.removeBody(cp.bodyB_id)
                    del self.obj_dict[cp.bodyB_id]


    self.obs_ids = list(self.obj_dict.keys()) 


  def squish_obstacles(self): 
    vel_mag = 1.0

    while 1:
      self.pb.stepSimulation()

      paddle_front_pos = self.paddle_front_obj.get_position()
      paddle_back_pos = self.paddle_back_obj.get_position()
      paddle_left_pos = self.paddle_left_obj.get_position()
      paddle_right_pos = self.paddle_right_obj.get_position()

      if paddle_front_pos[0] < self.x_pos + (self.paddle_halfextent + self.final_scene_size[0]/2 + self.paddle_buffer): 
        self.paddle_front_obj.apply_velocity([0,0])
      else: 
        self.paddle_front_obj.apply_velocity([-vel_mag,0])


      if paddle_back_pos[0] > self.x_pos - (self.paddle_halfextent + self.final_scene_size[0]/2 + self.paddle_buffer): 
        self.paddle_back_obj.apply_velocity([0,0])
      else: 
        self.paddle_back_obj.apply_velocity([vel_mag,0])

      if (paddle_front_pos[0] < self.x_pos + (self.paddle_halfextent + self.final_scene_size[0]/2 + self.paddle_buffer)) and (paddle_back_pos[0] > self.x_pos - (self.paddle_halfextent + self.final_scene_size[0]/2 + self.paddle_buffer)): 

        if paddle_left_pos[1] > self.y_pos - (self.paddle_halfextent + self.final_scene_size[1]/2): 
          self.paddle_left_obj.apply_velocity([0,0])
        else: 
          self.paddle_left_obj.apply_velocity([0,vel_mag])


        if paddle_right_pos[1] < self.y_pos + (self.paddle_halfextent + self.final_scene_size[1]/2): 
          self.paddle_right_obj.apply_velocity([0,0])
        else: 
          self.paddle_right_obj.apply_velocity([0,-vel_mag])

        if (paddle_left_pos[1] > self.y_pos - (self.paddle_halfextent + self.final_scene_size[1]/2)) and (paddle_right_pos[1] < self.y_pos + (self.paddle_halfextent + self.final_scene_size[1]/2)): 
          break 

  def remove_paddles(self): 
    self.paddle_front_obj.destruct()
    self.paddle_back_obj.destruct()
    self.paddle_left_obj.destruct()
    self.paddle_right_obj.destruct()
    self.pb.stepSimulation()


  def remove_obstacles(self, record_position=False): 
    for obs_id in self.obs_ids: 
      if record_position: 
        pos = np.array(self.pb.getBasePositionAndOrientation(obs_id)[0])
        ori = np.array(self.pb.getBasePositionAndOrientation(obs_id)[1])
        self.obj_dict[obs_id].pos = pos
        self.obj_dict[obs_id].ori = ori

      self.pb.removeBody(obs_id)


  def place_obstacles(self, use_labels=False): 
    obs_fric_coeff = 0.4
    final_obs_ids = []
    if not use_labels: 
      self.obj_mass_labels = {}
    for obs_id in self.obj_dict.keys(): 
      ghost_obj = self.obj_dict[obs_id]
      if use_labels: 
        if self.obj_mass_labels[obs_id] == 1: 
          mass = self.fixed_obs_mass
          color = [0.8,0.161,0.212,1]
        else: 
          mass = np.random.uniform(self.min_mass, self.max_mass)
          color = [0.22,0.525,0.592,1]
      else: 
        if np.random.uniform(0,1) > 1.0: 
          mass = self.fixed_obs_mass
          color = [0.8,0.161,0.212,1]
          self.obj_mass_labels[obs_id] = 1
        else: 
          mass = np.random.uniform(self.min_mass, self.max_mass)
          if mass < 0.4:
            color = [0.22*1.2,0.525*1.2,0.592*1.2,1] 
          elif mass < 0.6:
            color = [0.22*0.9,0.525*0.9,0.592*0.9,1] 
          else:
            color = [0.22*0.6,0.525*0.6,0.592*0.6,1] 
          
          self.obj_mass_labels[obs_id] = 0

      radius = ghost_obj.radius
      height = np.random.uniform(self.min_height, self.max_height)
      obs_pos = ghost_obj.pos 

      if ghost_obj.shape == 'cylinder': 
        obs_cid = self.pb.createCollisionShape(shapeType=pybullet.GEOM_CYLINDER, radius=ghost_obj.radius, height=height)
        obs_vid = self.pb.createVisualShape(shapeType=pybullet.GEOM_CYLINDER, radius=ghost_obj.radius, length=height, rgbaColor=color)

      elif ghost_obj.shape == 'rectangle': 
        half_extent = ghost_obj.halfExtents
        half_extent[2] = height/2.0
        obs_cid = self.pb.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent)
        obs_vid = self.pb.createVisualShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)


      final_obs_id = self.pb.createMultiBody(mass, baseCollisionShapeIndex=obs_cid, baseVisualShapeIndex=obs_vid, 
        basePosition=[obs_pos[0],obs_pos[1],height/2.0 + self.scene_height],
        baseOrientation=ghost_obj.ori)

      pb.changeDynamics(final_obs_id,linkIndex=-1,lateralFriction=obs_fric_coeff)

      final_obs_ids.append(final_obs_id)

    self.obs_ids = final_obs_ids

  def generate_clutter(self): 
    self.setGUICamera()
    self.generate_paddles()
    self.generate_obstacles()
    self.squish_obstacles()
    self.remove_paddles()
    self.remove_obstacles(record_position=True)
    self.add_cabinet_walls()
    self.place_obstacles()

  def reset_clutter(self): 
    self.remove_obstacles()
    self.place_obstacles(use_labels=True)


  def add_cabinet_walls(self): 
    ori = [0,0,0,1]
    color= [0.925, 0.706, 0.514, 1]
    mass = 100000
    width = 0.2
    height = 2.0
    bottom_width = 1.0
    wall_fric_coeff = 0.5

    # Bottom 
    half_extent = [self.final_scene_size[0]/2 + width , self.final_scene_size[1]/2 + width, bottom_width/2] #+ 1.0
    wall_cid = self.pb.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent)
    wall_vid = self.pb.createVisualShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)
    self.bottom_wall_id = self.pb.createMultiBody(mass, baseCollisionShapeIndex=wall_cid, baseVisualShapeIndex=wall_vid, 
        basePosition=[self.final_scene_size[0]/2,0, bottom_width/2],
        baseOrientation=ori)
    pb.changeDynamics(wall_cid,linkIndex=-1,lateralFriction=wall_fric_coeff)

    # Back
    half_extent = [width/2, self.final_scene_size[1]/2, height/2]
    wall_cid = self.pb.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent)
    wall_vid = self.pb.createVisualShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)
    self.pb.createMultiBody(mass, baseCollisionShapeIndex=wall_cid, baseVisualShapeIndex=wall_vid, 
        basePosition=[self.final_scene_size[0]+width/2.0 ,0,height/2.0 + bottom_width], #+ 0.25
        baseOrientation=ori)
    pb.changeDynamics(wall_cid,linkIndex=-1,lateralFriction=wall_fric_coeff)

    # Left
    half_extent = [self.final_scene_size[0]/2 + width , width/2, height/2] #+ 1.0
    wall_cid = self.pb.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent)
    wall_vid = self.pb.createVisualShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)
    self.pb.createMultiBody(mass, baseCollisionShapeIndex=wall_cid, baseVisualShapeIndex=wall_vid, 
        basePosition=[self.final_scene_size[0]/2,-self.final_scene_size[1]/2-width/2,height/2.0 + bottom_width],
        baseOrientation=ori)
    pb.changeDynamics(wall_cid,linkIndex=-1,lateralFriction=wall_fric_coeff)

    # Right
    half_extent = [self.final_scene_size[0]/2 + width , width/2, height/2] #+ 1.0
    wall_cid = self.pb.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent)
    wall_vid = self.pb.createVisualShape(shapeType=pybullet.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)
    self.pb.createMultiBody(mass, baseCollisionShapeIndex=wall_cid, baseVisualShapeIndex=wall_vid, 
        basePosition=[self.final_scene_size[0]/2,self.final_scene_size[1]/2+width/2,height/2.0 + bottom_width],
        baseOrientation=ori)
    pb.changeDynamics(wall_cid,linkIndex=-1,lateralFriction=wall_fric_coeff)

    self.scene_height = bottom_width

  def setGUICamera(self):
    camTargPos = {0.1, -0.28, 3.39}
    camDist = 4.5
    pitch = -89.9
    yaw = -89.9
    pb.resetDebugVisualizerCamera(cameraDistance=camDist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=camTargPos)







