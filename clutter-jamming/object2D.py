# File: object2D.py
# Author: Marion Lepert
# Description: Object2D class.

import numpy as np
from object import *

class Object2D(Object):
    """
    Pybullet object that can only move along x-y plane and rotate
    about the z axis.
    """
    def __init__(self, pb, name, pos=[0,0,0], ori=[0,0,0], scene_depth=""):
        self.dim = 2
        self.scene_depth = scene_depth
        super().__init__(pb, name, self.dim, pos, ori, scene_depth_for_obj=self.scene_depth)


    def reset_position(self, pos, theta):
        """
        Moves the object to position, pos, and orientation, theta, about
        the z-axis.
        """
        for joint_idx in range(2):
            self.pb.resetJointState(bodyUniqueId=self.obj_id,
                                    jointIndex=joint_idx,
                                    targetValue=pos[joint_idx],
                                    targetVelocity=0)
            
        self.pb.resetJointState(bodyUniqueId=self.obj_id,
                                jointIndex=2,
                                targetValue=theta,
                                targetVelocity=0)

        self.pb.stepSimulation()


    def reset_orientation(self, theta):
        self.pb.resetJointState(bodyUniqueId=self.obj_id,
                                jointIndex=2,
                                targetValue=theta,
                                targetVelocity=0)

        self.pb.stepSimulation()


    def reset_base_height(self, height):
        self.pb.resetBasePositionAndOrientation(bodyUniqueId=self.obj_id,
                                                posObj=[0,0,height],
                                                ornObj=[0,0,0,1])


    def apply_velocity(self, velocity, max_force=None):
        """
        Sets commanded object velocity to velocity.
        CAREFUL: does not step simulation
        """
        if max_force is None:
            self.pb.setJointMotorControlArray(bodyUniqueId=self.obj_id,
                                        jointIndices=[0,1],
                                        controlMode=self.pb.VELOCITY_CONTROL,
                                        targetVelocities=velocity[0:2])
        else:
            self.pb.setJointMotorControlArray(bodyUniqueId=self.obj_id,
                                        jointIndices=[0,1],
                                        controlMode=self.pb.VELOCITY_CONTROL,
                                        targetVelocities=velocity[0:2],
                                        forces=[max_force, max_force])

            
    def apply_ang_velocity(self, ang_velocity, max_torque=None):
        """ 
        Sets commanded object angular velocity on object to ang_velocity.
        CAREFUL: does not step simulation
        """
        if max_torque is None:
            self.pb.setJointMotorControl2(bodyUniqueId=self.obj_id,
                                        jointIndex=2,
                                        controlMode=self.pb.VELOCITY_CONTROL,
                                        targetVelocity=ang_velocity)
        else:
            self.pb.setJointMotorControl2(bodyUniqueId=self.obj_id,
                                        jointIndex=2,
                                        controlMode=self.pb.VELOCITY_CONTROL,
                                        targetVelocity=ang_velocity,
                                        force=max_torque) 

        
    def apply_force(self, force):
        """
        Sets commanded external force on object to force.
        First, set all motor forces to zero
        See github issue for explanation:
        https://github.com/bulletphysics/bullet3/issues/2341 
        (Darren Levine's answer)
        CAREFUL: does not step simulation
        """
        self.pb.setJointMotorControlArray(bodyUniqueId=self.obj_id,
                                        jointIndices=[0,1],
                                        controlMode=self.pb.VELOCITY_CONTROL,
                                        forces=[0,0])
        self.pb.setJointMotorControlArray(bodyUniqueId=self.obj_id,
                                        jointIndices=[0,1],
                                        controlMode=self.pb.TORQUE_CONTROL,
                                        forces=force[0:2])

        
    def apply_torque(self, torque):
        """
        Sets commanded external torque on object to torque.
        First, set all motor forces to zero
        See github issue for explanation:
        https://github.com/bulletphysics/bullet3/issues/2341 
        (Darren Levine's answer)
        CAREFUL: does not step simulation
        """
        self.pb.setJointMotorControl2(bodyUniqueId=self.obj_id,
                                      jointIndex=2,
                                      controlMode=self.pb.VELOCITY_CONTROL,
                                      force=0)
        self.pb.setJointMotorControl2(bodyUniqueId=self.obj_id,
                                      jointIndex=2,
                                      controlMode=self.pb.TORQUE_CONTROL,
                                      force=torque)

    def stop(self):
        """
        Set the velocity of the object to zero. 
        """
        self.apply_velocity(velocity=[0,0,0])
        self.apply_ang_velocity(ang_velocity=0)
        self.pb.stepSimulation()
        

    def get_position(self):
        """
        Returns object x,y,z position in the world frame.
        """
        base_pos = np.array(self.pb.getBasePositionAndOrientation(self.obj_id)[0])
        joint_pos_x, _, _, _ = self.pb.getJointState(self.obj_id, 0)
        joint_pos_y, _, _, _ = self.pb.getJointState(self.obj_id, 1)
        return base_pos + np.array([joint_pos_x, joint_pos_y, 0])        

    
    def get_orientation(self):
        """
        Returns theta value of object orientation around the world z axis.
        """
        joint_theta, _, _, _ = self.pb.getJointState(self.obj_id, 2)
        return joint_theta


    def get_linear_velocity(self):
        """
        Return the x, y, z linear velocity of the object in the world frame.
        """
        _, joint_vel_x, _, _ = self.pb.getJointState(self.obj_id, 0)
        _, joint_vel_y, _, _ = self.pb.getJointState(self.obj_id, 1)
        return np.array([joint_vel_x, joint_vel_y, 0])


    def get_angular_velocity(self):
        """
        Return the angular velocity of the object about the z axis.
        """
        _, joint_vel_theta, _, _ = self.pb.getJointState(self.obj_id, 2)
        return joint_vel_theta


    def get_speed(self):
        linear_vel = self.get_linear_velocity()
        angular_vel = self.get_angular_velocity()
        speed = np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel)