# File: object.py
# Author: Marion Lepert
# Description: Object class.

import numpy as np
import pkg_resources

from contact_point import ContactPoint


class Object():
    """
    Base class for pybullet objects
    """
    def __init__(self, pb, name, dim, pos=[0,0,0], ori=[0,0,0], scene_depth_for_obj=""):
        self.pb = pb
        self.name = name
        self.dim = dim
        self.scene_depth = scene_depth_for_obj

        self.pos = pos
        self.ori = self.pb.getQuaternionFromEuler(ori)

        self.create_object()
        self.mass = self.get_mass()
        
        if 'length' not in self.__dict__.keys():
            self.length, self.width, self.height = self.get_dimensions()


    def create_object(self):
        filename = self.name + str(self.dim) + "D.urdf"
        obj_urdf = pkg_resources.resource_filename('clutter-jamming',
                                                   'resources/'+filename)
        
        self.obj_id = self.pb.loadURDF(fileName=obj_urdf,
                                   basePosition=self.pos,
                                   baseOrientation=self.ori)

        self.get_num_joints()


    def get_num_joints(self): 
        """
        Return the number of joints in the robot, 
        excluding fixed joints. 
        """
        num_joints = self.pb.getNumJoints(self.obj_id)
        self.num_joints = 0
        for idx in range(num_joints): 
            joint_info = self.pb.getJointInfo(self.obj_id, idx)
            if joint_info[2] != self.pb.JOINT_FIXED: 
                self.num_joints += 1

        
    def destruct(self):
        self.pb.removeBody(self.obj_id)


    def neutralize_motors(self):
        """
        Needed to remove friction in prismatic joints. 
        See github issue for explanation:
        https://github.com/bulletphysics/bullet3/issues/2341 
        (Darren Levine's answer)
        """
        for joint_idx in range(-1, self.num_joints):
            self.pb.setJointMotorControl2(bodyUniqueId=self.obj_id,
                                          jointIndex=max(0,joint_idx),
                                          controlMode=self.pb.VELOCITY_CONTROL,
                                          force=0)
        
    def remove_friction(self):
        """
        Remove all friction from the surface of the object.
        """
        for joint_idx in range(-1, self.num_joints):
            self.pb.changeDynamics(bodyUniqueId=self.obj_id,
                                   linkIndex=joint_idx,
                                   lateralFriction=0,
                                   spinningFriction=0,
                                   rollingFriction=0,
                                   linearDamping=0,
                                   angularDamping=0)


    def add_friction(self):
        """
        Add lateral friction to the surface of the object.
        """
        for joint_idx in range(-1, self.num_joints):
            self.pb.changeDynamics(bodyUniqueId=self.obj_id,
                                   linkIndex=joint_idx,
                                   lateralFriction=0.85) # used to be 0.5


    def enable_collisions(self, external_ids=None):
        self.toggle_collisions(enable_status=True, external_ids=external_ids)


    def disable_collisions(self, external_ids=None):
        self.toggle_collisions(enable_status=False, external_ids=external_ids)


    def toggle_collisions(self, enable_status, external_ids=None):
        if external_ids is None:
            for idx in range(self.pb.getNumBodies()):
                body_id = self.pb.getBodyUniqueId(idx)
                if body_id != self.obj_id:
                    ext_num_joints = self.pb.getNumJoints(body_id)
                    self.pb.setCollisionFilterPair(bodyUniqueIdA=self.obj_id,
                                                   bodyUniqueIdB=body_id,
                                                   linkIndexA=self.num_joints-1,
                                                   linkIndexB=ext_num_joints-1,
                                                   enableCollision=enable_status)
        else:
            for body_id in external_ids:
                ext_num_joints = self.pb.getNumJoints(body_id)
                self.pb.setCollisionFilterPair(bodyUniqueIdA=self.obj_id,
                                                   bodyUniqueIdB=body_id,
                                                   linkIndexA=self.num_joints-1,
                                                   linkIndexB=ext_num_joints-1,
                                                   enableCollision=enable_status)
        self.pb.stepSimulation()
            

    def is_in_contact(self, pos=None, ori=None, external_ids=None,
                        require_all=False):
        """
        If external_ids is not None: 
             - If require_all is True: return True if the object is in 
               collision with all elements in external_ids 
             - If require_all is False: return True if the object is in 
               collision with any of the elements in external_ids
        If list external_ids is None, return true if the object in 
        position pos is in contact (at least touching) with any element. 
        """
        if pos is not None:
            self.reset_position(pos, ori)

        if external_ids is None:
            cps = self.pb.getContactPoints(bodyA=self.obj_id)
            return len(cps) > 0
        else:
            contacts = []
            for external_id in external_ids:
                cps = self.pb.getContactPoints(bodyA=self.obj_id,
                                               bodyB=external_id)

                if len(cps) == 0:
                    contacts.append(0)
                else:
                    contacts.append(1)

            contacts = np.array(contacts)
            if require_all:
                return contacts.all()
            else:
                return contacts.any()


    def is_in_collision(self, pos=None, ori=None, external_ids=None,
                        require_all=False, dist_threshold=-1e-5):
        """
        If external_ids is not None: 
             - If require_all is True: return True if the object is in 
               collision with all elements in external_ids 
             - If require_all is False: return True if the object is in 
               collision with any of the elements in external_ids
        If list external_ids is None, return true if the object in 
        position pos is in collision (with penetration) with any element. 
        """
        if pos is not None:
            self.reset_position(pos, ori)

        if external_ids is None:
            cps = self.pb.getContactPoints(bodyA=self.obj_id)
            for cp_data in cps:
                cp = ContactPoint(cp_data)
                if cp.contact_distance < dist_threshold:
                    return True
            return False
        else:
            contacts = []
            for external_id in external_ids:
                cps = self.pb.getContactPoints(bodyA=self.obj_id,
                                               bodyB=external_id)

                made_contact = False
                for cp_data in cps:
                    cp = ContactPoint(cp_data)
                    if cp.contact_distance < dist_threshold:
                        made_contact = True

                if made_contact:
                    contacts.append(1)
                else:
                    contacts.append(0)

            contacts = np.array(contacts)
            if require_all:
                return contacts.all()
            else:
                return contacts.any()


    def get_contact_ids(self):
        """
        Return list of body ids that the object is in contact with.
        """
        cps = self.get_contact_points()
        if cps is None:
            return []
        return set([cp.bodyB_id for cp in cps])


    def get_contact_points(self, external_id=None, link_index=None):
        """
        Return a list of ContactPoints between the object and the external
        object with id, external_id.
        """
        if external_id is None:
            if link_index is None: 
                cps = self.pb.getContactPoints(bodyA=self.obj_id)
            else: 
                cps = self.pb.getContactPoints(bodyA=self.obj_id, linkIndexA=link_index)
        else:
            if link_index is None: 
                cps = self.pb.getContactPoints(bodyA=self.obj_id, bodyB=external_id)
            else: 
                cps = self.pb.getContactPoints(bodyA=self.obj_id, bodyB=external_id, linkIndexA=link_index)

        if len(cps) == 0: return None
        return [ContactPoint(cp_data) for cp_data in cps]


    def get_contact_normal(self, external_id=None, exclusion_ids=None, link_index=None):
        """
        Return the average normal vector of the surface of the external object 
        with id, external_id, that is closest to the object.
        """
        list_cps = self.get_contact_points(external_id, link_index)
        if list_cps is None: return None
        contact_normals = []
        for cp in list_cps:
            if cp.bodyB_id not in exclusion_ids: 
                contact_normals.append(np.array(cp.contact_normal_on_B))
        contact_normals=np.array(contact_normals)
        return np.mean(contact_normals, axis=0)

    
    def get_resultant_contact_normals(self, list_external_id):
        force = 0
        for idx in list_external_id:
            contact_normal = self.get_contact_normal(idx)
            if contact_normal is not None:
                force += contact_normal
        if np.linalg.norm(force) > 0:
            return force
        else:
            return None

    def get_summed_contact_force_mags(self, external_id=None, exclusion_ids=None, link_index=None):
        list_cps = self.get_contact_points(external_id, link_index)
        if list_cps is None: return 0
        contact_force_mags = []
        for cp in list_cps:
            if cp.bodyB_id not in exclusion_ids: 
                contact_force_mags.append(cp.normal_force)
        contact_force_mags = np.array(contact_force_mags)
        return np.sum(contact_force_mags)
        

    def get_contact_force_mag(self, external_id=None, exclusion_ids=None, link_index=None):
        """
        Return the average magnitude of the external force applied on the
        object by the external object with id, external_id.
        """
        list_cps = self.get_contact_points(external_id, link_index)
        if list_cps is None: return 0
        contact_forces = []
        for cp in list_cps:
            if cp.bodyB_id not in exclusion_ids: 
                contact_forces.append(cp.normal_force)
        contact_forces = np.array(contact_forces)
        return np.mean(contact_forces)


    def get_external_force_torque(self):
        """
        Return the net external force and torque applied on the 
        object at its center. 
        """
        obj_center = self.get_position()

        list_cps = self.get_contact_points(external_id=None)
        total_force = np.array([0.0,0.0,0.0])
        total_torque = 0.0

        if list_cps is not None: 
            for cp in list_cps:
                force_mag = cp.normal_force
                force_dir = -np.array(cp.contact_normal_on_B)
                force_vec = force_mag * force_dir 
                total_force += force_vec
                force_pos = np.array(cp.pos_on_A) - obj_center
                force_pos[2] = 0
                total_torque += np.cross(force_pos[0:2], force_vec[0:2])

        return total_force, total_torque


    def get_distance(self, external_id):
        """
        Return the distance between the closest points on the 
        object and the object with id, external_id. 
        """
        max_dist = 1000
        cps = self.pb.getClosestPoints(self.obj_id, external_id, max_dist)
        dist = np.inf
        if len(cps) > 0:
            for cp_data in cps:
                cp = ContactPoint(cp_data)
                dist = min(dist, cp.contact_distance)
            return dist
        else:
            return max_dist 
                
                    
    def get_mass(self):
        """
        Return object mass in kg.
        """
        mass = 0
        for joint_idx in range(-1,self.num_joints):
            dyn_info = self.pb.getDynamicsInfo(self.obj_id, joint_idx)
            mass += dyn_info[0]
        return mass


    def get_dimensions(self):
        """
        Return the length, width, and height of the dimensions
        of the bounding box around the object.
        """
        length, width, height = 0, 0, 0
        for joint_idx in range(-1,self.num_joints):
            aabb = self.pb.getAABB(self.obj_id, joint_idx)
            length = max(length, abs(aabb[1][0]-aabb[0][0]))
            width = max(width, abs(aabb[1][1]-aabb[0][1]))
            height = max(height, abs(aabb[1][2]-aabb[0][2]))
        return length, width, height
        