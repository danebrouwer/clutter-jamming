# File: contact_point.py
# Author: Marion Lepert
# Description: ContactPoint class.


class ContactPoint():
    def __init__(self, data):
        self.bodyA_id = data[1]
        self.bodyB_id = data[2]
        self.linkA = data[3]
        self.linkB = data[4]
        self.pos_on_A = data[5]
        self.pos_on_B = data[6]
        self.contact_normal_on_B = data[7]
        self.contact_distance = data[8]
        self.normal_force = data[9]
        self.lateral_friction_mag_1 = data[10]
        self.lateral_friction_dir_1 = data[11]
        self.lateral_friction_mag_2 = data[12]
        self.lateral_friction_dir_2 = data[13]