import pybullet as pb 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


# Function to plot contact force vectors
def plot_contact_forces(obj, lifetime):
    """
    Display contact force vectors on body. 
    """
    
    list_cps = obj.get_contact_points(external_id=None)

    if list_cps is not None: 
        for cp in list_cps:
            Fn_mag = cp.normal_force
            Fn_dir = np.array(cp.contact_normal_on_B)
            Fn_vec = Fn_mag * Fn_dir 
            contact_pos = np.array(cp.pos_on_A)
            pb.addUserDebugLine(contact_pos, contact_pos + 0.1*Fn_vec, lineColorRGB=[1,0,0], lineWidth=3.0, lifeTime=lifetime)
            Fs1_mag = cp.lateral_friction_mag_1
            Fs1_dir = np.array(cp.lateral_friction_dir_1)
            Fs1_vec = Fs1_mag * Fs1_dir 
            pb.addUserDebugLine(contact_pos, contact_pos + 0.1*Fs1_vec, lineColorRGB=[0,0,1], lineWidth=3.0, lifeTime=lifetime)


def store_contact_locations(obj, contact_map, force_map):
    """
    Store contact locations to a contact map to plot later. 
    """

    list_cps = obj.get_contact_points(external_id=None)

    if list_cps is not None: 
        for cp in list_cps:
            Fn_mag = cp.normal_force
            Fn_dir = np.array(cp.contact_normal_on_B)
            Fn_vec = Fn_mag * Fn_dir
            
            contact_pos = np.array(cp.pos_on_A)
            
            if Fn_mag >= 1.0:
                if contact_map is not None:
                    contact_map = np.vstack((contact_map,[contact_pos[0],contact_pos[1]]))
                else:
                    contact_map = np.array([contact_pos[0],contact_pos[1]])
                Fs1_mag = cp.lateral_friction_mag_1
                Fs1_dir = np.array(cp.lateral_friction_dir_1)
                Fs1_vec = Fs1_mag * Fs1_dir 
                F_vec = Fn_vec + Fs1_vec
                if force_map is not None:
                    force_map = np.vstack((force_map,[F_vec[0],F_vec[1]]))
                else:
                    force_map = np.array(([F_vec[0],F_vec[1]]))

    return contact_map, force_map


def plot_contact_map(contact_map, force_map):
    if contact_map is not None:
        x_contact = contact_map[:,0]
        y_contact = contact_map[:,1]
        x_force = -force_map[:,0] # Force from paddle to objects (more interpretable)
        y_force = -force_map[:,1]
        X, Y = -y_contact, x_contact #Remap to simulation frame
        U, V = -y_force, x_force

        X_clust = np.vstack((X,Y)).T
        F_clust = np.vstack((U,V)).T
        clust_model = OPTICS(min_samples=10, cluster_method = "dbscan", max_eps = 0.1, min_cluster_size=5) 
        clust_model.fit(X_clust)
        # labels = clust_model.labels_[clust_model.ordering_]

        # colors = ["g", "r", "b", "y", "c", "m", "tab:orange", "tab:brown"]
        num_clusters = np.max(clust_model.labels_) + 1
        for i in range(num_clusters): #klass, color in zip(range(0, 8), colors):
            Xk = X_clust[clust_model.labels_ == i]
            Fk = F_clust[clust_model.labels_ == i]
            M = len(Xk[:, 0])
            if M > 0:
                color = "C" + str(i)
                # print(color)
                plt.quiver(Xk[:, 0], Xk[:, 1], Fk[:, 0], Fk[:, 1], color=color, angles='xy',scale=100, width=0.0025*4, alpha=np.logspace(0.0, 1.0, M)/10) # 

        # pdb.set_trace()
        # print(np.max(clust_model.labels_))
        # print(np.min(clust_model.labels_))
        
        plt.quiver(X_clust[clust_model.labels_ == -1, 0], X_clust[clust_model.labels_ == -1, 1], F_clust[clust_model.labels_ == -1, 0], \
        F_clust[clust_model.labels_ == -1, 1], color="k", angles='xy',scale=100, width=0.0025*4, alpha=0.1) #
        plt.title("Clustered Contact Force Quiver")

        # plt.scatter(X,Y)
        # plt.quiver(X,Y,U,V,color=[0,0,1],angles='xy',scale=100) 
        plt.ylim(-1,5)
        plt.xlim(-2,2)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()


def get_impeding_contact_loc(obj):
    """
    Return the location of the contact with highest force
    """
    
    contact_pos_max = np.array([0.0,0.0])
    F_net_max = 0.0

    list_cps = obj.get_contact_points(external_id=None)

    if list_cps is not None: 
        for cp in list_cps:
            Fn_mag = cp.normal_force
            contact_pos = np.array(cp.pos_on_A)
            Fs1_mag = cp.lateral_friction_mag_1
            F_net = np.linalg.norm([Fn_mag, Fs1_mag])
            if F_net > F_net_max:
                F_net_max = F_net
                contact_pos_max = contact_pos
    return contact_pos_max

def contact_at_tip(obj, tip_pos):
    list_cps = obj.get_contact_points(external_id=None)

    if list_cps is not None: 
        for cp in list_cps:
            Fn_mag = cp.normal_force
            # print(Fn_mag)
            contact_pos = np.array(cp.pos_on_A)
            # print(contact_pos[0:2])
            # print(tip_pos[0:2])
            # pos_diff = contact_pos[0:1] - tip_pos[0:1]
            # Fs1_mag = cp.lateral_friction_mag_1
            # F_net = np.linalg.norm([Fn_mag, Fs1_mag])
            # print("force: ", np.round(Fn_mag,4), 
                #   "     distance: ", np.round(np.linalg.norm((contact_pos[0:2] - tip_pos[0:2])),4))
            if Fn_mag > 0.05 and Fn_mag < 5.0:

                if np.linalg.norm((contact_pos[0:2] - tip_pos[0:2])) < 0.25:
                    # print("force: ", Fn_mag)
                    return True
    return False    



def contact_force_exceeds(obj,force_thresh):
    """
    Return true if any contact exceeds force threshold
    """

    list_cps = obj.get_contact_points(external_id=None)

    if list_cps is not None: 
        for cp in list_cps:
            Fn_mag = cp.normal_force
            # contact_pos = np.array(cp.pos_on_A)
            Fs1_mag = cp.lateral_friction_mag_1
            F_net = np.linalg.norm([Fn_mag, Fs1_mag])
            if F_net > force_thresh:
                return True
    else:
        return False
    

def save_results(log_path, results_dict):
    print(results_dict)
    df = pd.DataFrame(results_dict)
    df.to_pickle(log_path)