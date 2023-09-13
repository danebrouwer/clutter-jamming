import numpy as np
import matplotlib.pyplot as plt
from pybullet_planning import connect, create_test_environment, draw_path, create_sphere, set_pose, plan_cartesian_path, plot_pose_array

# Connect to the PyBullet simulator
connect(use_gui=True)

# Define the start and end points
start = [0, 0, 0]
goal = [10, 10, 0]

# Define the circular obstacles in the scene
obstacles = [
    create_sphere(radius=1.5, position=[5, 5, 0]),
    create_sphere(radius=1, position=[8, 2, 0])
]

# Create an environment with the start, goal, and obstacles
env = create_test_environment(obstacles=obstacles, robot_base_pose=start, goal_poses=[goal])

# Plan a path to the goal
path = plan_cartesian_path(env, end_link_name='ee_link', rrt=True, epsilon=0.05)
draw_path(path)

# Plot the resulting path
path_poses = np.array([set_pose(p, [0, 0, 0, 1])[:2] for p in path])
plt.plot(path_poses[:, 0], path_poses[:, 1], '-o')
for ob in obstacles:
    plt.gca().add_artist(plt.Circle(ob[:2], ob[3], color='r', alpha=0.5))
plt.show()
