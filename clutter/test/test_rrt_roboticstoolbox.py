
from roboticstoolbox import RRTPlanner
from spatialmath import Polygon2
from math import pi

# create polygonal obstacles
map = PolygonMap(workspace=[0, 10])
map.add([(5, 50), (5, 6), (6, 6), (6, 50)]) 
map.add([(5, 4), (5, -50), (6, -50), (6, 4)])

# create outline polygon for vehicle
l, w = 3, 1.5
vpolygon = Polygon2([(-l/2, w/2), (-l/2, -w/2), (l/2, -w/2), (l/2, w/2)])

# create vehicle model
vehicle = Bicycle(steer_max=1, L=2, polygon=vpolygon)

# create planner
rrt = RRTPlanner(map=map, vehicle=vehicle, npoints=50, seed=0)
# start and goal configuration
qs = (2, 8, -pi/2)
qg = (8, 2, -pi/2)

# plan path
rrt.plan(goal=qg)
path, status = rrt.query(start=qs)
print(path[:5,:])
print(status)