import os
import numpy as np
import sys
sys.path.append('/home/zhaoming/Documents/open_robot/raisim_build/lib')
import raisimpy as raisim
import time
from scipy.spatial.transform import Rotation as R


raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
anymal_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/anymal/urdf/anymal.urdf"
laikago_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/laikago/laikago.urdf"
atlas_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/atlas/robot.urdf"
monkey_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/monkey/monkey.obj"
solo8_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/solo8_URDF_v6/solo8.urdf"
virtual_human_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/virtual_human.urdf"
walker_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/humanoid.urdf"
dummy_inertia = np.zeros([3, 3])
np.fill_diagonal(dummy_inertia, 0.1)

world = raisim.World()
world.setTimeStep(0.001)
server = raisim.RaisimServer(world)
ground = world.addGround()

anymal = world.addArticulatedSystem(anymal_urdf_file)
anymal.setName("anymal")
anymal_nominal_joint_config = np.array([0, -1.5, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8,
                                        -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8])
anymal.setGeneralizedCoordinate(anymal_nominal_joint_config)
anymal.setPdGains(200*np.ones([18]), np.ones([18]))
anymal.setPdTarget(anymal_nominal_joint_config, np.zeros([18]))

laikago = world.addArticulatedSystem(laikago_urdf_file)
laikago.setName("laikago")
laikago_nominal_joint_config = np.array([0, 1.5, 0.48, 1, 0.0, 0.0, 0.0, 0.0, 0.5, -1, 0, 0.5, -1,
                                         0.00, 0.5, -1, 0, 0.5, -0.7])
laikago.setGeneralizedCoordinate(laikago_nominal_joint_config)
laikago.setPdGains(200*np.ones([18]), np.ones([18]))
laikago.setPdTarget(laikago_nominal_joint_config, np.zeros([18]))

atlas = world.addArticulatedSystem(atlas_urdf_file)
atlas.setName("atlas")
atlas_nominal_joint_config = np.zeros(atlas.getGeneralizedCoordinateDim())
atlas_nominal_joint_config[2] = 1.5
atlas_nominal_joint_config[3] = 1
atlas.setGeneralizedCoordinate(atlas_nominal_joint_config)

server.launchServer(8080)

for i in range(5):
    for j in range(5):
        object_type = (i + j*6) % 5

        if object_type == 0:
            obj = world.addMesh(monkey_file, 5.0, dummy_inertia, np.array([0, 0, 0]), 0.3)
            obj.setAppearance("blue")
        elif object_type == 1:
            obj = world.addCylinder(0.2, 0.3, 2.0)
            obj.setAppearance("red")
        elif object_type == 2:
            obj = world.addCapsule(0.2, 0.3, 2.0)
            obj.setAppearance("green")
        elif object_type == 3:
            obj = world.addBox(0.4, 0.4, 0.4, 2.0)
            obj.setAppearance("purple")
        else:
            obj = world.addSphere(0.3, 2.0)
            obj.setAppearance("orange")

        obj.setPosition(i-2.5, j-2.5, 5)

time.sleep(2)
world.integrate1()

### get dynamic properties
# mass matrix
# mass_matrix = anymal.getMassMatrix()
# non-linear term (gravity+coriolis)
non_linearities = anymal.getNonlinearities([0,0,-9.81])
# Jacobians
# jaco_foot_lh_linear = anymal.getDenseFrameJacobian("LF_ADAPTER_TO_FOOT")
# jaco_foot_lh_angular = anymal.getDenseFrameRotationalJacobian("LF_ADAPTER_TO_FOOT")

reference = np.zeros(walker.getGeneralizedCoordinateDim())
# reference[2] = 2
# reference[3] = 1
# reference[5] = 0
t = 0
period = 30

import pandas as pd
from axis_to_quaternion import *

with open('030004_001_20_T_ST_0100_2_JM_Player2/Input.txt') as f:
	data = f.readlines()
# with open('Data/4_CR_JUGGLE_1St001_01_Player1_Standard/Input.txt') as f:
# 	data = f.readlines()

def coordinate_transform(q):
	q[0:4] = q[[0, 3, 1, 2]]
	q[1] = -q[1]
	q[3] = -q[3]
	return q

def rotate_translation(x):
	x[0:3] = x[[2, 0, 1]]
	# x[0] = -x[0]
	x[1] = -x[1]
	return x

starting_frame = 414
num_frame = 30
record_data = np.zeros([num_frame, 46])

while True:
	for i in range(starting_frame, starting_frame+num_frame):

		frame =  i#int(i / 2)

		from scipy.spatial.transform import Rotation as R

		frame_data = np.asarray(data[frame].split(' ')).astype(float)

		# print(frame_data)
		reference[0:15] = frame_data[0:15]
		reference[15:19] = frame_data[15:19]
		reference[19] = frame_data[19]
		reference[20:24] = frame_data[24:28]
		reference[24] = frame_data[28]
		reference[25:43] = frame_data[33:51]

		reference[19] = -reference[19] / 180 * 3.1415
		reference[24] = -reference[24] / 180 * 3.1415
		reference[29] = -reference[29] / 180 * 3.1415
		reference[38] = -reference[38] / 180 * 3.1415
		reference[3:7] = coordinate_transform(reference[3:7])
		reference[15:19] = coordinate_transform(reference[15:19])
		reference[20:24] = coordinate_transform(reference[20:24])
		reference[25:29] = coordinate_transform(reference[25:29])
		reference[30:34] = coordinate_transform(reference[30:34])
		reference[34:38] = coordinate_transform(reference[34:38])
		reference[39:43] = coordinate_transform(reference[39:43])
		reference[0:3] = rotate_translation(reference[0:3])
		
		#rotate the frame of robot to align with global frame
		if i == starting_frame:
			base_rot = R.from_quat([0.707, 0, 0, 0.707]).as_matrix()
			r = R.from_quat(reference[[4,5,6,3]])
			rotation = base_rot.dot(r.as_matrix().T)
			translation = rotation.dot(reference[0:3]) - np.array([0, 0, 0.9])
		r = R.from_quat(reference[[4,5,6,3]])
		reference[3:7] = R.from_matrix(rotation.dot(r.as_matrix())).as_quat()[[3,0,1,2]]
		reference[0:3] = rotation.dot(reference[0:3]) - translation
		ball_reference = np.array([frame_data[53], -frame_data[51], frame_data[52]])
		ball_reference = rotation.dot(ball_reference) - translation

		walker.setState(reference, np.zeros([walker.getDOF()]))
		walker.setPdTarget(reference, np.zeros([walker.getDOF()]))
		obj.setPosition(ball_reference[0], ball_reference[1], ball_reference[2])
		obj.setVelocity(0, 0, 0, 0, 0, 0)
		# print(reference[0:3], frame_data[51], frame_data[53], frame_data[52])
		print(i)
		world.integrate()
		import time; time.sleep(0.04)
		t += 1
		if t > period:
			t = 0
		record_data[frame-starting_frame, 0:43] = reference[:].copy()
		record_data[frame-starting_frame, 43:46] = ball_reference[:].copy()
		print(frame, ball_reference[:])
	#print(record_data)
	np.savetxt('data.txt', record_data, fmt='%f')

for i in range(500000):
    server.integrateWorldThreadSafe()

server.killServer()
