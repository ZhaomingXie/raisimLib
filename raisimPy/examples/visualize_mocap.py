import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sklearn.neighbors import KDTree
import sys
sys.path.append('/home/zhaoming/Documents/open_robot/raisim_build/lib')
import raisimpy as raisim

from fairmotion.data import bvh
# BVH_FILENAME = "../../../../../Downloads/MotionCapture/Mocap/Loco/RunSideBack1.bvh"
# motion = bvh.load(BVH_FILENAME)

world = raisim.World()
world.setTimeStep(0.001)
server = raisim.RaisimServer(world)
ground = world.addGround()
virtual_human_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/virtual_human.urdf"
virtual_human = world.addArticulatedSystem(virtual_human_urdf_file)
virtual_human.setName("virtual_human")
virtual_human_nominal_joint_config = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
 0., 0., 0.,])

# visBox = server.addVisualBox("v_box", 1, 1, 1, 1, 1, 1, 1)
# visAngularVelBox = server.addVisualBox("v_box2", 1, 1, 1, 1, 1, 0, 1)

def coordinate_transform(q):
	# q[0:4] = q[[0, 3, 1, 2]]
	# q[1] = -q[1]
	# q[3] = -q[3]
	return q

def process_mocap(mocap_file, phase, mirror=False):
	motion = bvh.load(mocap_file)
	matrix = motion.to_matrix(local=False)[:, :, :, :]
	local_matrix = motion.to_matrix(local=True)[:, :, : ,:]

	if mirror: 
		matrix[:, :, 0, 3] *= -1
		matrix[:, :, 0, 1] *= -1
		matrix[:, :, 0, 2] *= -1
		matrix[:, :, 1, 0] *= -1
		matrix[:, :, 2, 0] *= -1

		local_matrix[:, [7,8,9,10, 11,12,13,14, 15,16,17,18, 19,20,21,22], :, :] = local_matrix[:, [11,12,13,14, 7,8,9,10, 19,20,21,22, 15,16,17,18], :, :]

		local_matrix[:, :, 0, 3] *= -1
		local_matrix[:, :, 0, 1] *= -1
		local_matrix[:, :, 0, 2] *= -1
		local_matrix[:, :, 1, 0] *= -1
		local_matrix[:, :, 2, 0] *= -1

	pos = matrix[:, :, :3, 3]
	num_frames = matrix.shape[0]

	#generate virtual root pose
	root_pos = np.zeros((num_frames, 3))
	root_rot = np.zeros((num_frames, 3, 3))
	root_pos[:, 0] = pos[:, 0, 0]
	root_pos[:, 2] = pos[:, 0, 2]
	root_rot[:, :, 0] = np.cross((pos[:, 15, :] - pos[:, 19, :]), (pos[:, 0, :] - pos[:, 1, :]))
	root_rot[:, 1, 0] = 0
	norms = np.apply_along_axis(np.linalg.norm, 1, root_rot[:, :, 0])
	root_rot[:, :, 0] /= norms[:, np.newaxis]
	root_rot[:, :, 1] = np.array([0, 1, 0])
	root_rot[:, :, 2] = np.cross(root_rot[:, :, 0], root_rot[:, :, 1])

	#root velocity
	root_linear_vel = np.zeros((num_frames, 3))
	root_angular_vel = np.zeros((num_frames, 3, 3))
	root_angular_vel[-1, :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	for i in range(num_frames-1):
		root_linear_vel[i] = root_rot[i, :, :].T.dot(root_pos[i+1, :] - root_pos[i, :])
		root_angular_vel[i, :, :] = root_rot[i+1, :, :].dot(root_rot[i, :, :].T)

	#generate bone data w.r.t virtual root
	store_data = np.zeros((num_frames, 23, 3))
	bone_velocity = np.zeros((num_frames, 23, 3))

	#bone position w.r.t the root
	for i in range(23):
		A = np.transpose(root_rot[:, : :], axes=[0, 2, 1])
		x = pos[:, i, :] - root_pos
		# store_data[:, i, :] = np.einsum('ijk, ij->k', A, x)
		store_data[:, i, :] = np.matmul(A, x[:, :, None]).squeeze()
	#bone velocity
	for i in range(num_frames-1):
		for j in range(23):
			bone_velocity[i, j, :] = store_data[i+1, j, :] - store_data[i, j, :]

	#root bone orientation
	bone_orientation = np.zeros((num_frames, 3, 3))
	for i in range(num_frames-1):
		bone_orientation[i, :, :] = root_rot[i, :, :].T.dot(matrix[i, 0, 0:3, 0:3])

	#kd tree features
	num_matched_frames = 10
	frames = []
	num_phase_matched_frames = 1
	phase_frames = [0, 1, 20]
	for i in range(num_matched_frames):
		frames.append(int((1+i) * 60 / num_matched_frames - 1))
	features = np.zeros((num_frames - 60, num_matched_frames * 4 + 6 * num_phase_matched_frames))
	for i in range(num_frames - 60):
		root_pos_feature = np.zeros(3)
		for j in range(num_matched_frames):
			features[i, j*4:j*4+2] = root_rot[i, :, :].T.dot(root_pos[frames[j] + i] - root_pos[i])[[0, 2]]
			features[i, j*4+2:j*4+4] = root_rot[i+frames[j], :, :].dot(root_rot[i, :, :].T).dot(np.array([1, 0, 0]))[[0, 2]]
			#features[i, j*9+6:j*9+9] = root_angular_vel[i+j, :, 1].copy()
		# features[i, 3 * 4: 3 * 4 + 3] = bone_velocity[i, 0, :].flatten().copy()
		# features[i, 3 * 4 + 3: 3 * 4 + 6] = store_data[i, 17, :].flatten().copy()
		# features[i, 3 * 4 + 6: 3 * 4 + 9] = store_data[i, 21, :].flatten().copy()
		# features[i, 3 * 4 + 9: 3 * 4 + 12] = bone_velocity[i, 17, :].flatten().copy()
		# features[i, 3 * 4 + 12: 3 * 4 + 15] = bone_velocity[i, 21, :].flatten().copy()
		for j in range(num_phase_matched_frames):
			features[i, num_matched_frames * 4 + 6 * j: num_matched_frames * 4 + 6 * j + 6] = phase[i+phase_frames[j], 0:6].copy()

	#build query data of shape (num_frames-60) * 60 * data_shape
	query_root_linear_vel = np.zeros((num_frames - 60, 60, 3))
	for i in range(num_frames-60):
		for j in range(60):
			query_root_linear_vel[i, j, :] = root_linear_vel[i+j, :].copy()

	query_root_angular_vel = np.zeros((num_frames-60, 60, 3, 3))
	for i in range(num_frames-60):
		for j in range(60):
			query_root_angular_vel[i, j, :, :] = root_angular_vel[i+j, :, :].copy()

	query_store_data = np.zeros((num_frames-60, 60, 3))
	for i in range(num_frames-60):
		for j in range(60):
			query_store_data[i, j, :] = store_data[i+j, 0, :].copy()

	query_local_matrix = np.zeros((num_frames-60, 60, 23, 4, 4))
	for i in range(num_frames-60):
		for j in range(60):
			query_local_matrix[i, j, :, :, :] = local_matrix[i+j, :, :, :].copy()

	query_bone_orientation = np.zeros((num_frames-60, 60, 3, 3))
	for i in range(num_frames - 60):
		for j in range(60):
			query_bone_orientation[i, j, :, :] = bone_orientation[i+j, :, :].copy()

	query_phase = np.zeros((num_frames-60, 60, 9))
	for i in range(num_frames - 60):
		for j in range(60):
			query_phase[i, j, :] = phase[i+j, :].copy()

	return root_pos, root_rot, query_root_linear_vel, query_root_angular_vel, query_store_data, bone_velocity, query_local_matrix, features, query_bone_orientation, query_phase


def set_human_pose(index, index_offset, hip_pose=None, hip_rot=None, previous_reference=None, weight=None):
	reference =  np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0.,])
	# base_rot = R.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()
	base_rot = R.from_quat([0.707, 0.0, 0.0, 0.707]).as_matrix()
		#r = R.from_matrix(matrix[j, 0, 0:3, 0:3])
		#rotation = base_rot.dot(r.as_matrix().T)
		#translation = r.as_matrix().T.dot(matrix[j, 0, 0:3, 3])

	#hip rot
	if hip_pose is None:
		reference[0:3] = base_rot.dot(matrix[j, 0, 0:3, 3])/100#(r.as_matrix().dot(matrix[j, 0, 0:3, 3]) - translation)[[0, 2, 1]] / 100 + np.array([0, 0, 1])
	else:
		reference[0:3] = base_rot.dot(hip_pose[[0, 1, 2]]) / 100
	if hip_rot is None:
		hip_rot = R.from_matrix(base_rot.dot(matrix[index, 0, 0:3, 0:3])).as_quat()
	else:
		hip_rot = R.from_matrix(base_rot.dot(hip_rot)).as_quat()
	reference[3:7] = hip_rot[[3, 0, 1, 2]]
	reference[2] += 0.02

	if previous_reference is not None:
		reference[0:3] = weight * reference[0:3] + (1-weight) * previous_reference[0:3]

		poses = R.from_quat([reference[[4, 5, 6, 3]], previous_reference[[4, 5, 6, 3]]])
		key_times = [0, 1]
		pose = Slerp(key_times, poses)([1-weight])
		reference[3:7] = pose.as_quat()[0, [3,0,1,2]]

	bone_index = [19, 20, 21, 15, 16, 17, 1, 2, 3, 4, 5, 11, 12, 13, 7, 8, 9]
	reference_index = [7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71]

	for i in range(len(bone_index)):
		
		if previous_reference is not None:
			pose1 = R.from_matrix(local_matrix[index, index_offset, bone_index[i], 0:3, 0:3]).as_quat()
			pose2 = R.from_quat(previous_reference[[reference_index[i]+1, reference_index[i]+2, reference_index[i]+3, reference_index[i]]])
			key_times = [0, 1]
			poses = R.from_quat([pose1, previous_reference[[reference_index[i]+1, reference_index[i]+2, reference_index[i]+3, reference_index[i]]]])
			pose = Slerp(key_times, poses)([1-weight])
			pose = pose.as_quat()[0, [3,0,1,2]]
		else:
			pose = local_matrix[index, index_offset, bone_index[i], 0:3, 0:3].as_quat()[[3,0,1,2]]
		reference[[reference_index[i], reference_index[i]+1, reference_index[i]+2, reference_index[i]+3]] = pose

	#set left leg
	# reference[7:11] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 19, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[11:15] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 20, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[15:19] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 21, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	# #set right leg
	# reference[19:23] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 15, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[23:27] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 16, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[27:31] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 17, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	# #set body
	# reference[31:35] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 1, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[35:39] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 2, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[39:43] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 3, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[43:47] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 4, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[47:51] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 5, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	# #set left arms
	# reference[51:55] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 11, 0:3, 0:3]).as_quat()[[3,0,2,1]])
	# reference[55:59] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 12, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[59:63] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 13, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	# #set right arms
	# reference[63:67] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 7, 0:3, 0:3]).as_quat()[[3,0,2,1]])
	# reference[67:71] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 8, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	# reference[71:75] = coordinate_transform(R.from_matrix(local_matrix[index, index_offset, 9, 0:3, 0:3]).as_quat()[[3,0,1,2]])


	virtual_human.setState(reference, np.zeros([virtual_human.getDOF()]))

	return reference

def motion_matching_query(kd_tree, feature_mean, feature_std, 
	current_root_pose, current_root_orientation, current_bone_pos, current_bone_velocity, current_phase,
	desired_root_linear_velocity):
	num_matched_frames = 10
	num_phase_matched_frames = 1
	query_feature = np.zeros((1, num_matched_frames * 4 + num_phase_matched_frames  * 6))
	frames = []
	phase_frames = [0, 1, 20]
	for i in range(num_matched_frames):
		frames.append(int((1+i) * 60 / num_matched_frames - 1))
	for i in range(num_matched_frames):
		query_feature[:, i*4:i*4+2] = current_root_orientation.T.dot(np.array([frames[i] * desired_root_linear_velocity[0], 0, frames[i] * desired_root_linear_velocity[2]]))[[0, 2]]
		query_feature[:, i*4+2:i*4+4] = current_root_orientation.T.dot(np.array([1, 0, 0]))[[0, 2]]
	# query_feature[0, 3 * 4:3 * 4+3] = current_bone_velocity[0, :].flatten().copy()
	# query_feature[0, 3 * 4 + 3:3 * 4 + 6] = current_root_orientation.T.dot(current_bone_pos[17, :].flatten().copy() - current_root_pose)
	# query_feature[0, 3 * 4 + 6:3 * 4 + 9] = current_root_orientation.T.dot(current_bone_pos[21, :].flatten().copy() - current_root_pose)
	# query_feature[0, 3 * 4 + 9:3 * 4 + 12] = current_bone_velocity[17, :].copy()
	# query_feature[0, 3 * 4 + 12:3 * 4 + 15] = current_bone_velocity[21, :].copy()

	phase_amp = np.zeros(3)
	phase_offset = np.zeros(3)
	for i in range(3):
		phase_amp[i] = np.sqrt(current_phase[i * 2]**2 + current_phase[i * 2 + 1]**2)
		phase_offset[i] = np.arctan2(current_phase[i*2+1]/phase_amp[i], current_phase[i*2]/phase_amp[i])

	for i in range(num_phase_matched_frames):
		for j in range(3):
			noise = np.random.randn(1)[0] * 2
			query_feature[0, num_matched_frames * 4 + i * 6 + j * 2: num_matched_frames * 4 + i * 6 + j * 2 + 2] = np.array([phase_amp[j] * np.cos(phase_offset[j] + 0.0167 * phase_frames[i] * (current_phase[6+j])), phase_amp[j] * np.sin(phase_offset[j] + 0.0167 * phase_frames[i] * (current_phase[6+j]))])
			query_feature[0, num_matched_frames * 4 + i * 6 + j * 2: num_matched_frames * 4 + i * 6 + j * 2 + 2] += np.random.randn(2) * 0
	# query_feature += np.random.randn(1, num_matched_frames * 4 + num_matched_frames * 6) * 0.1
	# print("real current_phase", current_phase)
	# print("query phase", query_feature[0, -6:])
	# query_feature[0, -6:] = current_phase[0:6].copy()
	query_feature = (query_feature-feature_mean) / feature_std
	# query_feature[:, 0:40] = 0
	dist, ind = kd_tree.query(query_feature, k=9)
	return dist, ind

balls = []
for i in range(23):
	balls.append(world.addSphere(0.02, 0.8))
balls.append(world.addSphere(0.1, 0.8))


# matrix = motion.to_matrix(local=False)[:, :, :, :]
# local_matrix = motion.to_matrix(local=True)[:, :, : ,:]
# pos = matrix[:, :, :3, 3]

# num_frames = matrix.shape[0]

# root_pos = np.zeros((num_frames, 3))
# root_rot = np.zeros((num_frames, 3, 3))
# root_pos[:, 0] = pos[:, 0, 0]
# root_pos[:, 2] = pos[:, 0, 2]

# root_rot[:, :, 0] = np.cross((pos[:, 15, :] - pos[:, 19, :]), (pos[:, 0, :] - pos[:, 1, :]))
# root_rot[:, 1, 0] = 0
# norms = np.apply_along_axis(np.linalg.norm, 1, root_rot[:, :, 0])
# root_rot[:, :, 0] /= norms[:, np.newaxis]
# root_rot[:, :, 1] = np.array([0, 1, 0])
# root_rot[:, :, 2] = np.cross(root_rot[:, :, 0], root_rot[:, :, 1])

# store_data = np.zeros((num_frames, 23, 3))
# bone_velocity = np.zeros((num_frames, 23, 3))


# #bone position w.r.t the root
# for i in range(23):
# 	A = np.transpose(root_rot[:, : :], axes=[0, 2, 1])
# 	x = pos[:, i, :] - root_pos
# 	# store_data[:, i, :] = np.einsum('ijk, ij->k', A, x)
# 	store_data[:, i, :] = np.matmul(A, x[:, :, None]).squeeze()
# #bone velocity
# for i in range(num_frames-1):
# 	for j in range(23):
# 		bone_velocity[i, j, :] = store_data[i+1, j, :] - store_data[i, j, :]


# #root velocity
# root_linear_vel = np.zeros((num_frames, 3))
# root_angular_vel = np.zeros((num_frames, 3, 3))
# root_angular_vel[-1, :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# for i in range(num_frames-1):
# 	root_linear_vel[i] = root_rot[i, :, :].T.dot(root_pos[i+1, :] - root_pos[i, :])
# 	root_angular_vel[i, :, :] = root_rot[i+1, :, :].dot(root_rot[i, :, :].T)

phase1 = np.loadtxt("NSM_phase/RunRandom.bvh/Phases_Standard.txt")[:, :]
phase2 = np.loadtxt("NSM_phase/WalkTurn1.bvh/Phases_Standard.txt")[:, :]
phase3 = np.loadtxt("NSM_phase/WalkRandom.bvh/Phases_Standard.txt")[:, :]
phase4 = np.loadtxt("NSM_phase/WalkSideBack1.bvh/Phases_Standard.txt")[:, :]
<<<<<<< HEAD
phase5 = np.loadtxt("NSM_phase/RunSideBack1.bvh/Phases_Standard.txt")[:, :]
=======
>>>>>>> 0da051044f942005b2448389cbfb3235d52a32f4

root_pos1, root_rot1, root_linear_vel1, root_angular_vel1, store_data1, bone_velocity1, local_matrix1, features1, bone_orientation1, phase1 = process_mocap("Loco/RunRandom.bvh", phase1)
root_pos2, root_rot2, root_linear_vel2, root_angular_vel2, store_data2, bone_velocity2, local_matrix2, features2, bone_orientation2, phase2 = process_mocap("Loco/WalkTurn1.bvh", phase2)
root_pos3, root_rot3, root_linear_vel3, root_angular_vel3, store_data3, bone_velocity3, local_matrix3, features3, bone_orientation3, phase3 = process_mocap("Loco/WalkRandom.bvh", phase3)
root_pos4, root_rot4, root_linear_vel4, root_angular_vel4, store_data4, bone_velocity4, local_matrix4, features4, bone_orientation4, phase4 = process_mocap("Loco/WalkSideBack1.bvh", phase4)
<<<<<<< HEAD
root_pos5, root_rot5, root_linear_vel5, root_angular_vel5, store_data5, bone_velocity5, local_matrix5, features5, bone_orientation5, phase5 = process_mocap("Loco/RunSideBack1.bvh", phase5)

root_pos = np.concatenate((root_pos1, root_pos2, root_pos3, root_pos4, root_pos5))
root_rot = np.concatenate((root_rot1, root_rot2, root_rot3, root_rot4, root_rot5))
root_linear_vel = np.concatenate((root_linear_vel1, root_linear_vel2, root_linear_vel3, root_linear_vel4, root_linear_vel5))
root_angular_vel = np.concatenate((root_angular_vel1, root_angular_vel2, root_angular_vel3, root_angular_vel4, root_angular_vel5))
store_data = np.concatenate((store_data1, store_data2, store_data3, store_data4, store_data5))
bone_velocity = np.concatenate((bone_velocity1, bone_velocity2, bone_velocity3, bone_velocity4, bone_velocity5))
local_matrix = np.concatenate((local_matrix1, local_matrix2, local_matrix3, local_matrix4, local_matrix5))
bone_orientation = np.concatenate((bone_orientation1, bone_orientation2, bone_orientation3, bone_orientation4, bone_orientation5))
phase = np.concatenate((phase1, phase2, phase3, phase4, phase5))
features = np.concatenate((features1, features2, features3, features4, features5))
=======


root_pos = np.concatenate((root_pos1, root_pos2, root_pos3, root_pos4))
root_rot = np.concatenate((root_rot1, root_rot2, root_rot3, root_rot4))
root_linear_vel = np.concatenate((root_linear_vel1, root_linear_vel2, root_linear_vel3, root_linear_vel4))
root_angular_vel = np.concatenate((root_angular_vel1, root_angular_vel2, root_angular_vel3, root_angular_vel4))
store_data = np.concatenate((store_data1, store_data2, store_data3, store_data4))
bone_velocity = np.concatenate((bone_velocity1, bone_velocity2, bone_velocity3, bone_velocity4))
local_matrix = np.concatenate((local_matrix1, local_matrix2, local_matrix3, local_matrix4))
bone_orientation = np.concatenate((bone_orientation1, bone_orientation2, bone_orientation3, bone_orientation4))
phase = np.concatenate((phase1, phase2, phase3, phase4))
features = np.concatenate((features1, features2, features3, features4))
>>>>>>> 0da051044f942005b2448389cbfb3235d52a32f4
num_frames = root_pos.shape[0]

print(features.shape, local_matrix.shape)

#build feature space for kd tree
# features = np.zeros((num_frames - 60, 3 * 4 + 3 * 5))
# for i in range(num_frames - 60):
# 	frames = [19, 39, 59]
# 	root_pos_feature = np.zeros(3)
# 	for j in range(3):
# 		features[i, j*4:j*4+2] = root_rot[i, :, :].T.dot(root_pos[frames[j] + i] - root_pos[i])[[0, 2]]
# 		features[i, j*4+2:j*4+4] = root_rot[i+frames[j], :, :].dot(root_rot[i, :, :].T).dot(np.array([1, 0, 0]))[[0, 2]]
# 		#features[i, j*9+6:j*9+9] = root_angular_vel[i+j, :, 1].copy()
# 	features[i, 3 * 4: 3 * 4 + 3] = bone_velocity[i, 0, :].flatten().copy()
# 	features[i, 3 * 4 + 3: 3 * 4 + 6] = store_data[i, 17, :].flatten().copy()
# 	features[i, 3 * 4 + 6: 3 * 4 + 9] = store_data[i, 21, :].flatten().copy()
# 	features[i, 3 * 4 + 9: 3 * 4 + 12] = bone_velocity[i, 17, :].flatten().copy()
# 	features[i, 3 * 4 + 12: 3 * 4 + 15] = bone_velocity[i, 21, :].flatten().copy()

feature_mean = np.mean(features, axis=0)
feature_std = np.std(features, axis=0)
feature_std[0:2] /= 1
# feature_std[4:6] /= 2
# feature_std[8:10] /= 2
# feature_std[0:12] /= 2
feature_std[-6:] /= 1
# feature_std[-18:-24] /= 4
# feature_std[12:14] /= 4
# feature_std[16:18] /= 4
# feature_std[-9:-3] /= 4
features = (features - feature_mean) / feature_std

#build kd tree
# features[:, 0:40] = 0
tree = KDTree(features)
dist, ind = tree.query(features[2:3, :], k=10)
print(dist, ind)

server.launchServer(8081)
world.integrate1()

world_vel = np.zeros(3)

#kd tree query
current_root_pose = np.zeros(3)
<<<<<<< HEAD
desired_root_linear_velocity = np.array([0, 0, 2])
=======
desired_root_linear_velocity = np.array([0, 0, 0])
>>>>>>> 0da051044f942005b2448389cbfb3235d52a32f4
current_bone_pos = store_data[0, :, :].copy()
current_bone_velocity = bone_velocity[0, :, :].copy()
current_root_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
current_bone_orientation = current_root_orientation.dot(bone_orientation[0, :, :])
current_phase = phase[0, 0, 0:9].copy()
desired_root_delta_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

save_frames = 60
save_reference = np.zeros((save_frames, 75))
frame_counter = 0
previous_reference =  np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0.,])

save_phase = np.zeros((200, 6))
save_time = 0
while True:
	dist, ind = motion_matching_query(tree, feature_mean, feature_std, 
		current_root_pose, current_root_orientation, current_bone_pos, current_bone_velocity, current_phase, 
		desired_root_linear_velocity)
<<<<<<< HEAD
	# print(dist, ind)
=======
	print(dist, ind)
>>>>>>> 0da051044f942005b2448389cbfb3235d52a32f4
	for t in range(5):
		# current_root_pose += desired_root_linear_velocity / 10
		# current_root_orientation = current_root_orientation.dot(desired_root_delta_orientation)
		
		current_root_pose += current_root_orientation.dot(root_linear_vel[ind[0,0], t])
		current_root_orientation = root_angular_vel[ind[0,0], t, :, :].dot(current_root_orientation)
		
		for i in range(1):
			current_bone_pos[i, :] = (store_data[ind[0,0], t+1, :]) + current_root_pose
		# 	balls[i].setPosition(current_bone_pos[i, 0]/200, current_bone_pos[i, 2]/200, current_bone_pos[i, 1]/200)
		# current_bone_velocity[0, :] = bone_velocity[ind[0, 0]+1+t, 0, :].copy()
		# current_bone_velocity[17, :] = bone_velocity[ind[0, 0]+1+t, 17, :].copy()
		# current_bone_velocity[21, :] = bone_velocity[ind[0, 0]+1+t, 21, :].copy()
		current_phase = phase[ind[0, 0], 1+t, 0:9].copy()
		# print("current phase", current_phase)

		current_bone_orientation = current_root_orientation.dot(bone_orientation[ind[0, 0], t+1, :])

		weight = np.exp((t+1) * 1.0 / 50) / np.exp(1.0)
		previous_reference = set_human_pose(ind[0, 0], 1+t, current_bone_pos[0, :], current_bone_orientation, previous_reference, weight)

		# save_reference[frame_counter, :] = reference.copy()
		# frame_counter += 1
		# if (frame_counter == save_frames):
		# 	with open("reference.npy", 'wb') as f:
		# 		np.save(f, save_reference)

<<<<<<< HEAD
		print("phase ", ind[0,0], current_phase[0:6])
		save_phase[save_time, :] = current_phase[0:6]
=======
		save_phase[save_time, :] = current_phase[0:6]
		phase_speed = current_phase[-3:] / 60 * 2 * np.pi
		print("phase speed", phase_speed)
>>>>>>> 0da051044f942005b2448389cbfb3235d52a32f4
		save_time += 1
		if save_time == 200:
			import matplotlib.pyplot as plt
			fig = plt.figure()
<<<<<<< HEAD
			for j in range(3):
				ax = fig.add_subplot(3,1,j+1)
				phase_amp = np.sqrt(save_phase[:, j * 2]**2 + save_phase[:, j*2+1]**2)
				phase_offset = np.arctan2(save_phase[:, j*2]/phase_amp[:], save_phase[:, j*2+1]/phase_amp[:])
				ax.plot(phase_amp)
=======
			ax = fig.add_subplot(111)
			phase_amp = np.sqrt(save_phase[:, 0]**2+save_phase[:, 1]**2)
			phase_offset = np.arctan2(save_phase[:, 0] / phase_amp, save_phase[:, 1] / phase_amp)
			ax.plot(phase_offset)
>>>>>>> 0da051044f942005b2448389cbfb3235d52a32f4
			plt.show()
			save_time = 0

		world.integrate()
		import time; time.sleep(0.016)

motion = bvh.load("Loco/RunRandom.bvh")
matrix = motion.to_matrix(local=False)[:, :, :, :]
while True:
	for j in range(num_frames):
		#set skeleton pose
		set_human_pose(j)

		# set point cloud pose
		# for i in range(23):
		# 	bone_pos = root_rot[j,:,:].dot(store_data[j, i, :]) + root_pos[j, :]
		# 	balls[i].setPosition(bone_pos[0]/200, bone_pos[2]/200, bone_pos[1]/200)
		# 	# balls[i].setPosition(pos[j, i, 0]/200, pos[j, i, 2] / 200, pos[j, i, 1] / 200)

		# balls[23].setPosition(root_pos[j, 0] / 200, root_pos[j, 2] / 200, root_pos[j, 1] / 200)

		# r = R.from_matrix(root_rot[j, :, :]).as_quat()
		# displacement = root_rot[j, :, :].dot(np.array([0.1, 0, 0])) + root_pos[j, :] / 200
		# direction.setPosition(displacement[0], displacement[2], displacement[1] + 0.1)
		# direction.setOrientation(r[3], r[0], r[2], -r[1])
		

		# #visualize linear velocity
		# world_vel = world_vel * 0.9 + 0.1 * root_rot[j, :, :].dot(root_linear_vel[j])
		# world_orientation = np.zeros((3, 3))
		# normalized_world_vel = world_vel / np.linalg.norm(world_vel)
		# world_orientation[:, 0] = normalized_world_vel
		# world_orientation[:, 1] = np.array([0, 1, 0])
		# world_orientation[:, 2] = np.cross(world_orientation[:, 0], world_orientation[:, 1])
		# world_quat = R.from_matrix(world_orientation).as_quat()
		# visBox.setBoxSize(np.linalg.norm(world_vel), 0.01, 0.01)
		# displacement = root_pos[j, :] / 200 + world_orientation.dot(np.array([1, 0, 0]) * np.linalg.norm(world_vel) / 2)
		# visBox.setPosition(np.array([displacement[0], displacement[2], displacement[1] + 0.1]))
		# visBox.setOrientation(np.array([world_quat[3], world_quat[0], world_quat[2], -world_quat[1]]))

		# #visualzie angular velocity
		# visAngularVelBox.setBoxSize(0.4, 0.1, 0.1)
		# displacement = root_rot[j, :, :].dot(np.array([0.1, 0, 0])) + root_pos[j, :] / 200
		# visAngularVelBox.setPosition(np.array([displacement[0], displacement[2], displacement[1] + 0.1]))
		# next_orientation = root_rot[j, : ,:].copy()
		# for k in range(10):
		# 	if k + j < num_frames-1:
		# 		next_orientation = root_angular_vel[j + k, :, :].dot(next_orientation)
		# next_quat = R.from_matrix(next_orientation).as_quat()
		# visAngularVelBox.setOrientation(np.array([next_quat[3], next_quat[0], next_quat[2], -next_quat[1]]))

		import time; time.sleep(0.016)

		world.integrate()