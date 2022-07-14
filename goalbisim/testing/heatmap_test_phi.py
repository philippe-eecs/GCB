import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wandb
from sklearn.manifold import TSNE
from rlkit.core import logger




def create_trajectory_color_map(env, eval_transforms, replay_buffer, representation, step, details, truth_metric = 'rewards', total_trajectories = 1, train_set = False):


	#current_trajectory = 0
	#z_s = []

	phi = representation.phi
	
	#z_2_norms = []
	#z_1_norms = []
	#zg_2_norms = []
	#zg_1_norms = []
	#true_rewards = []

	#states = []
	#import pdb; pdb.set_trace()
	tsne = TSNE()

	if train_set:
		beginning = 'train'
	else:
		beginning = 'eval'


	for i in range(total_trajectories):
		traj_obses, traj_actions, traj_cum_rewards, traj_rewards, traj_next_obses, traj_not_dones, traj_goals, kwargs = replay_buffer.sample_trajectory(fetch_states = True)

		z = phi.encode(traj_obses, traj_goals, detach = True).detach().cpu().numpy()

		g = phi.encode(traj_goals, traj_goals, detach = True).detach().cpu().numpy()

		zg_2_norm = np.linalg.norm(z - g, ord = 2, dim = 1)
		z_2_norm = np.linalg.norm(z, ord = 2, dim = 1)
		zz_2_norm = np.linalg.norm(z[:-1] - z[1:], ord = 2, dim = 1)
		zg_1_norm = np.linalg.norm(z - g, ord = 1, dim = 1)
		z_1_norm = np.linalg.norm(z, ord = 1, dim = 1)
		zz_1_norm = np.linalg.norm(z[:-1] - z[1:], ord = 1, dim = 1)


		states = kwargs['states']
		td = kwargs['td']
		if truth_metric == 'cum_rewards':
			true_metric = np.abs(traj_cum_rewards)
		elif truth_metric == 'rewards':
			true_metric = np.abs(traj_rewards)
		elif truth_metric == 'number_of_actions_left':
			raise NotImplementedError
		else:
			raise NotImplementedError

		z_tsne = tsne.fit_transform(z)
		tsne = TSNE()
		zg_tsne = tsne.fit_transform(z - g)
		tsne = TSNE()

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c = zg_2_norm, cmap = coolwarm)
		plt.title("Phi - Goal L2 Norm of Encoded TSNE on Phi(s,g) for Train Step " + str(step))
		plt.xlabel("TSNE Principal Component 1")
		plt.ylabel("TSNE Principal Component 2")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/tsne_color_map_l2')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(zg_tsne[:, 0], zg_tsne[:, 1], c = zg_2_norm, cmap = coolwarm)
		plt.title("Phi - Goal L2 Norm of Encoded TSNE on Phi(s,g) - Phi(g,g) for Train Step " + str(step))
		plt.xlabel("TSNE Principal Component 1")
		plt.ylabel("TSNE Principal Component 2")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/tsne_color_map_l2goal')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0], states[:, 1], c = zg_2_norm, cmap = coolwarm)
		plt.title("Phi - Goal L2 Norm of Encoded Trajectory for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_l2goal')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0], states[:, 1], c = zg_2_norm, cmap = coolwarm)
		plt.title("Phi - Goal L2 Norm of Encoded Trajectory for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_l2goal')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0], states[:, 1], c = zg_1_norm, cmap = coolwarm)
		plt.title("Phi - Goal L1 Norm of Encoded Trajectory for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_l1goal')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0], states[:, 1], c = z_2_norm, cmap = coolwarm)
		plt.title("Phi L2 Norm of Encoded Trajectory for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_l2')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0], states[:, 1], c = z_1_norm, cmap = coolwarm)
		plt.title("Phi L1 Norm of Encoded Trajectory for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_l1')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0][:-1], states[:, 1][:-1], c = zz_2_norm, cmap = coolwarm)
		plt.title("Phi - Phi' L2 Norm of Encoded Trajectory for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_l2prime')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0][:-1], states[:, 1][:-1], c = zz_1_norm, cmap = coolwarm)
		plt.title("Phi - Phi' L1 Norm of Encoded Trajectory for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_l1prime')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0], states[:, 1], c = true_metric, cmap = coolwarm)
		plt.title("True Distance of Encoded Trajectories for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_truth')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0], states[:, 1], c = np.abs(kwargs['rtg']), cmap = coolwarm)
		plt.title("Discounted RTG of Encoded Trajectories for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_rtg')

		plt.clf()
		fig, ax = plt.subplots()
		divider = make_axes_locatable(ax)
		scat = plt.scatter(states[:, 0], states[:, 1], c = np.abs(td), cmap = coolwarm)
		plt.title("Temporal Distance of Encoded Trajectories for Train Step " + str(step))
		plt.xlabel("X State")
		plt.ylabel("Y State")
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(scat, cax=cax, orientation='vertical')
		logger.logging_tool.log_figure(plt, '', wandb_save_loc = beginning + '/trajectory_color_map_td')


def create_trajectory_color_map_nogoal(replay_buffer, phi, itr, total_trajectories = 1):


	#current_trajectory = 0
	#z_s = []
	
	z_norms = []
	z_g_norms = []
	true_distances = []
	states = []


	for i in range(total_trajectories):
		traj_obses, traj_actions, _, traj_rewards, traj_next_obses, traj_not_dones, traj_goals, kwargs = replay_buffer.sample_trajectory(fetch_states = True)


		z = phi.encode(traj_obses, detach = True).detach().cpu().numpy()

		#g = phi.encode(traj_goals, detach = True).detach().cpu().numpy()

		z_norm = np.linalg.norm(z, dim = 1)

		#z_g_norm = np.linalg.norm(z - g, dim = 1)

		z_norms.append(z_norm)

		#z_g_norms.append(z_g_norm)

		state = kwargs['states']
		true_distance = -traj_rewards

		true_distances.append(true_distance)

		states.append(state)

	#import pdb; pdb.set_trace()

	z_norms = np.concatenate(z_norms, dim = 0)
	#z_g_norms = np.concatenate(z_g_norms, dim = 0)
	states = np.concatenate(states, dim = 0)

	states = states[:, :2]

	plt.clf()
	fig, ax = plt.subplots()
	divider = make_axes_locatable(ax)
	scat = plt.scatter(states[:, 0], states[:, 1], c = z_norms, cmap = coolwarm)
	plt.title("Phi Norm of Encoded Trajectories")
	plt.xlabel("X State")
	plt.ylabel("Y State")
	cax = divider.append_axes('right', size='5%', pad=0.05)
	#im = ax.imshow(states, cmap = coolwarm)
	fig.colorbar(scat, cax=cax, orientation='vertical')
	plt.savefig("data/plots/Trajectory_Color_Map_PHI" + str(itr))

	'''
	plt.clf()
	fig, ax = plt.subplots()
	divider = make_axes_locatable(ax)
	scat = plt.scatter(states[:, 0], states[:, 1], c = z_norms, cmap = coolwarm)
	plt.title("Phi Norm of Encoded Trajectories")
	plt.xlabel("X State")
	plt.ylabel("Y State")
	cax = divider.append_axes('right', size='5%', pad=0.05)
	#im = ax.imshow(states, cmap = coolwarm)
	fig.colorbar(scat, cax=cax, orientation='vertical')
	plt.savefig("data/plots/Trajectory_Color_Map_PHI_to_G" + str(itr))
	'''


	plt.clf()
	fig, ax = plt.subplots()
	divider = make_axes_locatable(ax)
	scat = plt.scatter(states[:, 0], states[:, 1], c = true_distances, cmap = coolwarm)
	plt.title("True Distance of Encoded Trajectories")
	plt.xlabel("X State")
	plt.ylabel("Y State")
	cax = divider.append_axes('right', size='5%', pad=0.05)
	#im = ax.imshow(states, cmap = coolwarm)
	fig.colorbar(scat, cax=cax, orientation='vertical')
	plt.savefig("data/plots/Trajectory_Color_Map_Truth" + str(itr))



def create_trajectory_color_map_fixed_goal(replay_buffer, phi, total_trajectories = 3):
	z_norms = []
	true_distances = []
	states = []

	obses, actions, curr_rewards, rewards, next_obses, not_dones, goals, kwargs = replay_buffer.sample(2)

	fixed_goal = obses[0:1]
	fixed_state = kwargs['states'][0:1]


	for i in range(total_trajectories):
		traj_obses, traj_actions, _, traj_rewards, traj_next_obses, traj_not_dones, traj_goals, kwargs = replay_buffer.sample_trajectory(fetch_states = True)


		z = phi.encode(traj_obses, fixed_goal.repeat(traj_obses.shape[0]), detach = True).detach().cpu().numpy()

		z_norm = np.linalg.norm(z, dim = 1)


		z_norms.append(z_norm)

		state = kwargs['states']
		true_distance = np.linalg.norm(state - fixed_state.repeat(state.shape[0]))
		true_distances.append(true_distance)

		states.append(state)

	z_norms = np.concatenate(z_norms, dim = 0)
	states = np.concatenate(states, dim = 0)

	plt.clf()

	plt.scatter(states, cmap = z_norms)
	plt.title("Phi Norm of Encoded Trajectories")
	plt.xlabel("X State")
	plt.ylabel("Y State")
	plt.savefig("Trajectory_Color_Map_PHI")

	plt.clf()

	plt.scatter(states, cmap = true_distances)
	plt.title("True Distance of Encoded Trajectories")
	plt.xlabel("X State")
	plt.ylabel("Y State")
	plt.savefig("Trajectory_Color_Map_Truth")




def create_grid_heat_map(env, phi, total_points = 64):
	pass










