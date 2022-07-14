import numpy as np
import torch
import sklearn
from sklearn.neighbors import NearestNeighbors
from goalbisim.utils.video import MultiVideoRecorder
from rlkit.core import logger


def nearest_neighbor_analogy(env, eval_transforms, device, replay_buffer, representation, step, details, samples = 5, k = 1, training_count = 1500, train_set = False, eval_replay_buffer = None, forced_samples = None):
	z_psis = []
	z_phis = []
	idxs = []

	nn_phi = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
	nn_1_phi = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', p=1)
	nn_psi = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')


	video = MultiVideoRecorder(dir_name = details['eval_analogy_save_dir'], width = (k * 3 + 1), fps = 9)
	video.init(num_trajectories = (k * 3 + 1) * samples, max_trajectory_length = env._max_episode_steps + 1)

	tot = env._max_episode_steps + 1
	for i in range(training_count // 500):
		obses, action, reward, next_obses, not_dones, goals, kwargs = replay_buffer.sample(batch_size = 500, fetch_states = True)

		z_phi = representation.phi(obses, goals).detach().cpu().numpy() - representation.phi(goals, goals).detach().cpu().numpy()
		z_psi = np.concatenate([representation.psi(obses).detach().cpu().numpy(), representation.psi(goals).detach().cpu().numpy()], axis = 1)
		idx = kwargs['idxs']

		z_phis.append(z_phi)
		z_psis.append(z_psi)
		idxs.append(idx)

	z_psis = np.concatenate(z_psis, axis = 0)
	z_phis = np.concatenate(z_phis, axis = 0)
	idxs = np.concatenate(idxs, axis = 0)


	nn_phi.fit(z_phis)
	nn_1_phi.fit(z_phis)
	nn_psi.fit(z_psis)

	sampled_obses, _, _, _, _, sampled_goals, kwargs = eval_replay_buffer.sample(batch_size = samples, fetch_states = True)
	sampled_idxs = kwargs['idxs']
	for i in range(samples):

		if forced_samples is not None:

			traj_obses = eval_transforms(forced_samples[0][i], device = device)
			traj_goals = eval_transforms(forced_samples[1][i], device = device)
			video.record(forced_samples[0][i])
			video.record(forced_samples[1][i])
			video.step()

		else:
		
			traj_obses = torch.as_tensor(sampled_obses[i], device = device)
			traj_goals = torch.as_tensor(sampled_goals[i], device = device)

			start_idx = sampled_idxs[i]
			end_idx = eval_replay_buffer.trajectory_end_idx[start_idx][0]
			print_obses = eval_replay_buffer.obses[start_idx : end_idx]
			for obs in print_obses[:tot]:
				video.record(obs)
			video.step()

		z_phi = representation.phi(traj_obses.unsqueeze(0), traj_goals.unsqueeze(0)).detach().cpu().numpy() - representation.phi(traj_goals.unsqueeze(0), traj_goals.unsqueeze(0)).detach().cpu().numpy()
		z_psi = np.concatenate([representation.psi(traj_obses.unsqueeze(0)).detach().cpu().numpy(), representation.psi(traj_goals.unsqueeze(0)).detach().cpu().numpy()], axis = 1)

		nn_idxs = nn_phi.kneighbors(z_phi, k, False)[0]
		for analogy in nn_idxs:
			analogy_start_idx = idxs[analogy]
			analogy_end_idx = replay_buffer.trajectory_end_idx[idxs[analogy]][0]
			analogy_obses = replay_buffer.obses[analogy_start_idx : analogy_end_idx]
			analogy_actions = replay_buffer.actions[analogy_start_idx : analogy_end_idx]

			for obs in analogy_obses[:tot]:
				video.record(obs)

			video.step()

		nn_idxs = nn_1_phi.kneighbors(z_phi, k, False)[0]
		for analogy in nn_idxs:
			analogy_start_idx = idxs[analogy]
			analogy_end_idx = replay_buffer.trajectory_end_idx[idxs[analogy]][0]

			analogy_obses = replay_buffer.obses[analogy_start_idx : analogy_end_idx]

			for obs in analogy_obses[:tot]:
				video.record(obs)

			video.step()

		nn_idxs = nn_psi.kneighbors(z_psi, k, False)[0]
		for analogy in nn_idxs:
			analogy_start_idx = idxs[analogy]
			analogy_end_idx = replay_buffer.trajectory_end_idx[idxs[analogy]][0]
			analogy_obses = replay_buffer.obses[analogy_start_idx : analogy_end_idx]

			for obs in analogy_obses[:tot]:
				video.record(obs)

			video.step()

	if train_set:
		start = 'train'
	else:
		start = 'eval'

	if forced_samples is not None:
		start += '_env_dist'

	video.save(start + "/video_nn_analogies", step)











