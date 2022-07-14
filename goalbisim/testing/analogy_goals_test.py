import numpy as np
import torch
import sklearn
from sklearn.neighbors import NearestNeighbors
from goalbisim.utils.video import MultiVideoRecorder
from rlkit.core import logger


def nearest_neighbor_analogy3(env, eval_transforms, device, replay_buffer, representation, step, details, samples = 5, k = 1, training_count = 1500, train_set = False, eval_replay_buffer = None, forced_samples = None):

	
	zs = []
	#z_phis = []
	idxs = []

	nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
	#nn_1_phi = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', p=1)
	#nn_psi = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')


	video = MultiVideoRecorder(dir_name = details['eval_analogy_save_dir'], width = 3, fps = 9)
	video.init(num_trajectories = (k + 2) * samples, max_trajectory_length = env._max_episode_steps + 1)

	tot = env._max_episode_steps + 1
	for i in range(training_count // 500):
		obses, action, reward, next_obses, not_dones, goals, kwargs = replay_buffer.sample(batch_size = 500, fetch_states = True)
		analogy_obses, analogy_goals = kwargs['analogy_obses'], kwargs['analogy_goals']

		#torch.cat([representation.psi(obses).detach().cpu().numpy(), representation.psi(goals).detach().cpu().numpy()], dim = 1)
		z = np.concatenate([representation.psi(obses).detach().cpu().numpy(), representation.psi(obses).detach().cpu().numpy(), representation.psi(goals).detach().cpu().numpy()], axis = 1)
		#z_psi = representation.psi(goals).detach().cpu().numpy() - representation.psi(obses).detach().cpu().numpy()
		idx = kwargs['idxs']

		zs.append(z)
		#z_psis.append(z_psi)
		idxs.append(idx)

	zs = np.concatenate(zs, axis = 0)
	#z_phis = np.concatenate(z_phis, dim = 0)
	idxs = np.concatenate(idxs, axis = 0)


	nn.fit(zs)
	#nn_1_phi.fit(z_phis)
	#nn_psi.fit(z_psis)

	avg_action_error = []


	sampled_obses, _, _, _, _, sampled_goals, kwargs = eval_replay_buffer.sample(batch_size = samples, fetch_states = True)
	sampled_idxs = kwargs['idxs']
	for i in range(samples):
		
		if forced_samples is not None:

			traj_obses = eval_transforms(forced_samples[0][i], device = device)
			traj_goals = eval_transforms(forced_samples[1][i], device = device)

			#for obs in print_obses[:tot]:
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

			#import pdb; pdb.set_trace()
			for _ in range(20):
				video.record(np.uint8(kwargs['analogy_obses'][i].cpu().detach().numpy().copy() * 255))
			for _ in range(20):
				video.record(np.uint8(kwargs['analogy_goals'][i].cpu().detach().numpy().copy() * 255))
			video.step()

		#context = representation.phi(traj_obses.unsqueeze(0), traj_goals.unsqueeze(0))
		z = np.concatenate([representation.psi(traj_obses.unsqueeze(0)).detach().cpu().numpy(), representation.psi(kwargs['analogy_obses'][i].unsqueeze(0)).detach().cpu().numpy(), representation.psi(kwargs['analogy_goals'][i].unsqueeze(0)).detach().cpu().numpy()], axis = 1)
		#z_psi = representation.psi(traj_goals.unsqueeze(0)).detach().cpu().numpy() - representation.psi(traj_obses.unsqueeze(0)).detach().cpu().numpy()

		nn_idxs = nn.kneighbors(z, k, False)[0]
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

	video.save(start + "/video_nn_analogies_policy", step)