import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wandb
from rlkit.core import logger
from sklearn.manifold import TSNE
from colour import Color

from numpy import inner
from numpy.linalg import norm

def cosine_sim(a, b):
	cosine = torch.nn.CosineSimilarity(dim = 1)
	val = cosine(torch.Tensor(a), torch.Tensor(b))

	return val.numpy()

def collect_analogy(env, eval_transforms, replay_buffer, representation, minimum_length = 2, maximum_length = 10):
	start_idx = 0
	end_idx = 0
	phi = representation.phi
	psi = representation.psi

	while end_idx - start_idx < minimum_length or end_idx - start_idx > maximum_length:
		traj_obses, traj_actions, traj_cum_rewards, traj_rewards, traj_next_obses, traj_not_dones, traj_goals, kwargs = replay_buffer.sample_trajectory(fetch_states = True)
		start_idx = np.random.randint(0, traj_actions.shape[0])
		end_idx = np.random.randint(start_idx, traj_actions.shape[0])

	action_set = traj_actions[start_idx : end_idx]

	#for obs in traj_obses[start_idx:end_idx]

	a_b = phi(traj_obses[start_idx].unsqueeze(0), traj_obses[end_idx].unsqueeze(0)).detach().cpu().numpy()
	b_b = phi(traj_obses[end_idx].unsqueeze(0), traj_obses[end_idx].unsqueeze(0)).detach().cpu().numpy()
	a = psi(traj_obses[start_idx].unsqueeze(0)).detach().cpu().numpy()
	b = psi(traj_obses[end_idx].unsqueeze(0)).detach().cpu().numpy()

	c, goal, extra = env.reset()

	#Could be the case that it bangs up against a wall, not particuarly interesting...

	for step in range(action_set.shape[0]):
		d, reward, done, extra = env.step(action_set[step])

	c = eval_transforms(c, 'cuda')
	d = eval_transforms(d, 'cuda')

	c_d = phi(c.unsqueeze(0), d.unsqueeze(0)).detach().cpu().numpy()
	d_d = phi(d.unsqueeze(0), d.unsqueeze(0)).detach().cpu().numpy()
	c = psi(c.unsqueeze(0)).detach().cpu().numpy()
	d = psi(d.unsqueeze(0)).detach().cpu().numpy()

	return a_b, b_b, a, b, c_d, d_d, c, d

def collect_analogies(env, eval_transforms, replay_buffer, representation, minimum_length = 5, maximum_length = 10, repeats = 10):
	a_bs = []
	b_bs = []
	a_s = []
	b_s = []
	c_ds = []
	d_ds = []
	c_s = []
	d_s = []
	#video = MultiVideoRecorder(dir_name = details['eval_analogy_save_dir'], width = 2, fps = 15)
	#video.init(num_trajectories = repeats, max_trajectory_length = maximum_length)

	for _ in range(repeats):
		a_b, b_b, a, b, c_d, d_d, c, d = collect_analogy(env, eval_transforms, replay_buffer, representation, minimum_length, maximum_length)
		a_bs.append(a_b)
		b_bs.append(b_b)
		a_s.append(a)
		b_s.append(b)
		c_ds.append(c_d)
		d_ds.append(d_d)
		c_s.append(c)
		d_s.append(d)



	a_bs = np.concatenate(a_bs, dim = 0)
	b_bs = np.concatenate(b_bs, dim = 0)
	a_s = np.concatenate(a_s, dim = 0)
	b_s = np.concatenate(b_s, dim = 0)

	c_ds = np.concatenate(c_ds, dim = 0)
	d_ds = np.concatenate(d_ds, dim = 0)
	c_s = np.concatenate(c_s, dim = 0)
	d_s = np.concatenate(d_s, dim = 0)

	return a_bs, b_bs, a_s, b_s, c_ds, d_ds, c_s, d_s

def compute_distances(v):
	import itertools
	all_pairs = np.array(list(itertools.combinations(range(len(v)), 2)))
	avg_distance = np.linalg.norm((v[all_pairs[:, 0]] - v[all_pairs[:, 1]]), dim = 1).mean()
	norm_distance = np.abs((np.linalg.norm(v[all_pairs[:, 0]], dim = 1) - np.linalg.norm(v[all_pairs[:, 1]], dim = 1))).mean()

	return avg_distance, norm_distance


def analogy_test(env, eval_transforms, replay_buffer, representation, step, details, train_set = False):


	phi = representation.phi
	psi = representation.psi

	tsne = TSNE()

	a_bs, b_bs, a_s, b_s, c_ds, d_ds, c_s, d_s = collect_analogies(env, eval_transforms, replay_buffer, representation)


	phi_distance_avg, phi_norm_avg = compute_distances(np.concatenate([a_bs, c_ds], dim = 0))
	phi_ground_distance_avg, phi_ground_norm_avg = compute_distances(np.concatenate([a_bs - b_bs, c_ds - d_ds], dim = 0))
	psi_distance_avg, psi_norm_avg = compute_distances(np.concatenate([b_s - a_s, d_s - c_s], dim = 0))
	phi_ground_psi_distance_avg, phi_ground_psi_norm_avg = compute_distances(np.concatenate([a_bs - b_bs, c_ds - d_ds, b_s - a_s, d_s - c_s], dim = 0))
	phi_psi_distance_avg, phi_psi_norm_avg = compute_distances(np.concatenate([a_bs, c_ds, b_s - a_s, d_s - c_s], dim = 0))

	tsne = TSNE()

	phi_tsne = tsne.fit_transform(np.concatenate([a_bs, c_ds], dim = 0))

	markers = ["v"] * a_bs.shape[0] + ["^"] * a_bs.shape[0]
	#, "o" , "v" , "^" , "<", ">"]

	red = Color("red")
	colors = list(red.range_to(Color("green"), a_bs.shape[0])) * 2

	plt.clf()
	fig, ax = plt.subplots()
	for x, y, m, c in zip(phi_tsne[:, 0], phi_tsne[:, 1], markers, colors):
		scat = plt.scatter(x, y, marker = m, cmap = c)
	plt.title("Phi(a,b) and Phi(c,d) TSNE Plot " + str(step))
	plt.xlabel("TSNE Principal Component 1")
	plt.ylabel("TSNE Principal Component 2")
	logger.logging_tool.log_figure(plt, '', wandb_save_loc = 'eval/phi_tsne')

	phi_goal_tsne = tsne.fit_transform(np.concatenate([a_bs - b_bs, c_ds - d_ds], dim = 0)) #Could be possible that phi(s,g) - phi(g,g) makes sense in a norm sense but NOT a representation sense
	tsne = TSNE()
	plt.clf()
	fig, ax = plt.subplots()
	for x, y, m, c in zip(phi_goal_tsne[:, 0], phi_goal_tsne[:, 1], markers, colors):
		scat = plt.scatter(x, y, marker = m, cmap = c)
	plt.title("Phi(a,b) - Phi(b,b) and Phi(c,d) - Phi(d,d) TSNE Plot " + str(step))
	plt.xlabel("TSNE Principal Component 1")
	plt.ylabel("TSNE Principal Component 2")
	logger.logging_tool.log_figure(plt, '', wandb_save_loc = 'eval/phi_goal_tsne')

	psi_goal_tsne = tsne.fit_transform(np.concatenate([b_s - a_s, d_s - c_s], dim = 0))
	tsne = TSNE()
	plt.clf()
	fig, ax = plt.subplots()
	for x, y, m, c in zip(psi_goal_tsne[:, 0], psi_goal_tsne[:, 1], markers, colors):
		scat = plt.scatter(x, y, marker = m, cmap = c)
	plt.title("Psi(b) - Psi(a) and Psi(c) - Psi(d) TSNE Plot " + str(step))
	plt.xlabel("TSNE Principal Component 1")
	plt.ylabel("TSNE Principal Component 2")
	logger.logging_tool.log_figure(plt, '', wandb_save_loc = 'eval/psi_goal_tsne')	

	colors *= 2
	markers += 	["<"] * a_bs.shape[0] + [">"] * a_bs.shape[0]

	psi_tsne = tsne.fit_transform(np.concatenate([a_s, b_s, c_s, d_s], dim = 0))
	tsne = TSNE()
	plt.clf()
	fig, ax = plt.subplots()
	for x, y, m, c in zip(psi_tsne[:, 0], psi_tsne[:, 1], markers, colors):
		scat = plt.scatter(x, y, marker = m, cmap = c)
	plt.title("Psi(a), Psi(b), Psi(c), Psi(d) TSNE Plot " + str(step))
	plt.xlabel("TSNE Principal Component 1")
	plt.ylabel("TSNE Principal Component 2")
	logger.logging_tool.log_figure(plt, '', wandb_save_loc = 'eval/psi_tsne')

	phi_psi_tsne = tsne.fit_transform(np.concatenate([a_bs, c_ds, b_s - a_s, d_s - c_s], dim = 0))
	tsne = TSNE()
	plt.clf()
	fig, ax = plt.subplots()
	for x, y, m, c in zip(phi_psi_tsne[:, 0], phi_psi_tsne[:, 1], markers, colors):
		scat = plt.scatter(x, y, marker = m, cmap = c)
	plt.title("Phi(a,b), Phi(c,d), Psi(b) - Psi(a), Psi(d) - Psi(c) TSNE Plot " + str(step))
	plt.xlabel("TSNE Principal Component 1")
	plt.ylabel("TSNE Principal Component 2")
	logger.logging_tool.log_figure(plt, '', wandb_save_loc = 'eval/phi_psi_tsne')

	phi_goal_psi_tsne = tsne.fit_transform(np.concatenate([a_bs - b_bs, c_ds - d_ds, b_s - a_s, d_s - c_s], dim = 0))
	tsne = TSNE()
	plt.clf()
	fig, ax = plt.subplots()
	for x, y, m, c in zip(phi_goal_psi_tsne[:, 0], phi_goal_psi_tsne[:, 1], markers, colors):
		scat = plt.scatter(x, y, marker = m, cmap = c)
	plt.title("Phi(a,b) - Phi(b,b), Phi(c,d) - Phi(d,d), Psi(b) - Psi(a), Psi(d) - Psi(c) TSNE Plot " + str(step))
	plt.xlabel("TSNE Principal Component 1")
	plt.ylabel("TSNE Principal Component 2")
	logger.logging_tool.log_figure(plt, '', wandb_save_loc = 'eval/phi_goal_psi_tsne')

	if train_set:
		beginning = 'train'
	else:
		beginning = 'eval'

'''
	stats = {
	'train_step' : step,
	beginning + '/phi/analogy_error': np.mean(np.linalg.norm(a_bs - c_ds, dim = 1)),
	beginning + '/phi/analogy_error_norm': np.mean(np.abs(np.linalg.norm(a_bs, dim = 1) - np.linalg.norm(c_ds, dim = 1))),
	beginning + '/phi/goal_analogy_error': np.mean(np.linalg.norm((a_bs - b_bs) - (c_ds - d_ds), dim = 1)),
	beginning + '/phi/goal_analogy_error_norm': np.mean(np.abs(np.linalg.norm(a_bs - b_bs, dim = 1) - np.linalg.norm(c_ds - d_ds, dim = 1))),
	beginning + '/psi/abs_error': np.mean(np.linalg.norm((b_s - a_s) - a_bs, dim = 1)),
	beginning + '/psi/abs_error_norm': np.mean(np.abs(np.linalg.norm(b_s - a_s, dim = 1) - np.linalg.norm(a_bs, dim = 1))),
	beginning + '/psi/abs_error_other': np.mean(np.linalg.norm((d_s - c_s) - c_ds, dim = 1)),
	beginning + '/psi/abs_error_other_norm': np.mean(np.abs(np.linalg.norm(d_s - c_s, dim = 1) - np.linalg.norm(c_ds, dim = 1))),
	beginning + '/psi/analogy_error': np.mean(np.linalg.norm((b_s - a_s) - c_ds, dim = 1)),
	beginning + '/psi/analogy_error_norm': np.mean(np.abs(np.linalg.norm(b_s - a_s, dim = 1) - np.linalg.norm(c_ds, dim = 1))),
	beginning + '/psi/analogy_psi_error': np.mean(np.linalg.norm((b_s - a_s) - (d_s - c_s), dim = 1)),
	beginning + '/psi/analogy_psi_error_norm': np.mean(np.abs(np.linalg.norm(b_s - a_s, dim = 1) - np.linalg.norm(d_s - c_s, dim = 1))),
	beginning + '/phi/average_distance': phi_distance_avg,
	beginning + '/phi/goal_average_distance': phi_ground_distance_avg,
	beginning + '/psi/average_distance': psi_distance_avg,
	beginning + '/phi/average_norm_distance': phi_norm_avg,
	beginning + '/phi/goal_average_norm_distance': phi_ground_norm_avg,
	beginning + '/psi/average_norm_distance': psi_norm_avg,
	beginning + '/phi/goal_psi_average_distance': phi_ground_psi_distance_avg,
	beginning + '/phi/goal_psi_average_norm_distance': phi_ground_psi_norm_avg,
	beginning + '/phi/psi_average_distance': phi_psi_distance_avg,
	beginning + '/phi/psi_average_norm_distance': phi_psi_norm_avg,
	beginning + '/psi/analogy_psi_error_norm': np.mean(np.abs(np.linalg.norm(b_s - a_s, dim = 1) - np.linalg.norm(d_s - c_s, dim = 1))),
	beginning + '/phi_analogy_error_cos': np.mean(cosine_sim(a_bs, c_ds)),
	beginning + '/phi_goal_analogy_error_cos': np.mean(cosine_sim(a_bs - b_bs, c_ds - d_ds)),
	beginning + '/psi_abs_error_cos': np.mean(cosine_sim((b_s - a_s), a_bs)),
	beginning + '/psi_abs_error_other_cos': np.mean(cosine_sim((d_s - c_s), c_ds)),
	beginning + '/psi_analogy_error_cos': np.mean(cosine_sim((b_s - a_s), c_ds)),
	beginning + '/psi_analogy_psi_error_cos': np.mean(cosine_sim(b_s - a_s, d_s - c_s)),
	}

	logger.logging_tool.log(stats)
		
'''

