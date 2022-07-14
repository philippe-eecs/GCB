import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wandb
from rlkit.core import logger


def state_regression_test(env, eval_transforms, replay_buffer, representation, step, details, count = 5000, alpha = 1, train_set = False, eval_replay_buffer = None):
	from sklearn.linear_model import Ridge

	model_psi_state = Ridge(alpha = alpha)
	model_phi_state = Ridge(alpha = alpha)

	model_phi_action = Ridge(alpha = alpha)


	model_phi_reward = Ridge(alpha = alpha)
	model_psi_reward = Ridge(alpha = alpha)
	#model_phi_action = Ridge(alpha = alpha)

	if train_set:
		beginning = 'train'
	else:
		beginning = 'eval'

	

	z_psis = []
	z_phis = []
	goal_phis = []
	states = []
	rewards = []
	rtgs = []
	actions = []

	for i in range(count // 256):
		obses, action, rtg, reward, next_obses, not_dones, goals, kwargs = eval_replay_buffer.sample(batch_size = 256, fetch_states = True)

		z_phi = representation.phi(obses, goals).detach().cpu().numpy()
		goal_phi = representation.phi(goals, goals).detach().cpu().numpy()
		try:
			z_psi = representation.psi(obses).detach().cpu().numpy()
		except:
			z_psi = z_phi.copy()
		
		z_psis.append(z_psi)
		z_phis.append(z_phi)
		rtgs.append(rtg.cpu().numpy())
		goal_phis.append(goal_phi)
		states.append(kwargs['states'])
		rewards.append(reward.cpu().numpy())
		actions.append(action.cpu().numpy())

	#import pdb; pdb.set_trace()
	z_psis = np.concatenate(z_psis, dim = 0)
	z_phis = np.concatenate(z_phis, dim = 0)
	goal_phis = np.concatenate(goal_phis, dim = 0)
	rtgs = np.concatenate(rtgs, dim = 0)
	states = np.concatenate(states, dim = 0)
	rewards = np.concatenate(rewards, dim = 0)
	actions = np.concatenate(actions, dim = 0)


	#import pdb; pdb.set_trace()
	#actions = np.concatenate(actions, dim = 0)


	model_psi_state.fit(z_psis[int(count * .1):], states[int(count * .1):])
	model_phi_state.fit(z_phis[int(count * .1):], states[int(count * .1):])

	model_phi_action.fit(z_phis[int(count * .1):], actions[int(count * .1):])

	model_phi_reward.fit(z_phis[int(count * .1):], rewards[int(count * .1):])
	model_psi_reward.fit(z_psis[int(count * .1):], rewards[int(count * .1):])

	training_error_psi = model_psi_state.score(z_psis[int(count * .1):], states[int(count * .1):])
	training_error_phi = model_phi_state.score(z_phis[int(count * .1):], states[int(count * .1):])

	training_error_phi_action = model_phi_action.score(z_phis[int(count * .1):], actions[int(count * .1):])

	training_error_phi_reward = model_phi_reward.score(z_phis[int(count * .1):], rewards[int(count * .1):])
	training_error_psi_reward = model_psi_reward.score(z_psis[int(count * .1):], rewards[int(count * .1):])

	test_error_psi = model_psi_state.score(z_psis[:int(count * .1)], states[:int(count * .1)])
	test_error_phi = model_phi_state.score(z_phis[:int(count * .1)], states[:int(count * .1)])

	test_error_phi_action = model_phi_action.score(z_phis[:int(count * .1)], actions[:int(count * .1)])

	test_error_phi_reward = model_phi_reward.score(z_phis[:int(count * .1)], rewards[:int(count * .1)])
	test_error_psi_reward = model_psi_reward.score(z_psis[:int(count * .1)], rewards[:int(count * .1)])

	stats = {
	'train_step' : step,
	beginning + '/training_error_psi_state_regression': training_error_psi,
	beginning + '/training_error_phi_state_regression': training_error_phi,
	beginning + '/phi_goal_abs': np.mean(np.abs(np.linalg.norm(z_phis - goal_phis, dim = 1) - (-rtgs.squeeze()))),
	beginning + '/phi_goal_abs_l1': np.mean(np.abs(np.linalg.norm(z_phis - goal_phis, ord = 1, dim = 1) - (-rtgs.squeeze()))),
	beginning + '/training_error_psi_reward_regression': training_error_psi_reward,
	beginning + '/training_error_phi_reward_regression': training_error_phi_reward,
	beginning + '/test_error_psi_state_regression': test_error_psi,
	beginning + '/test_error_phi_state_regression': test_error_phi,
	beginning + '/test_error_phi_action_regression' : test_error_phi_action,
	beginning + '/training_error_phi_action_regression' : test_error_phi_action,
	beginning + '/test_error_phi_reward_regression': test_error_phi_reward,
	beginning + '/test_error_psi_reward_regression': test_error_psi_reward
	}

	logger.logging_tool.log(stats)
		





