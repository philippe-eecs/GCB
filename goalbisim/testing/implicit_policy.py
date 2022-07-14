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









def plan_action(obs, goal, representation):
	a_b = representation.phi(obs.unsqueeze(0), goal.unsqueeze(0)).detach()
	action = representation.phi.policy_decoder(a_b).detach().cpu().squeeze().numpy()

	return action


def implicit_analogy_test(env, eval_transforms, replay_buffer, representation, step, train_set = False):
	episode_rewards = []
	successes = []
	for i in range(20):
		episode_step, episode_reward, done = 0, 0, False
		success = 0

		obs, goal, extra = env.reset()

		state = extra['state']

		while not done:

			obs = eval_transforms(obs, 'cuda')
			goal = eval_transforms(goal, 'cuda')

			action = plan_action(obs, goal, representation)

			next_obs, reward, is_success, extra = env.step(action)

			next_state = extra['state']

			if episode_step + 1 == env._max_episode_steps:
				done_bool = float(is_success)
				goal_met = 0
				done_trajectory = 1
				done = True
			else:
				done_bool = float(is_success)
				done_trajectory = float(is_success)

			if is_success:
				success = 1
				done = True

			episode_reward += (0.99 ** episode_step) * reward

			obs = next_obs
			state = next_state
			episode_step += 1

		episode_rewards.append(episode_reward)
		successes.append(success)

	if train_set:
		beginning = 'train'
	else:
		beginning = 'eval'

	stats = {
	'train_step' : step,
	beginning + '/implicit_analogy_avg_episode_reward': np.mean(episode_rewards),
	beginning + '/implicit_analogy_avg_success': np.mean(successes)
	}

	logger.logging_tool.log(stats)









