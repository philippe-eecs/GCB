import numpy as np
import torch
'''
Reward Functions, mainly used for relabeling in HER
'''



def point_mass_reward(dense = False, use_state = False, eps = 0.05, reward_scaling = 10):

	def relabel_reward_function(obs, action, next_obs, state, next_state, goal, goal_state, obs_idx, goal_idx):

		#import pdb; pdb.set_trace()

		if dense and use_state:
			reward = np.linalg.norm(goal_state[:2] - next_state[:2])

			if reward < eps:
				reward = 0
			else:
				reward = -reward * reward_scaling

			done_flag = abs(reward) < eps

		elif not dense and use_state:
			reward = np.linalg.norm(goal_state[:2] - next_state[:2])
			if reward < eps:
				reward = 1 * reward_scaling
			else:
				reward = 0

			done_flag = abs(reward) < eps

		else:
			reward = int(obs_idx + 1 == goal_idx) * reward_scaling

			done_flag = (obs_idx + 1 == goal_idx)


		return reward, done_flag

	return relabel_reward_function

def point2d_reward(dense = False, use_state = False, eps = 0.60, reward_scaling = 5):

	def relabel_reward_function(obs, action, next_obs, state, next_state, goal, goal_state, obs_idx, goal_idx):

		if dense and use_state:

			norm = np.linalg.norm(goal_state - next_state, axis=-1)
			reward = norm
			reward = -reward * reward_scaling

			done_flag = norm < eps

		elif not dense and use_state:
			norm = np.linalg.norm(goal_state - next_state, axis=-1)
			reward = norm
			if reward < eps:
				reward = 1 * reward_scaling
			else:
				reward = 0

			done_flag = norm < eps

		else:
			reward = (obs_idx + 1 == goal_idx) * reward_scaling

			done_flag = (obs_idx + 1 == goal_idx)


		return reward, done_flag

	return relabel_reward_function





