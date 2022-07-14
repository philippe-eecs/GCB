import torch
import torch.nn as nn
import numpy as np


class BrownianRandomPolicy(nn.Module):
	'''
	Random selects action based on normal distribution,
	Where mean is set as the prior sampled action and variance is fixed by user
	Initial mean is set by user as well
	'''
	def __init__(self, action_dimension = 2, mean = 0, variance = 0.1, min_value = -1, max_value = 1, seed = None):
		self.action_dimension = action_dimension
		self.original_mean = mean
		self.mean = mean
		self.variance = variance
		if seed:
			np.random.seed(seed)
		else:
			np.random.seed()

		self.min_value = min_value
		self.max_value = max_value


	def __call__(self, state, goal):
		sampled_action = np.random.normal(self.mean, self.variance, self.action_dimension)
		sampled_action = np.clip(sampled_action, self.min_value, self.max_value)
		self.mean = sampled_action
	
		return sampled_action

	def reset(self):
		np.random.seed()
		self.mean = self.original_mean