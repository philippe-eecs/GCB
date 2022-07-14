import torch
#from goalbisim.rewards.base import RewardFn

#TODO: Determine if you want to figure out which reward to apply here, or in the general function...
#Which ever is cleaner
class MetricReward(torch.nn.Module):
	def __init__(self , parameters):
		super().__init__()
		self.temporal = parameters['temporal_cost']
		if self.temporal:
			self.temporal_scaling = parameters['temporal_scaling']

	def forward(self, inputs, outputs):
		s1 = outputs['s1']
		g1 = outputs['g1']
		td1 = inputs['td1']
		r1 = torch.norm(s1 - g1, dim = 1)
		if self.temporal:
			td1 = self.temporal_scaling(td1)
			r1 = torch.where(torch.isnan(td1), r1, td1)

		s2 = outputs['s2']
		g2 = outputs['g2']
		td2 = inputs['td2']
		r2 = torch.norm(s2 - g2, dim = 1)
		if self.temporal:
			td2 = self.temporal_scaling(td2)
			r2 = torch.where(torch.isnan(td2), r2, td2)
			

		rewards = {'r1' : r1, 'r2' : r2}
		return rewards


class L2MetricReward(MetricReward):

	#Confusing exactly what "rewards" means when it really isn't a reward but a cost
	def forward(self, inputs, outputs):
		rewards = super().forward(inputs, outputs)
		return rewards


class BallMetricReward(MetricReward):

	def __init__(self, parameters):
		super().__init__()
		self.epsilon = parameters['reward_epsilon']
		self.goal_reward = parameters['goal_reward']

	#Confusing exactly what "rewards" means when it really isn't a reward but a cost
	def forward(inputs, outputs, **kwargs):
		s1 = outputs['s1']
		g1 = outputs['g1']
		td1 = inputs['td1']
		r1 = torch.norm(s1 - g1, dim = 1)
		r1 = torch.where(r1 < self.epsilon, torch.zeros_like(distances), \
				torch.ones_like(r1) * self.goal_reward)
		if self.temporal:
			td1 = self.temporal_scaling(td1)
			r1 = torch.where(torch.isnan(td1), r1, td1)

		s2 = outputs['s2']
		g2 = outputs['g2']
		td2 = inputs['td2']
		r2 = torch.norm(s2 - g2, dim = 1)
		r2 = torch.where(r2 < self.epsilon, torch.zeros_like(distances), \
				torch.ones_like(r2) * self.goal_reward)
		if self.temporal:
			td2 = self.temporal_scaling(td2)
			r2 = torch.where(torch.isnan(td2), r2, td2)
			

		rewards = {'r1' : r1, 'r2' : r2}
		return rewards