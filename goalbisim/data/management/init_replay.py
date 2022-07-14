import torch
import numpy as np
from goalbisim.data.manipulation.transform import initialize_transform



def init_replay(obs_shape, action_shape, device, details, state_shape = None, transforms = True):

	if transforms:
		transform = initialize_transform(details['training_transforms'])
	else:
		transform = None



	replay_args = details['replay_buffer_kwargs']

	if details['replay_buffer_type'] == 'Normal':
		from goalbisim.data.management.replaybuffer import ReplayBuffer

		replay_buffer = ReplayBuffer(obs_shape, action_shape, device = device, transform = transform, **replay_args)

	elif details['replay_buffer_type'] == 'Contrastive':
		from goalbisim.data.management.contrastivereplaybuffer import ContrastiveReplayBuffer

		replay_buffer = ContrastiveReplayBuffer(obs_shape, state_shape, action_shape, device = device, transform = transform, **replay_args)

	elif details['replay_buffer_type'] == 'Goal':
		from goalbisim.data.management.goalreplaybuffer import GoalReplayBuffer

		replay_buffer = GoalReplayBuffer(obs_shape, state_shape, action_shape, device = device, transform = transform, **replay_args)
		
	elif details['replay_buffer_type'] == 'GoalAnalogy':

		from goalbisim.data.management.goalanalogyreplaybuffer import GoalAnalogyReplayBuffer

		replay_buffer = GoalAnalogyReplayBuffer(obs_shape, state_shape, action_shape, device = device, transform = transform, **replay_args)

	elif details['replay_buffer_type'] == 'HER':
		from goalbisim.data.management.herreplaybuffer import HERReplayBuffer
		from goalbisim.data.management.conditionalherreplaybuffer import ConditionalHERReplayBuffer
		from goalbisim.data.management.cpvherreplaybuffer import CPVHERReplayBuffer
		
		relabel_strategy = None

		if details.get('representation_algorithm', None) == 'Ccvae':
			replay_buffer = ConditionalHERReplayBuffer(obs_shape, state_shape, action_shape, reward_strategy = relabel_strategy, device = device, transform = transform, **replay_args)
		elif details['rl_algorithm'] == 'IQL' and details['iql_kwargs'].get('phi_config') == 'cpv':
			replay_buffer = CPVHERReplayBuffer(obs_shape, state_shape, action_shape, reward_strategy = relabel_strategy, device = device, transform = transform, **replay_args)
		else:
			replay_buffer = HERReplayBuffer(obs_shape, state_shape, action_shape, reward_strategy = relabel_strategy, device = device, transform = transform, **replay_args)
	else:
		raise NotImplementedError


	return replay_buffer		