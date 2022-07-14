


def load_env(details):
	env_args = details['env_kwargs']

	if env_args['package'] == 'multiworld':

		domain_name = env_args['domain_name']

		if domain_name == 'point2d':
			import multiworld
			from multiworld.envs.pygame import Point2DEnv
			env = Point2DEnv(**env_args['domain_kwargs']) #Goal Conditioned Enviornment

			if env_args['frame_stack_count'] > 1:
				raise NotImplementedError
				from rlkit.envs.wrappers.stack_observation_env import StackObservationEnv

				env = GoalStackObservationEnv(env, stack_obs = env_args['frame_stack_count'])

	elif env_args['package'] == 'dm_control':
		import dmc2gym
		import dm_control
		env = dmc2gym.make(**details['env_kwargs'])

		if env_args['frame_stack_count'] > 1:
				from goalbisim.utils.misc_utils import set_seed_everywhere, FrameStack, GoalFrameStack

				env = GoalFrameStack(env, k=details['frame_stack_count'])

	else:
		raise NotImplementedError

	return env

