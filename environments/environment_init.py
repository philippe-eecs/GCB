


def load_env(details):
	env_args = details['env_kwargs']

	if env_args['package'] == 'multiworld':

		domain_name = env_args['domain_name']

		if domain_name == 'point2d':
			import multiworld
			from multiworld.envs.pygame.point2d import Point2DEnv
			env = Point2DEnv(**env_args['domain_kwargs']) #Goal Conditioned Enviornment

			if env_args['frame_stack_count'] > 1:
				raise NotImplementedError
				from rlkit.envs.wrappers.stack_observation_env import StackObservationEnv

				env = GoalStackObservationEnv(env, stack_obs = env_args['frame_stack_count'])
		else:
			raise NotImplementedError

	elif env_args['package'] == 'dm_control':
		import dmc2gym
		import dm_control
		env = dmc2gym.make(**details['env_kwargs']['domain_kwargs'])

		#if env_args['frame_stack_count'] > 1:
		from environments.dmcontrol.goaldmcwrapper import GoalDMCEnvWrapper
		env = GoalDMCEnvWrapper(env, k=env_args['frame_stack_count'])

	elif env_args['package'] == 'roboverse':
		from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
		import roboverse
		domain_name = env_args['domain_name']
		from environments.roboverse.sawyer_env_wrapper import SawyerWrapper

		if domain_name == 'sawyer_rig_v0':
			from roboverse.envs.sawyer_rig_affordances_v0 import SawyerRigAffordancesV0
			env = roboverse.make('SawyerRigAffordances-v0', **env_args['domain_kwargs'])
			imsize = env.obs_img_dim

			renderer_kwargs=dict(
				create_image_format='HWC',
				output_image_format='CWH',
				width=imsize,
				height=imsize,
				flatten_image=False,)
			renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
			env = InsertImageEnv(env, renderer=renderer)

			#env = SawyerRigMultiobjTrayV0(**env['domain_kwargs'])
		elif domain_name == 'sawyer_rig_v1':
			from roboverse.envs.sawyer_rig_affordances_v0 import SawyerRigAffordancesV0
			env = roboverse.make('SawyerRigAffordances-v1', **env_args['domain_kwargs'])
			imsize = env.obs_img_dim

			renderer_kwargs=dict(
				create_image_format='HWC',
				output_image_format='CWH',
				width=imsize,
				height=imsize,
				flatten_image=False,)
			renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
			env = InsertImageEnv(env, renderer=renderer)

		elif domain_name == 'sawyer_rig_v2':
			#from roboverse.envs.sawyer_rig_affordances_v0 import SawyerRigAffordancesV0
			env = roboverse.make('SawyerRigAffordances-v2', **env_args['domain_kwargs'])
			imsize = env.obs_img_dim

			renderer_kwargs=dict(
				create_image_format='HWC',
				output_image_format='CWH',
				width=imsize,
				height=imsize,
				flatten_image=False,)
			renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
			env = InsertImageEnv(env, renderer=renderer)

		elif domain_name == 'sawyer_rig_v4':
			from roboverse.envs.sawyer_rig_affordances_v4 import SawyerRigAffordancesV4
			env = roboverse.make('SawyerRigAffordances-v4', **env_args['domain_kwargs'])
			imsize = env.obs_img_dim

			renderer_kwargs=dict(
				create_image_format='HWC',
				output_image_format='CWH',
				width=imsize,
				height=imsize,
				flatten_image=False,)
			renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
			env = InsertImageEnv(env, renderer=renderer)

		elif domain_name == 'sawyer_rig_v5':
			from roboverse.envs.sawyer_rig_affordances_v5 import SawyerRigAffordancesV5
			env = roboverse.make('SawyerRigAffordances-v5', **env_args['domain_kwargs'])
			imsize = env.obs_img_dim

			renderer_kwargs=dict(
				create_image_format='HWC',
				output_image_format='CWH',
				width=imsize,
				height=imsize,
				flatten_image=False,)
			renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
			env = InsertImageEnv(env, renderer=renderer)

		else:
			raise NotImplementedError



		env = SawyerWrapper(env, env_args['frame_stack_count'], env_args['action_repeat'])

	else:
		raise NotImplementedError

	return env

