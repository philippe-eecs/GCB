import roboverse as rv
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
#from IPython.display import clear_output
from roboverse.bullet.misc import load_obj, quat_to_deg, draw_bbox
import time
plt.ion()

# quat_dict={'mug': [0, 0, 0, 1], 'long_sofa': [0, 0, 0, 1], 'camera': [-1, 0, 0, 0], 'grill_trash_can': [0, 0, 0, 1], 'beer_bottle': [0, 0, 1, 1]}

# Variables to define!
DoF = 3 # (3, 4, 6)
num_timesteps = 75
object_subset = 'train' # (all, train, test)
task = 'goal_reaching' # (pickup, goal_reaching)
randomize = False
# Variables to define!

# Set Up Enviorment
#spacemouse = rv.devices.SpaceMouse(DoF=DoF)
#state_env = rv.make('SawyerRigAffordances-v0', spawn_prob=1.0, test_env=True, gui=True)
state_env = rv.make('SawyerRigMultiobjTray-v0', random_color_p=0.0, object_subset=['grill_trash_can'], gui=True)
# state_env = rv.make('SawyerRigMultiobjTray-v0', gui=True, DoF=DoF, object_subset=object_subset,
# 		quat_dict={}, task=task, randomize=randomize, visualize=False)
imsize = state_env.obs_img_dim
imlength = imsize * imsize * 3

demo_save_path = "/Users/sasha/Desktop/spacemouse/demo_data/{0}_{1}.pkl".format(object_subset, time.time())
recon_save_path = "/Users/sasha/Desktop/spacemouse/recon_data/{0}_{1}.npy".format(object_subset, time.time())
num_trajectories = 0

renderer_kwargs=dict(
	create_image_format='HWC',
	output_image_format='CWH',
	width=imsize,
	height=imsize,
	flatten_image=True)

renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
env = InsertImageEnv(state_env, renderer=renderer)
imlength = env.obs_img_dim * env.obs_img_dim * 3
act_dim = env.action_space.shape[0]
# Set Up Enviorment

def get_empty_traj_dict():
	return {
		'observations': [],
		'next_observations': [],
		'actions': np.zeros((num_timesteps, act_dim), dtype=np.float),
		'rewards': np.zeros((num_timesteps), dtype=np.float),
		'terminals': np.zeros((num_timesteps), dtype=np.uint8),
		'agent_infos': np.zeros((num_timesteps), dtype=np.uint8),
		'env_infos': np.zeros((num_timesteps), dtype=np.uint8),
	}

def get_empty_recon_dict():
	return {
		'observations': [],
		'env': [],
	}

def get_recon_image(env):
	return np.uint8(env.render_obs().transpose()).reshape(1, -1)

def combine_datasets(name_1, name_2, new_name='comb'):
	new_recon_path = '/Users/sasha/Desktop/spacemouse/recon_data/' + new_name + '.npy'
	new_demo_path = '/Users/sasha/Desktop/spacemouse/demo_data/' + new_name + '.pkl'

	recon_path_1 = '/Users/sasha/Desktop/spacemouse/recon_data/' + name_1 + '.npy'
	demo_path_1 = '/Users/sasha/Desktop/spacemouse/demo_data/' + name_1 + '.pkl'

	recon_path_2 = '/Users/sasha/Desktop/spacemouse/recon_data/' + name_2 + '.npy'
	demo_path_2 = '/Users/sasha/Desktop/spacemouse/demo_data/' + name_2 + '.pkl'
	
	recon_dataset_1 = np.load(open(recon_path_1, "rb"), allow_pickle=True).item()
	demo_dataset_1 = pkl.load(open(demo_path_1, "rb"))

	recon_dataset_2 = np.load(open(recon_path_2, "rb"), allow_pickle=True).item()
	demo_dataset_2 = pkl.load(open(demo_path_2, "rb"))
	
	recon_dataset_1['observations'] = recon_dataset_1['observations'].reshape(-1, num_timesteps, imlength)
	recon_dataset_1['env'] = recon_dataset_1['env'].reshape(-1, imlength)

	comb_recon_dataset = {'observations': np.concatenate([recon_dataset_1['observations'], recon_dataset_2['observations']], axis=0),
						'env': np.concatenate([recon_dataset_1['env'], recon_dataset_2['env']], axis=0)}

	comb_demo_dataset = demo_dataset_1 + demo_dataset_2

	# Save Recon Data
	file = open(new_recon_path, 'wb')
	np.save(file, comb_recon_dataset)
	file.close()
	
	# Save Demo Data
	file = open(new_demo_path, 'wb')
	pkl.dump(comb_demo_dataset, file)
	file.close()

	

# def get_dataset_objects():

# 	if extend_dataset_path is False:
# 		return get_empty_recon_dict(), []
	
# 	recon_path = '/Users/sasha/Desktop/spacemouse/recon_data/' + extend_dataset_path + '.npy'
# 	demo_path = '/Users/sasha/Desktop/spacemouse/demo_data/' + extend_dataset_path + '.pkl'
	
# 	recon_dataset = np.load(open(recon_path, "rb"), allow_pickle=True).item()
# 	demo_dataset = pkl.load(open(demo_path, "rb"))
	
# 	recon_dataset['observations'] = recon_dataset['observations'].reshape(-1, num_timesteps, imlength)
# 	recon_dataset['env'] = recon_dataset['env'].reshape(-1, imlength)

# 	recon_dataset['observations'] = [recon_dataset['observations'][i].reshape(1, num_timesteps, imlength)
# 			for i in range(recon_dataset['observations'].shape[0])]
# 	recon_dataset['env'] = [recon_dataset['env'][i].reshape(1, imlength)
# 			for i in range(recon_dataset['env'].shape[0])]

# 	return recon_dataset, demo_dataset

def recompute_rewards(trajectory):
	final_state = trajectory['next_observations'][-1]['state_observation']
	for j in range(num_timesteps):
		trajectory['observations'][j]['state_desired_goal'] = final_state
		trajectory['next_observations'][j]['state_desired_goal'] = final_state
		trajectory['rewards'][j] = state_env.compute_reward(
					trajectory['observations'][j],
					trajectory['actions'][j],
					trajectory['next_observations'][j],
					trajectory['next_observations'][j])


def save_datasets():
	curr_recon_dataset = {'observations': np.concatenate(recon_dataset['observations'], axis=0),
						'env': np.concatenate(recon_dataset['env'], axis=0)}

	# Save Recon Data
	file = open(recon_save_path, 'wb')
	np.save(file, curr_recon_dataset)
	file.close()
	
	# Save Demo Data
	file = open(demo_save_path, 'wb')
	pkl.dump(demo_dataset, file)
	file.close()

def render():
	clear_output(wait=True)
	img = env.render_obs()
	plt.imshow(img)
	plt.show()
	plt.pause(0.01)

def get_and_process_response(trajectory, traj_images):
	response = input(
			'Enter: Add trajectory to both datasets \
			\n D: Add trajectory to demo dataset \
			\n R: Add trajectory to reconstruction dataset \
			\n S: Skip trajectory \
			\n Q: Quit \n')

	add_both = (response == '')
	end = 'Q' in response

	if 'D' in response or add_both:
		# Save To Demo Dataset
		recompute_rewards(trajectory)
		demo_dataset.append(trajectory)
		print('Returns:', sum(trajectory['rewards']))
	
	if 'R' in response or add_both:
		# Save To Reconstruction Dataset
		recon_dataset['env'].append(env_image)
		traj_images = np.concatenate(traj_images, axis=0)
		traj_images = traj_images.reshape(1, num_timesteps, imlength)
		recon_dataset['observations'].append(traj_images)

	save_datasets()
	return end

def rollout_trajectory():
	trajectory = get_empty_traj_dict()

	#env.reset()
	env_image, traj_images = get_recon_image(env), []

	env.demo_reset()

	print(env.goal_pos)
	for j in tqdm(range(num_timesteps)):
		traj_images.append(get_recon_image(env))	
		action = env.get_demo_action(test=True)

		# ee_pos = env.get_end_effector_pos()
		# target_pos = env.get_object_midpoint('obj')
		# achieved_goal = np.linalg.norm(env.goal_pos - target_pos) < 0.05
		# aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.04
		# aligned_goal = np.linalg.norm(env.goal_pos[:2] - target_pos[:2]) < 0.04
		# enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.025
		# above = ee_pos[2] > -0.3

		# if not aligned and not above:
		# 	print('Stage 1')
		# 	action = (target_pos - ee_pos) * 3.0
		# 	action[2] = 1
		# 	grip = -1.
		# elif not aligned:
		# 	print('Stage 2')
		# 	action = (target_pos - ee_pos) * 3.0
		# 	action[2] = 0.
		# 	action *= 3.0
		# 	grip = -1.
		# elif aligned and not enclosed:
		# 	print('Stage 3')
		# 	action = target_pos - ee_pos
		# 	action[2] -= 0.03
		# 	action *= 3.0
		# 	action[2] *= 2.0
		# 	grip = -1.
		# elif enclosed and grip < 1:
		# 	print('Stage 4')
		# 	action = target_pos - ee_pos
		# 	action[2] -= 0.03
		# 	action *= 3.0
		# 	action[2] *= 2.0
		# 	grip += 0.5
		# else:
		# 	print('Stage 5')
		# 	action = env.goal_pos - ee_pos
		# 	action[2] *= 2.0
		# 	action *= 3.0
		# 	grip = 1.

		# if aligned_goal:
		# 	grip = -1.

		# action = np.append(action, [grip])
		# action = np.random.normal(action, 0.1)
		# action = np.clip(action, a_min=-1, a_max=1)

		#action = spacemouse.get_action()

		observation = env.get_observation()
		next_observation, reward, done, info = env.step(action)
		print('Reward: ', reward)

		trajectory['observations'].append(observation)
		trajectory['actions'][j, :] = action
		trajectory['next_observations'].append(next_observation)
		trajectory['rewards'][j] = reward
	return trajectory, env_image, traj_images


recon_dataset = get_empty_recon_dict()
demo_dataset = []

#combine_datasets('train_old', 'train_new')

while True:
	print("Trajectory Number:", num_trajectories)
	trajectory, env_image, traj_images = rollout_trajectory()
	end = get_and_process_response(trajectory, traj_images)
	num_trajectories += 1
	if end: break