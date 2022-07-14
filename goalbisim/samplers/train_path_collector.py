import torch
import numpy as np
import wandb
from goalbisim.logging.logging import logger





def collect_gcrl_trajectory(env, agent, cycle, episode, replay_buffer, update_per_step = False):

	episode_step, episode_reward, done = 0, 0, False

	obs, goal, extra = env.reset()

	state = extra['state']

	while not done:

		with eval_mode(agent): #Gather Data but detach
			action = agent.sample_action(obs, goal)

		next_obs, reward, done, extra = env.step(action)

		next_state = extra['state']

		if episode_step + 1 == env._max_episode_steps:
			goal_met = 0
			done_trajectory = 1
		else:
			goal_met = float(done)
			done_trajectory = float(done)

		episode_reward += reward

		replay_buffer.add(obs, state, action, episode_reward, reward, next_obs, next_state, goal_met, done_trajectory, goal)

		if update_per_step:
			agent.update(replay_buffer, cycle)

		obs = next_obs
		state = next_state
		episode_step += 1

	stats ={
	'train_cycle' : cycle,
	'train/episode': episode,
	'train/final_reward': reward,
	'train/success' : goal_met,
	'train/episode_reward' : episode_reward
	}

	wandb.log(stats)
	logger.record_dict(stats)

	return reward, goal_met, episode_reward, episode_step








def collect_gcrl_trajectories(env, agent, total_steps, cycle, episode, replay_buffer, update_per_step = False):

	current_steps = 0
	current_episode = 0

	rewards = []
	success = []
	episode_rewards = []

	while current_steps < total_steps:

		current_episode += 1

		reward, goal_met, episode_reward, episode_step = collect_gcrl_trajectory(env, agent, cycle, episode + current_episode, replay_buffer, update_per_step)

		rewards.append(reward)
		success.append(goal_met)
		episode_rewards.append(episode_reward)

		current_steps += episode_step

		

	stats = {
	'train_cycle' : cycle,
	'train/avg_final_reward': np.average(rewards),
	'train/avg_success' : np.average(success),
	'train/avg_episode_reward' : np.average(episode_rewards)
	}

	wandb.log(stats)
	logger.record_dict(stats)

	return episode + current_episode

