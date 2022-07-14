import torch
from goalbisim.utils.misc_utils import eval_mode
from goalbisim.utils.video import MultiVideoRecorder
import numpy as np
from rlkit.core import logger
import imageio


def evaluate_agent(env, agent, video, num_episodes):
    # embedding visualization
    obses = []
    values = []
    embeddings = []
    rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with eval_mode(agent): #Might Need Debug...
                action = agent.select_action(obs)

            obs, reward, done, info = env.step(action)

            video.record(env)
            episode_reward += reward

        #video.save('%d.mp4' % step) #Not sure what this does...
        rewards.append(episode_reward)

    return np.array(rewards)


def evaluate_goal_agent(env, agent, discount, num_episodes, step, details, input_seed = None, record_number = 10, conditional = False, analogy_goal = False, fps = 15):
    # embedding visualization
    #goals = []
    video = MultiVideoRecorder(dir_name = details['eval_video_save_dir'], width = 5, fps = fps)
    goalvideo = MultiVideoRecorder(dir_name = details['eval_video_save_dir'], width = 5, fps = 1)
    goalvideo.init(num_trajectories = record_number, max_trajectory_length = 4)
    video.init(num_trajectories = record_number, max_trajectory_length = env._max_episode_steps + 1)

    #if distractor is not None:
        #env.set

    obses = []
    values = []
    embeddings = []
    rewards = []
    successes = []
    final_distance = []

    seeds = []

    gather_samples = [[], []]

    records = 0

    for i in range(num_episodes):
        if input_seed is not None:
            obs, goal, extra = env.reset(seed = input_seed[i])
            if analogy_goal:
                old_obs = obs
                old_goal = goal
                obs, extra = env.jitter_reset(seed = input_seed[i])
                goal = [old_obs, old_goal]
            seeds.append(input_seed[i])
        else:
            obs, goal, extra = env.reset()
            if analogy_goal:
                old_obs = obs
                old_goal = goal
                obs, extra = env.jitter_reset()
                goal = [old_obs, old_goal]
            seeds.append(extra['seed'])
        if records < record_number:
            if analogy_goal:
                for _ in range(2):
                    goalvideo.record(goal[0])
                for _ in range(2):
                    goalvideo.record(goal[1])
            else:
                for _ in range(4):
                    goalvideo.record(goal)
            goalvideo.step()
            video.record(obs)
            #records += 1
            
        state = extra['state']
        done = False
        episode_reward = 0
        episode_step = 0
        init_obs = obs
        if analogy_goal:
            gather_samples[0].append(obs)
            gather_samples[1].append(goal[1])
        else:
            gather_samples[0].append(obs)
            gather_samples[1].append(goal)

        while not done:
            with eval_mode(agent): #Might Need Debug...
                if conditional:
                    concat_obs = np.concatenate((obs, init_obs), axis=0)
                    concat_goal = np.concatenate((goal, init_obs), axis=0)
                    action = agent.sample_action(concat_obs, concat_goal, init_obs=init_obs)
                else:
                    action = agent.sample_action(obs, goal, init_obs=init_obs)

            obs, reward, done, info = env.step(action)
            if records < record_number:
                video.record(obs)

            state = info['state']

            #video.record(env)
            if episode_step + 1 >= env._max_episode_steps:
                done_bool = 1
                done = True
                success = float(reward > 0.01)
                #done_trajectory = 1
            else: 
                done_bool = 0
                success = float(reward > 0.01)
                if success:
                    done_bool = 1
                    done = True
                else:
                    done = False
                #done_trajectory = float(done)

            episode_reward += reward * (discount ** episode_step)

            episode_step += 1

        if records < record_number:
            video.step()
            records += 1
        successes.append(success)
        rewards.append(episode_reward)

    #video.save("eval/video_rollouts", step)
    #goalvideo.save("eval/goals", step)

    rewards, success = np.array(rewards), np.array(successes)

    stats = {'train_step' : step,
    'eval/avg_sucess': np.mean(success),
    'eval/std_sucess' : np.std(success),
    'eval/avg_episode_reward' : np.mean(rewards),
    'eval/std_episode_reward' : np.std(rewards),
    }
    #logger.logging_tool.log(stats)

    #gather_samples[0] = np.concatenate(gather_samples[0])
    #gather_samples[1] = np.concatenate(gather_samples[1])

    return np.mean(success), video, goalvideo, stats, seeds, gather_samples