import torch
import numpy as np
import dmc2gym
import time
from goalbisim.utils.misc_utils import set_seed_everywhere, FrameStack, GoalFrameStack
from goalbisim.data.management.replaybuffer import ReplayBuffer
from goalbisim.agents.pixelsac import PixelSACAgent
from goalbisim.agents.goalpixelsac import GoalPixelSACAgent
from goalbisim.representation.base_representation import initialize_representation
from rlkit.core import logger
from goalbisim.testing.eval_rl import evaluate_agent
from goalbisim.testing.eval_rl import evaluate_goal_agent
from goalbisim.utils.misc_utils import eval_mode
from loggers.wandb.wandb_init import setup_logger
from environments.environment_init import load_env
from goalbisim.data.management.init_replay import init_replay
from goalbisim.data.manipulation.transform import initialize_transform
from goalbisim.testing.init_testing import init_testing
from goalbisim.agents.agent_init import agent_initalization
import wandb
import random


def train_representation_offline(details):

    random.seed(details['seed'])
    np.random.seed(details['seed'])
    torch.manual_seed(details['seed'])

    logging_tool = setup_logger(details)
    if details['use_wandb']:
        wandb.log(details) #Logs Details

    env = load_env(details)
    eval_env = load_env(details)

    device = torch.device(details['device'])

    test_functions = init_testing(details)
    env = load_env(details)
    obs_shape = env.observation_space.shape
    eval_transforms = initialize_transform(details['eval_transforms'])

    replay_buffer = init_replay(obs_shape, env.action_space.shape, device, details, state_shape = env.state_space.shape)
    eval_replay_buffer = init_replay(obs_shape, env.action_space.shape, device, details, state_shape = env.state_space.shape)

    #Technically offline replay buffer, with consistent policy, but shouldn't matter...

    if details['training_form'] == 'dataset':
        replay_buffer.load(details['dataset_loc'], start = 0, end = details['number_training_points'])
        eval_replay_buffer.load(details['dataset_loc'], start = details['number_training_points'])
    elif details['training_form'] == 'policy':
        demo_critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details['demo_policy_representation_kwargs'], main = True)
        demo_actor_representation = initialize_representation(obs_shape, env.action_space.shape, device, details['demo_policy_representation_kwargs'])
        demo_target_critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details['demo_policy_representation_kwargs'])
        
        demo_agent = GoalPixelSACAgent(obs_shape, env.action_space.shape, device, eval_transforms, demo_actor_representation, demo_critic_representation, demo_target_critic_representation,
        None, **details['sac_kwargs'])
        load_policy = demo_agent.load(details['policy_loc'], 'best_agent')
        replay_buffer = collect_offline_replay(demo_agent, env, details['discount'], replay_buffer, sample_points = details['sample_points'])
        eval_replay_buffer = collect_offline_replay(demo_agent, env, details['discount'], eval_replay_buffer, sample_points = details['eval_sample_points'])

        del demo_agent

    elif details['training_form'] == 'gail':
        raise NotImplementedError

    else:
        raise NotImplementedError

    critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details, main = True)
    actor_representation = initialize_representation(obs_shape, env.action_space.shape, device, details)
    target_critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details)

    best_critic_representation = initialize_representation(env.observation_space.shape, env.action_space.shape, device, details, main = True)
    best_actor_representation = initialize_representation(env.observation_space.shape, env.action_space.shape, device, details)
    best_target_critic_representation = initialize_representation(env.observation_space.shape, env.action_space.shape, device, details)

    if details['add_her_relabels']:
        replay_buffer.relabel()

    agent = agent_initalization(details, env, device, eval_transforms, actor_representation, critic_representation, target_critic_representation)
    best_agent = agent_initalization(details, env, device, eval_transforms, best_actor_representation, best_critic_representation, best_target_critic_representation)

    if details['use_distractor']:
        from goalbisim.utils.video import VideoDistractor
        train_distractor = VideoDistractor(env, details['distractor_kwargs'], replay_buffer = replay_buffer, ratio_end = .2)
        eval_distractor = VideoDistractor(eval_env, details['distractor_kwargs'], replay_buffer = eval_replay_buffer, ratio_start = .2, ratio_end = .4)

        env.set_distractor(train_distractor)
        eval_env.set_distractor(eval_distractor)

        replay_buffer.overlay(train_distractor)
        eval_replay_buffer.overlay(eval_distractor)


    best_success_rate = 0
    add_to = 0

    for step in range(details['training_iterations']):
        if step % details['eval_freq'] == 0:
            conditional = True if details['representation_algorithm'] == 'Ccvae' else False

            success_rate, video, goalvideo, success_stats, seed_list, samples = evaluate_goal_agent(eval_env, agent, details['discount'], details['num_eval_episodes'], step + add_to, details, conditional = conditional, analogy_goal = details.get('analogy_goal', False))
            rel_success_rate, rel_video, rel_goalvideo, rel_success_stats, rel_seed_list, rel_samples = evaluate_goal_agent(eval_env, best_agent, details['discount'], details['num_eval_episodes'], step + add_to, details, conditional = conditional, analogy_goal = details.get('analogy_goal', False))
            
            if success_rate > rel_success_rate:
                agent.save(logger.get_snapshot_dir(), 'best_agent')
                best_agent.load(logger.get_snapshot_dir(), 'best_agent')

                if best_success_rate < success_rate:
                    best_success_rate = success_rate

                goalvideo.save("eval/goals", step + add_to)
                video.save("eval/video_rollouts", step + add_to)
                logger.logging_tool.log(success_stats)

            else:
                if details['reload_best_agent']:
                    try:
                        agent.load(logger.get_snapshot_dir(), 'best_agent')
                    except:
                        pass
                if details['step_lr']:
                    agent.step_all()
                    
                rel_goalvideo.save("eval/goals", step + add_to)
                rel_video.save("eval/video_rollouts", step + add_to)
                logger.logging_tool.log(rel_success_stats)

                if rel_success_rate > best_success_rate:
                    best_success_rate = rel_success_rate

            stats = {'train_step' : step,
                    'eval/best_sucess': best_success_rate,
                    }
            logger.logging_tool.log(stats)



                

            for test in test_functions:
                try:
                    test(eval_env, eval_transforms, device, replay_buffer, critic_representation, step + add_to, details, train_set = True, eval_replay_buffer = replay_buffer)
                    test(eval_env, eval_transforms, device, replay_buffer, critic_representation, step + add_to, details, eval_replay_buffer = eval_replay_buffer)
                    test(eval_env, eval_transforms, device, replay_buffer, critic_representation, step + add_to, details, eval_replay_buffer = eval_replay_buffer, forced_samples = samples)
                except:
                    pass

            agent.test_representation(eval_replay_buffer, step + add_to)
            logger.logging_tool.record()

        agent.update(replay_buffer, step + add_to)

    add_to = add_to + details['training_iterations']
    ignore_first = True
    total_steps = 0
    total_episode_reward = 0
    total_successes = 0
    #assert replay_buffer.capacity > details['online_training_iterations'] + replay_buffer.idx
    discount = 0.99
    for traj in range(details['online_training_trajectories']):
        if traj % details['eval_traj_freq'] == 0:
            success_rate, video, goalvideo, success_stats, seed_list, samples = evaluate_goal_agent(eval_env, agent, details['discount'], details['num_eval_episodes'], total_steps + add_to, details)
            goalvideo.save("online/eval/goals", total_steps + add_to)
            video.save("online/eval/video_rollouts", total_steps + add_to)
            logger.logging_tool.log(success_stats)
            if success_rate > best_success_rate:
                agent.save(logger.get_snapshot_dir(), 'best_agent')
                #best_agent.load(logger.get_snapshot_dir(), 'best_agent')

            else:
                #try:
                if details['reload_best_agent']:
                    try:
                        agent.load(logger.get_snapshot_dir(), 'best_agent')
                    except:
                        pass
                if details['step_lr']:
                    agent.step_all()
                

            for test in test_functions:
                test(eval_env, eval_transforms, device, replay_buffer, critic_representation, total_steps + add_to, details, train_set = True, eval_replay_buffer = replay_buffer)
                test(eval_env, eval_transforms, device, replay_buffer, critic_representation, total_steps + add_to, details, eval_replay_buffer = eval_replay_buffer)
                test(eval_env, eval_transforms, device, replay_buffer, critic_representation, total_steps + add_to, details, eval_replay_buffer = eval_replay_buffer, forced_samples = samples)

            agent.test_representation(eval_replay_buffer, total_steps + add_to)
            logger.logging_tool.record()

        if not ignore_first:

            total_successes += float(is_success)
            total_episode_reward += float(episode_reward)


            stats = {'train_step' : total_steps,
                'online/train/success': done_bool,
                'online/train/average_episode_reward' : total_episode_reward/traj,
                'online/train/average_success' : total_successes/traj}
            logger.logging_tool.log(stats)

            episode_reward = 0
            episode_step = 0
            reward = 0
            #total
                #logger.dump_tabular(with_prefix=True, with_timestamp=False)

        else:
            ignore_first = False
            #state = extra['state']
            episode_reward = 0
            episode_step = 0
            reward = 0
            #logger.logging_tool.record()


        obs, goal, extra = env.reset()

        state = extra['state']

        done = False

        #ignore_first = True

        while not done:
            
            #episode += 1
            

            with eval_mode(agent): #Gather Data but detach
                action = agent.sample_action(obs, goal)

            curr_reward = reward
            next_obs, reward, is_success, extra = env.step(action)

            next_state = extra['state']

            # allow infinit bootstrap

            if episode_step + 1 == env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
                done_bool = float(is_success)
                done_trajectory = 1
                done = True
            else: 
                done_bool = float(is_success)
                done_trajectory = float(is_success)

            if is_success:
                total_successes += 1
                #total
                done = True 

            episode_reward += reward * (discount ** episode_step)

            replay_buffer.add(obs, state, action, episode_reward, reward, next_obs, next_state, done_bool, done_trajectory, goal)

            for _ in range(details['grad_steps_per_online_step']):
                agent.update(replay_buffer, total_steps + add_to)

            obs = next_obs
            state = next_state
            episode_step += 1
            total_steps += 1

def collect_offline_replay(agent, env, discount, replay_buffer, sample_points = 100000):

    episode, episode_reward, done = 0, 0, True
    episode_step = 0
    ignore_first = True
    start_time = time.time()
    success = 0
    average_reward = 0
    total_steps = 0
    for step in range(sample_points):
        if done:
            print(success)
            total_steps += episode_step
            print(total_steps)
            if agent == 'demo':
                obs, goal, extra = env.demo_reset()
            else:
                obs, goal, extra = env.reset()

            state = extra['state']
            done = False
            average_reward += episode_reward
            episode += 1
            episode_reward = 0

            episode_step = 0
            reward = 0

        if agent == 'demo':
            action = env.get_demo_action()
        else:
            with eval_mode(agent):
                action = agent.select_action(obs, goal, init_obs=init_obs)

        curr_reward = reward
        next_obs, reward, is_success, extra = env.step(action)

        next_state = extra['state']

        if episode_step + 1 == env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
            done_bool = 1 #Not Sure what this should be?
            done_trajectory = 1
            done = True
        else: 
            done_bool = float(is_success)
            done_trajectory = float(is_success)

        if is_success:
            success += 1
            done = True  

        episode_reward += (discount ** episode_step) * reward

        replay_buffer.add(obs, state, action, episode_reward, reward, next_obs, next_state, done_bool, done_trajectory, goal)

        obs = next_obs
        if episode_step == 0:
            init_obs = obs
        state = next_state
        episode_step += 1

    print("Success_Rate of Offline Set: " , success/episode)
    print("Average Reward of Offline Set: ", average_reward/episode)

    return replay_buffer

def collect_analogy_offline_replay(agent, env, discount, analogy_replay_buffer, trajectories = 5000):

    episode, episode_reward, done = 0, 0, True
    episode_step = 0
    ignore_first = True
    start_time = time.time()
    success = 0
    average_reward = 0
    total_steps = 0
    for traj in range(trajectories):
        if agent == 'demo':
            obs, goal, extra = env.demo_reset()
        else:
            obs, goal, extra = env.reset()

        print(traj)
        total_steps += episode_step
        #print(total_steps)
        

        state = extra['state']
        done = False
        average_reward += episode_reward
        episode += 1
        episode_reward = 0

        episode_step = 0
        reward = 0

        while not done:
            

            if agent == 'demo':
                action = env.get_demo_action()
            else:
                with eval_mode(agent):
                    action = agent.select_action(obs, goal)

            curr_reward = reward
            next_obs, reward, is_success, extra = env.step(action)

            next_state = extra['state']

            if episode_step + 1 == env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
                done_bool = 1
                done_trajectory = 1
                done = True
            else: 
                done_bool = float(is_success)
                done_trajectory = float(is_success)

            if is_success:
                success += 1
                done = True  

            episode_reward += (discount ** episode_step) * reward

            analogy_replay_buffer.add(obs, action, reward, next_obs, done_bool, done_trajectory, goal)

            obs = next_obs
            state = next_state
            episode_step += 1

        obs, goal, extra = env.demo_jitter_reset()

        done = False
        episode_step = 0
        reward = 0

        while not done:

            if agent == 'demo':
                action = env.get_demo_action()
            else:
                with eval_mode(agent):
                    action = agent.select_action(obs, goal)

            curr_reward = reward
            next_obs, reward, is_success, extra = env.step(action)

            next_state = extra['state']

            if episode_step + 1 >= env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
                done_bool = 1
                done_trajectory = 1
                done = True
            else: 
                done_bool = float(is_success)
                done_trajectory = float(is_success)

            if is_success:
                done = True  

            #episode_reward += (discount ** episode_step) * reward

            analogy_replay_buffer.add_analogy(obs, action, reward, next_obs, done_bool, done_trajectory, goal)

            obs = next_obs
            state = next_state
            episode_step += 1

    print("Success_Rate of Offline Set: " , success/episode)
    #print("Average Reward of Offline Set: ", average_reward/episode)

    return analogy_replay_buffer
