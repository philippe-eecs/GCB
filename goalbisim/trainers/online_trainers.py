import torch
import numpy as np
import dmc2gym
import time
from goalbisim.utils.misc_utils import set_seed_everywhere, FrameStack, GoalFrameStack
from goalbisim.data.management.replaybuffer import ReplayBuffer
from goalbisim.agents.pixelsac import PixelSACAgent
from goalbisim.agents.goalpixelsac import GoalPixelSACAgent
from goalbisim.representation.base_representation import initialize_representation
from goalbisim.testing.eval_rl import evaluate_agent
from goalbisim.testing.eval_rl import evaluate_goal_agent
#from goalbisim.utils.video import VideoRecorder
from goalbisim.utils.misc_utils import eval_mode
from goalbisim.data.management.init_replay import init_replay
from goalbisim.data.manipulation.transform import initialize_transform
from loggers.wandb.wandb_init import setup_logger
from environments.environment_init import load_env
from goalbisim.testing.init_testing import init_testing
from goalbisim.agents.agent_init import agent_initalization
from rlkit.core import logger
import random
import wandb


def train_representation_online(details):
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

    obs_shape = env.observation_space.shape
    eval_transforms = initialize_transform(details['eval_transforms'])
    replay_buffer = init_replay(obs_shape, env.action_space.shape, device, details, state_shape = env.state_space.shape)

    critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details, main = True)
    actor_representation = initialize_representation(obs_shape, env.action_space.shape, device, details)
    target_critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details)

    agent = agent_initalization(details, env, device, eval_transforms, actor_representation, critic_representation, target_critic_representation)

    best_success_rate = 0
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    discount = details['discount']
    for step in range(details['training_iterations']):
        if done:
            # evaluate agent periodically
            if episode % details['eval_freq'] == 0:
                conditional = True if details['representation_algorithm'] == 'Ccvae' else False
                success_rate, video, goalvideo, success_stats, seed_list, samples = evaluate_goal_agent(eval_env, agent, details['discount'], details['num_eval_episodes'], step , details, conditional = conditional, fps = 50)
                goalvideo.save("eval/goals", step)
                video.save("eval/video_rollouts", step )
                logger.logging_tool.log(success_stats)

                if success_rate > best_success_rate:
                    agent.save(logger.get_snapshot_dir(), 'best_agent')
                    best_success_rate = success_rate
                else:
                    if details['reload_best_agent']:
                        try:
                            agent.load(logger.get_snapshot_dir(), 'best_agent')
                        except:
                            pass
                    if details['step_lr']:
                        agent.step_all()

                if episode != 0:
                    for test in test_functions:
                        try:
                            test(eval_env, eval_transforms, device, replay_buffer, critic_representation, step , details, eval_replay_buffer = replay_buffer)
                        except:
                            pass

                    agent.test_representation(replay_buffer, step)

                logger.logging_tool.record()

            obs, goal, extra = env.reset()
            state = extra['state']
            done = False

            try:
                stats = {'train_step' : step,
                    'train/episode'   : episode,
                    'train/episode_reward' : episode_reward}

                logger.logging_tool.log(stats)
            except:
                pass

            episode += 1
            episode_reward = 0
            episode_step = 0
            reward = 0

            

            
        print(step)

        # sample action for data collection
        if step < details['initial_exploration_steps']:
            action = env.action_space.sample() #Random Sampling... Brownian Motion Might Be needed
            #TODO: Customizable init steps, obv not important for pixelsac but super important for goalbisim
        else:
            with eval_mode(agent): #Gather Data but detach
                action = agent.sample_action(obs, goal)

        # run training update
        if step >= details['initial_exploration_steps']:
            num_updates = 100 if step == details['initial_exploration_steps'] else 1 #updates every transition... maybe overkill
            for _ in range(num_updates):
                agent.update(replay_buffer, step)

        curr_reward = reward
        next_obs, reward, done_flag, extra = env.step(action)

        if reward > 0.05:
            is_success = True
        else:
            is_success = False

        next_state = extra['state']

        # allow infinit bootstrap

        if episode_step + 1 >= env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
            done_bool = 1
            done_trajectory = 1
            done = True
        else:
            if is_success:
                done_bool = 1
                done_trajectory = 1
                done = True
            else:
                done_bool = 0
                done_trajectory = 0
                done = False

        episode_reward += reward * (discount ** episode_step)

        replay_buffer.add(obs, state, action, episode_reward, reward, next_obs, next_state, done_bool, done_trajectory, goal)

        obs = next_obs
        next_state = state
        episode_step += 1









































def representation_goalsac_online_rl(details):
    logging_tool = setup_logger(details)
    wandb.log(details)
    set_seed_everywhere(details['seed']) #good to use....
    env = load_env(details)
    eval_env = load_env(details)

    device = torch.device('cuda')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    replay_buffer = init_replay(env.observation_space.shape, env.action_space.shape, device, details, state_shape = env.state_space.shape)
    
    critic_representation = initialize_representation(env.observation_space.shape, env.action_space.shape, details, main = True)
    actor_representation = initialize_representation(env.observation_space.shape, env.action_space.shape, details)
    target_critic_representation = initialize_representation(env.observation_space.shape, env.action_space.shape, details)

    eval_transforms = initialize_transform(details['eval_transforms'])

    agent = GoalPixelSACAgent(env.observation_space.shape, env.action_space.shape, device, eval_transforms, actor_representation, critic_representation, target_critic_representation,
        None, **details['sac_kwargs'])

    video = VideoRecorder(logger.get_snapshot_dir())

    from goalbisim.testing.init_testing import init_testing


    test_functions = init_testing(details)
    #logger 
    #Maybe Change?, Or We can use interally either way

    episode, episode_reward, done = 0, 0, True
    ignore_first = True
    discount = details['discount']
    start_time = time.time()
    best_reward = None
    reward = 10
    for step in range(details['num_train_steps']):
        if done:
            stats = {'train_step' : step,
            'train/duration' : time.time() - start_time}
            logger.logging_tool.log(stats)
            

            # evaluate agent periodically
            if episode % details['eval_freq'] == 0:
                rewards, success = evaluate_goal_agent(eval_env, agent, discount, details['num_eval_episodes'], step)
                stats = {'train_step' : step,
                'eval/episode' : episode,
                'eval/avg_sucess': np.mean(success),
                'eval/std_sucess' : np.std(success),
                'eval/avg_episode_reward' : np.mean(rewards),
                'eval/std_episode_reward' : np.std(rewards),
                }

                logger.logging_tool.log(stats)
                if episode > 1:
                    for test in test_functions:
                        test(eval_env, eval_transforms, replay_buffer, critic_representation, step)
                
                if best_reward is None or np.mean(rewards) > best_reward:
                    agent.save(logger.get_snapshot_dir(), "best_agent") #Saves Actor and Critic
                                                          #Collect Stationary Policy afterword....

            obs, goal, extra = env.reset()
            
            done = False
            episode += 1
            if not ignore_first:
                stats = {'train_step' : step,
                    'train/episode'   : episode,
                    'train/success': done_bool,
                    'train/episode_reward' : episode_reward}
                logger.logging_tool.log(stats)
                #logger.dump_tabular(with_prefix=True, with_timestamp=False)

            else:
                ignore_first = False
            state = extra['state']
            episode_reward = 0
            episode_step = 0
            reward = 0
            logger.logging_tool.record()


        # sample action for data collection
        if step < details['initial_exploration_steps']:
            action = env.action_space.sample() #Random Sampling... Brownian Motion Might Be needed
            #TODO: Customizable init steps, obv not important for pixelsac but super important for goalbisim
        else:
            with eval_mode(agent): #Gather Data but detach
                action = agent.sample_action(obs, goal)

        # run training update
        if step >= details['initial_exploration_steps']:
            num_updates = details['initial_exploration_steps'] if step == details['initial_exploration_steps'] else 1 #updates every transition... maybe overkill
            for _ in range(num_updates):
                agent.update(replay_buffer, step)

        curr_reward = reward
        next_obs, reward, is_success, extra = env.step(action)

        if reward > 0:
            is_success = True

        next_state = extra['state']

        # allow infinit bootstrap

        if episode_step + 1 == env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
            done_bool = 1
            done_trajectory = 1
            done = True
        else: 
            done_bool = float(is_success)
            done_trajectory = float(is_success)
            if is_success:
                done = True

        episode_reward += reward * (discount ** episode_step)

        replay_buffer.add(obs, state, action, curr_reward, reward, next_obs, next_state, done_bool, done_trajectory, goal)

        obs = next_obs
        state = next_state
        episode_step += 1


