import torch
import numpy as np
import dmc2gym
import time
from goalbisim.utils.misc_utils import set_seed_everywhere, FrameStack, GoalFrameStack
from goalbisim.data.management.replaybuffer import ReplayBuffer
from goalbisim.agents.pixelsac import PixelSACAgent
from goalbisim.agents.goalpixelsac import GoalPixelSACAgent
from goalbisim.representation.base_representation import initialize_representation
from goalbisim.logging.logging import logger
from goalbisim.testing.eval_rl import evaluate_agent
from goalbisim.testing.eval_rl import evaluate_goal_agent
from goalbisim.utils.video import VideoRecorder
from goalbisim.utils.misc_utils import eval_mode
from goalbisim.data.management.init_replay import init_replay
from goalbisim.data.manipulation.transform import initialize_transform
from goalbisim.samplers.path_collector import collect_gcrl_trajectories
import wandb





def onpolicy_gcrl_sac(details):
    set_seed_everywhere(details['seed']) #good to use....

    env = dmc2gym.make(**details['env_kwargs'])
    env.seed(details['seed'])

    eval_env = dmc2gym.make(**details['env_kwargs'])

    # stack several consecutive frames together
    env = GoalFrameStack(env, k=details['frame_stack_count'])
    eval_env = GoalFrameStack(eval_env, k=details['frame_stack_count'])

    device = torch.device('cuda')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    #Use Generic Transition Replay Buffer, We Will need more complex replay buffer later
    replay_buffer = init_replay(env.observation_space.shape, env.action_space.shape, device, details, state_shape = env.state_space.shape)
    
    critic_representation = initialize_representation(env.observation_space.shape, details)
    actor_representation = initialize_representation(env.observation_space.shape, details)
    target_critic_representation = initialize_representation(env.observation_space.shape, details)

    eval_transforms = initialize_transform(details['eval_transforms'])

    agent = GoalPixelSACAgent(env.observation_space.shape, env.action_space.shape, device, eval_transforms, actor_representation, critic_representation, target_critic_representation,
        None, **details['sac_kwargs'])

    video = VideoRecorder(wandb.run.dir)

    episode, episode_reward, done = 0, 0, True
    ignore_first = True
    start_time = time.time()
    for cycle in range(details['num_cycles']):


        # evaluate agent periodically
        if cycle % details['eval_freq'] == 0:
            rewards, success, final_distances = evaluate_goal_agent(eval_env, agent, video, details['num_eval_episodes'], step)
            stats = {'train_cycle' : cycle,
            'eval/episode' : episode,
            'eval/avg_sucess': np.mean(success),
            'eval/std_sucess' : np.std(success),
            'eval/avg_episode_reward' : np.mean(rewards),
            'eval/std_episode_reward' : np.std(rewards),
            'eval/avg_final_distance' : np.mean(final_distances),
            'eval/std_final_distance' : np.std(final_distances)
            }

            logger.record_dict(stats)
            wandb.log(stats)
            
            if details['save_model']:
                agent.save(logger.get_snapshot_dir()) #Saves Actor and Critic

        if not ignore_first:
            stats = {'train_step' : step,
                'train/episode'   : episode,
                'train/final_distance': -reward,
                'train/success': done_bool,
                'train/episode_reward' : episode_reward}
            logger.record_dict(stats)
            wandb.log(stats)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)

        else:
            ignore_first = False

        episode = collect_gcrl_trajectories(env, agent, details['training_steps_per_cycle'], replay_buffer, False)

        for _ in range(details['updates_per_cycle']):
            agent.update(replay_buffer, cycle)

        replay_buffer.dump() #Agent updated, policy data now off-policy

