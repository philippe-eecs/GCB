import roboverse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg
import os
from PIL import Image
import math
import argparse
from goalbisim.data.management.herreplaybuffer import HERReplayBuffer
from goalbisim.trainers.offline_trainers import collect_offline_replay
from environments.environment_init import load_env
from goalbisim.data.manipulation.transform import initialize_transform
import roboverse

replay_buffer_kwargs = dict(
    capacity = 200000,
    batch_size = 256,
    num_goals=0
    )

save_loc = '/2tb/home/patrickhaoy/data_goalbisim/data_g/'

details = dict(
        discount=0.99,
        env_kwargs = dict(
            package = 'roboverse',
            domain_name = 'sawyer_rig_v4',
            domain_kwargs = dict(
                expl = False,
                random_color_p=1,
                max_episode_steps = 75,
                obs_img_dim = 64,
                claw_spawn_mode='fixed', #Maybe Uniform Instead, so distractor is more distracting....?
                drawer_yaw_setting = (0, 360),
                color_range = (0, 255),
                drawer_bounding_x = [.46, .84],
                drawer_bounding_y = [-.19, .19],
                demo_action_variance = 0.3,
                view_distance = 0.55,
                max_distractors = 4
                ),
            frame_stack_count = 1,
            action_repeat = 1,
            ),)
env = load_env(details)
transform = initialize_transform([])

replay_buffer = HERReplayBuffer((3, 64, 64), env.state_space.shape, env.action_space.shape, reward_strategy = None, device = 'cuda', transform = transform, **replay_buffer_kwargs)
replay_buffer = collect_offline_replay('demo', env, 0.99, replay_buffer, 120000)
replay_buffer.save(save_loc, "replay_sawyer_env4_v1_easy")