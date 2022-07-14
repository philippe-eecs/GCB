import copy
import os
import pickle

import numpy as np
from absl import app, flags
from tqdm import tqdm

import matplotlib.pyplot as plt

from mujoco_offline_navigation.env_utils import make_dmc_env
from mujoco_offline_navigation.policy import get_policy

from dm_control.utils.transformations import euler_to_quat, quat_to_euler

from  test_moco import embedding

from dm_control import viewer
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from common import env_names

FLAGS = flags.FLAGS

flags.DEFINE_string('outprefix', 'mujoco', '')

save_folder = '/media/dcist-user/scratch/rl_traj_img64'

# objects = [np.array([-1.5, 1.5, 0]),
#             np.array([-1.5, 2.5, 0]),
#             np.array([-1.5, -1.5, 0]),
#             np.array([-1.5, -2.5, 0]),
#             np.array([4.5, 4.5, 0]),
#             np.array([4.5, 0.5, 0]),
#             np.array([4.5, -1.5, 0])]

# def calc_dist(objects, pos, orient):
#     res = []
#     pos = pos[:2]
#     for object in objects:
#         object = object[:2]
#         temp = ((object[1] - pos[1]) / (object[0] - pos[0]))
#         t1 = np.arctan(temp)
#         t2 = quat_to_euler(orient)[-1]

#         if(np.abs(t1 - t2) < np.deg2rad(21)):
#             res.append(np.linalg.norm(object - pos))
#         else:
#             res.append(np.Inf)
    
#     return min(res)

def main(_):
    # for env_name in env_names:
    spawn_x = [-5, -4]
    spawn_y = [-5, 3]
    goal_x = [1, 3]
    goal_y = [-5, 3]

    env = make_dmc_env("maze", spawn_x, spawn_y, goal_x, goal_y)
    policy, waypoints = get_policy(env)

    # viewer.launch(env, policy)

    timestep = None
    counter = 0
    total_count = 0

    for i in tqdm(range(80)):
        done = False
        timestep = env.reset()

        data = dict(states=[],
                actions=[],
                next_states=[],
                rewards=[],
                dones=[],
                images_embed=[],
                next_images_embed=[],
                spawngoal=[])

        for k in range(10):

            total_count += 1
            done = False
            for j in range(1000):
                if done:
                    counter += 1
                    break
                curr_state = timestep
                action = policy(curr_state)

                timestep  = env.step(action)
                done = timestep.last()

                # print("angle")
                # print(quat_to_euler(np.copy(curr_state.observation)[()]['walker/body_rotation'][0])[-1])
                # time.sleep(1000)

                data['states'].append(np.copy(curr_state.observation))
                data['actions'].append(np.copy(action))
                data['next_states'].append(np.copy(timestep.observation))
                data['rewards'].append(timestep.reward)
                data['dones'].append(timestep.last())
                data['images_embed'].append(embedding(np.copy(curr_state.observation)[()]['walker/realsense_camera'][0].T))
                data['next_images_embed'].append(embedding(np.copy(timestep.observation)[()]['walker/realsense_camera'][0].T))
                data['spawngoal'].append(env.task.get_spawn_goal())

                # print(data['next_images_embed'][-1])
            print(" ")
            print(f"Trajectory length: {j}")
            print(f"Total collection runs: {total_count}")
            print(f"Successfull collection runs: {counter}")

        os.makedirs(save_folder, exist_ok=True)

        save_file = os.path.join(save_folder, 'traj' + str(i) + '.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    app.run(main)