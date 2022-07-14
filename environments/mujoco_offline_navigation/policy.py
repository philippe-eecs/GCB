import os

import numpy as np
from dm_control.utils.transformations import quat_to_euler

from mujoco_offline_navigation.search import get_waypoints
from mujoco_offline_navigation.env_utils import make_dmc_env

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'models')

PROB = 0.0

def get_policy(env):
    waypoints = []

    # stopped_step = 0

    def policy(time_step):
        global waypoints
        # global stopped_step

        if time_step.first():
            waypoints = get_waypoints(env.task._maze_arena, env.task.get_spawn_goal()[0], env.task.get_spawn_goal()[1])
            # print("wps: ", waypoints)

        body_position = time_step.observation['walker/body_position']

        # target_position = env.task._maze_arena.target_positions[0][:2]
        walker_position = body_position[0, :2]

        current_waypoint = len(waypoints) - 1

        target_position = None

        while True:
            target_position = waypoints[current_waypoint]
            direction = target_position[:2] - walker_position[:2]
            if np.linalg.norm(direction) < 1.3 or current_waypoint == 0:
                break
            current_waypoint -= 1

        body_quat = time_step.observation['walker/body_rotation'][0]
        body_yaw = quat_to_euler(body_quat)[-1]
        goal_yaw = np.arctan2(direction[1], direction[0])

        diff = (goal_yaw - body_yaw)

        steer = 0.6 * diff + np.random.normal(0, 0.07)
        throttle = min(1*np.linalg.norm(direction), 1) + np.random.normal(0, 0.07)

        action = [steer, throttle]

        if env.task.get_stop_timestep() > 0:
            action = [0, 0]
            # print("decrememnt step", env.task.get_stop_timestep())
            env.task.set_stop_timestep(env.task.get_stop_timestep() - 1)

        else:
            p = np.random.rand()
            # print(p)
            if p < PROB:
                print("Pause in effect")
                env.task.set_stop_timestep(30)


        # print("curr_pos ", walker_position)
        # print("target pt ", target_position)
        # print("th = ", np.linalg.norm(direction))
        # print(action)
        return action

    return policy, waypoints
