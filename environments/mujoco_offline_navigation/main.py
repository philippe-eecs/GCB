import numpy as np
from dm_control import composer, viewer
from matplotlib import pyplot as plt
import warnings

from mujoco_offline_navigation.task import CarNavigate
from mujoco_offline_navigation.env_utils import make_dmc_env

from mujoco_offline_navigation.policy import get_policy

warnings.filterwarnings("ignore", category=DeprecationWarning)

spawn_x = [-5, -4]
spawn_y = [-5, 4]
goal_x = [1, 4]
goal_y = [-5, 4]

env = make_dmc_env("maze", spawn_x, spawn_y, goal_x, goal_y)

policy = get_policy(env)

o = env.reset()

viewer.launch(env, policy)
