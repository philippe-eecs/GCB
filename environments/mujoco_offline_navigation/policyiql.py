import os

import numpy as np
import jax.numpy as jnp
from dm_control.utils.transformations import quat_to_euler

from search import get_waypoints

import jax
from env_utils import make_dmc_env

from common import Model
import policyjax
from dataset_utils import MujDataset, PKLDataset


def get_policy(env):
    new_dir = "/media/dcist-user/scratch/rl_traj_noimg_mujoco/traj0.pkl"
    dataset1 = MujDataset(new_dir)
    samp_a = dataset1.sample(1)

    actor_def = policyjax.NormalTanhPolicy((256, 256),
                                            samp_a.actions[0][np.newaxis].shape[-1],
                                            log_std_scale=1e-1,
                                            log_std_min=-1.0,
                                            log_std_max=-1.0,
                                            dropout_rate=None,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

    # actor_def = policyjax.NormalTanhPolicy((256, 256), samp_a.actions[0][np.newaxis].shape[-1], tanh_squash_distribution=False)
    actor = Model.create(actor_def, inputs=[jax.random.PRNGKey(42), samp_a.observations[0][np.newaxis]])
    factor = actor.load("/home/dcist-user/iql_training/implicit_q_learning/model1")
    print("loaded")
    def policy(time_step):
        if time_step.first():
            prevaction = [0, 0]
            return [0, 0]

        body_position = time_step.observation['walker/body_position'][0]
        body_rotation = np.array([quat_to_euler(time_step.observation['walker/body_rotation'][0])[-1]])
        prevaction = np.array(env.task.get_prev_action())
        boday_goal = np.array(env.task.get_spawn_goal()[1])
        print(prevaction)
        o = np.concatenate((body_position, body_rotation, prevaction, boday_goal))
        
        # action = factor(o).sample(seed=jax.random.PRNGKey(42))
        action = policyjax.sample_actions(jax.random.PRNGKey(42), factor.apply_fn,
                                             factor.params, o,
                                             0.0)
        # print(action[1])
        return action[1]

    return policy
