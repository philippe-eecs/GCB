import collections
from typing import Optional

# import d4rl
# import gym
import numpy as np
import sys
# np.set_printoptions(threshold=sys.maxsize)

from tqdm import tqdm
import math
import pickle
import time
from dm_control.utils.transformations import quat_to_euler


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

ImageBatch = collections.namedtuple(
    'ImageBatch',
    ['observations', 'image_observations', 'actions', 'rewards', 'masks', 'next_observations', 'next_image_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

class ImageDataset(Dataset):
    def __init__(self, dataset, image_observations: np.ndarray,
                    next_image_observations: np.ndarray,
                    dones_float: np.ndarray,):
        self.image_observations = image_observations
        self.next_image_observations = next_image_observations
        
        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


    def sample(self, batch_size: int) -> ImageBatch:
        indx = np.random.randint(self.size, size=batch_size)
        return ImageBatch(observations=self.observations[indx],
                     image_observations=self.image_observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     next_image_observations=self.next_image_observations[indx])

class MujDataset(Dataset):
    def __init__(self,
                 dir: str):

        print("Loading data")
        with open(dir, 'rb') as f:
            trajectories = pickle.load(f)

        print("loaded data")

        dataset = {}
        dataset['observations'] = []
        dataset['next_observations'] = []
        dataset['actions'] = []
        dataset['rewards'] = []
        dataset['terminals'] = []


        # print(trajectories['states'][0])
        # print(trajectories['states'][1])
        # print(np.array([quat_to_euler(trajectories['states'][1][()]['walker/body_rotation'][0])[-1]]))
        # time.sleep(100)
        counter = 0
        for t in range(1, 10):
            if trajectories['rewards'][t] != None:
            # print(len(t['actions']))
                # print(trajectories['actions'][t])
                # print(trajectories['actions'][t-1])
                # print(np.concatenate((trajectories['states'][t][()]['walker/body_position'][0],
                #                                                 np.array([quat_to_euler(trajectories['states'][t][()]['walker/body_rotation'][0])[-1]]), 
                #                                                 trajectories['actions'][t-1], 
                #                                                 trajectories['spawngoal'][t][1])))
                # time.sleep(100)
                dataset['observations'].append(np.concatenate((trajectories['states'][t][()]['walker/body_position'][0],
                                                                np.array([quat_to_euler(trajectories['states'][t][()]['walker/body_rotation'][0])[-1]]), 
                                                                trajectories['actions'][t-1], 
                                                                trajectories['spawngoal'][t][1])))
                dataset['next_observations'].append(np.concatenate((trajectories['next_states'][t][()]['walker/body_position'][0],
                                                                    np.array([quat_to_euler(trajectories['next_states'][t][()]['walker/body_rotation'][0])[-1]]), 
                                                                    trajectories['actions'][t],
                                                                    trajectories['spawngoal'][t][1])))
                dataset['actions'].append(trajectories['actions'][t])
                dataset['rewards'].append(trajectories['rewards'][t])
                # if trajectories['rewards'][t] == None or True:
                #     print(t)
                #     # print(counter)
                #     o = np.concatenate((trajectories['states'][t][()]['walker/body_position'][0], trajectories['spawngoal'][t][1]))
                #     counter += 1
                #     # print(-np.linalg.norm(o[:3] - o[3:]))
                #     print(trajectories['rewards'][t])
                #     # dataset['rewards'].append(-np.linalg.norm(o[:3] - o[3:]))
                #     print(np.concatenate((trajectories['states'][t][()]['walker/body_position'][0], trajectories['spawngoal'][t][1])))
                #     print(np.concatenate((trajectories['next_states'][t][()]['walker/body_position'][0], trajectories['spawngoal'][t][1])))
                #     #time.sleep(10)
                dataset['terminals'].append(int(trajectories['dones'][t]))
                # if trajectories['dones'][t]:
                #     time.sleep(1000)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
            print(np.shape(dataset[key]))

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if dataset['terminals'][i] == 1:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(np.float32),
                         size=len(dataset['observations']))

class PKLDataset(ImageDataset):
    def __init__(self,
                 dir: str):

        with open(dir, 'rb') as f:
            trajectories = pickle.load(f)

        dataset = {}
        dataset['observations'] = []
        dataset['next_observations'] = []
        dataset['image_observations'] = []
        dataset['next_image_observations'] = []
        dataset['actions'] = []
        dataset['rewards'] = []
        dataset['terminals'] = []

        for t in trajectories:
            # print(len(t['actions']))
            dataset['observations'].append(t['obervations'])
            dataset['next_observations'].append(t['next_observations'])
            dataset['image_observations'].append(t['image_observations'])
            dataset['next_image_observations'].append(t['next_image_observations'])
            dataset['actions'].append(t['actions'])
            dataset['rewards'].append(t['rewards'])
            dataset['terminals'].append(t['terminals'])

        for key in dataset.keys():
            dataset[key] = np.concatenate(dataset[key], axis=0)
            # print(key + " dimention is")
            # print(np.shape(dataset[key]))

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if dataset['terminals'][i] == 1:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset,
                         image_observations=dataset['image_observations'].astype(np.float32),
                         next_image_observations=dataset['next_image_observations'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32))

# class D4RLDataset(Dataset):
#     def __init__(self,
#                  env: gym.Env,
#                  clip_to_eps: bool = True,
#                  eps: float = 1e-5):
#         dataset = d4rl.qlearning_dataset(env)

#         if clip_to_eps:
#             lim = 1 - eps
#             dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

#         dones_float = np.zeros_like(dataset['rewards'])

#         for i in range(len(dones_float) - 1):
#             if np.linalg.norm(dataset['observations'][i + 1] -
#                               dataset['next_observations'][i]
#                               ) > 1e-6 or dataset['terminals'][i] == 1.0:
#                 dones_float[i] = 1
#             else:
#                 dones_float[i] = 0

#         dones_float[-1] = 1

#         print(np.shape(dataset['observations']))
#         super().__init__(dataset['observations'].astype(np.float32),
#                          actions=dataset['actions'].astype(np.float32),
#                          rewards=dataset['rewards'].astype(np.float32),
#                          masks=1.0 - dataset['terminals'].astype(np.float32),
#                          dones_float=dones_float.astype(np.float32),
#                          next_observations=dataset['next_observations'].astype(
#                              np.float32),
#                          size=len(dataset['observations']))


# class ReplayBuffer(Dataset):
#     def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
#                  capacity: int):

#         observations = np.empty((capacity, *observation_space.shape),
#                                 dtype=observation_space.dtype)
#         actions = np.empty((capacity, action_dim), dtype=np.float32)
#         rewards = np.empty((capacity, ), dtype=np.float32)
#         masks = np.empty((capacity, ), dtype=np.float32)
#         dones_float = np.empty((capacity, ), dtype=np.float32)
#         next_observations = np.empty((capacity, *observation_space.shape),
#                                      dtype=observation_space.dtype)
#         super().__init__(observations=observations,
#                          actions=actions,
#                          rewards=rewards,
#                          masks=masks,
#                          dones_float=dones_float,
#                          next_observations=next_observations,
#                          size=0)

#         self.size = 0

#         self.insert_index = 0
#         self.capacity = capacity

#     def initialize_with_dataset(self, dataset: Dataset,
#                                 num_samples: Optional[int]):
#         assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

#         dataset_size = len(dataset.observations)

#         if num_samples is None:
#             num_samples = dataset_size
#         else:
#             num_samples = min(dataset_size, num_samples)
#         assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

#         if num_samples < dataset_size:
#             perm = np.random.permutation(dataset_size)
#             indices = perm[:num_samples]
#         else:
#             indices = np.arange(num_samples)

#         self.observations[:num_samples] = dataset.observations[indices]
#         self.actions[:num_samples] = dataset.actions[indices]
#         self.rewards[:num_samples] = dataset.rewards[indices]
#         self.masks[:num_samples] = dataset.masks[indices]
#         self.dones_float[:num_samples] = dataset.dones_float[indices]
#         self.next_observations[:num_samples] = dataset.next_observations[
#             indices]

#         self.insert_index = num_samples
#         self.size = num_samples

#     def insert(self, observation: np.ndarray, action: np.ndarray,
#                reward: float, mask: float, done_float: float,
#                next_observation: np.ndarray):
#         self.observations[self.insert_index] = observation
#         self.actions[self.insert_index] = action
#         self.rewards[self.insert_index] = reward
#         self.masks[self.insert_index] = mask
#         self.dones_float[self.insert_index] = done_float
#         self.next_observations[self.insert_index] = next_observation

#         self.insert_index = (self.insert_index + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)
