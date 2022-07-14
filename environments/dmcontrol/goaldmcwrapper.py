import torch
import gym
import numpy as np
import random
from collections import deque

class GoalDMCEnvWrapper(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)

        shp = env.observation_space.shape
        self.shp =  shp
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self.action_space = env.action_space
        self.goal_random_steps = 25
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs, extra = self.env.reset()
        for i in range(self.goal_random_steps):
            action = self.action_space.sample()
            obs, reward, done, info = self.env.step(action)

        self.env.mark_goal()

        goal = []
        for _ in range(self._k):
            goal.append(obs.copy())

        goal = np.concatenate(goal, axis=0)

        obs, extra = self.env.reset() #I think it shouldn't differ + ignore goal

        extra['seed'] = None

        for _ in range(self._k):
            self._frames.append(obs.copy())

        return self._get_obs(), goal, extra

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

    def get_goal_obs(self):
        return self._get_goal_obs()

    def _get_goal_obs(self):
        img = self.env.goal_image(self.shp[1], self.shp[2], 0).transpose(2, 0, 1).copy() #Might need to fix camera id?
        imgs = [img]
        for i in range(self._k - 1):
            imgs.append(img.copy())

        img = np.concatenate(imgs, axis = 0)
        return img