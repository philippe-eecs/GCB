import torch
import gym
import numpy as np
import random
from collections import deque

class Object(object):
    pass

class SawyerWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack = 1, action_repeat = 1):
        k = frame_stack
        self.distractor = None
        self.action_repeat = action_repeat #TODO
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        env.demo_reset()
        act_high = np.ones(env.action_space.shape[0])
        self.action_space = gym.spaces.Box(-act_high, act_high)
        self._max_episode_steps = env.max_episode_steps
        observation_dim = env.state_space.shape[0]
        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)
        self.state_space = state_space

        self.observation_space = Object()
        self.observation_space.shape = (3 * frame_stack, env.obs_img_dim, env.obs_img_dim)
        #self.current_goal = None
        
        self.env = env

    def set_distractor(self, distractor):
        self.distractor = distractor

    def reset(self, seed = None):
        if seed is None:
            seed = np.random.randint(9999999)
        self.env.demo_reset(seed = seed)
        old_obs = self.env.render_obs()
        goal = self._get_goal_obs()
        self.env.demo_reset(seed = seed)
        obs = self.env.render_obs()
        assert np.sum(np.abs(obs - old_obs)) == 0, "Problem With Seeded Reset"
        extra = {}
        extra['state'] = self.env.get_observation()['state_observation']
        extra['seed'] = seed
        for _ in range(self._k):
            self._frames.append(obs)



        obs = self._get_obs()

        if self.distractor is not None:
            self.distractor.step()
            obs = self.distractor.augment(obs)
            goal = self.distractor.goal_augment(goal)

        return obs, goal, extra

    def jitter_reset(self, seed = None):
        if seed is None:
            seed = np.random.randint(9999999)
        #self.env.force_state = True
        self.env.env.reset(seed = seed, force_state = True) #env.env is correct, weird wrapping issue
        obs = self.env.render_obs()
        #goal = self._get_goal_obs()
        #self.env.reset(seed = seed, force_state = True)
        #obs = self.env.render_obs()
        #assert np.sum(np.abs(obs - old_obs)) == 0, "Problem With Seeded Reset"
        extra = {}
        extra['state'] = self.env.get_observation()['state_observation']
        extra['seed'] = seed
        for _ in range(self._k):
            self._frames.append(obs)



        obs = self._get_obs()

        if self.distractor is not None:
            self.distractor.step()
            obs = self.distractor.augment(obs)
            #goal = self.distractor.goal_augment(goal)

        return obs, extra

    def demo_reset(self, seed = None):
        if seed is None:
            seed = np.random.randint(9999999)
        #print("collecting goal")
        self.env.demo_reset(seed = seed)
        old_obs = self.env.render_obs()
        goal = self._get_goal_obs()
        #print("goal collected")
        self.env.demo_reset(seed = seed)
        obs = self.env.render_obs()
        assert np.sum(np.abs(obs - old_obs)) == 0, "Problem With Seeded Reset"
        extra = {}
        extra['state'] = self.env.get_observation()['state_observation']
        for _ in range(self._k):
            self._frames.append(obs)

        obs = self._get_obs()

        if self.distractor is not None:
            self.distractor.step()
            obs = self.distractor.augment(obs)
            goal = self.distractor.goal_augment(goal)

        return obs, goal, extra

    def demo_jitter_reset(self, seed = None):    
        if seed is None:
            seed = np.random.randint(9999999)
        #print("collecting goal")
        self.env.demo_reset(seed = seed, force_state = True)
        obs = self.env.render_obs()
        goal = self._get_goal_obs()
        #print("goal collected")
        #self.env.demo_reset(seed = seed, force_state = True)
        #obs = self.env.render_obs()
        #assert np.sum(np.abs(obs - old_obs)) == 0, "Problem With Seeded Reset"
        extra = {}
        extra['state'] = self.env.get_observation()['state_observation']
        for _ in range(self._k):
            self._frames.append(obs)

        obs = self._get_obs()

        if self.distractor is not None:
            self.distractor.step()
            obs = self.distractor.augment(obs)
            goal = self.distractor.goal_augment(goal)

        return obs, goal, extra  

    def step(self, action):
        #assert self.current_goal is not None, "Make Sure you Reset!"
        for _ in range(self.action_repeat):
            obs, reward, done, info = self.env.step(action)
        info['state'] = self.env.get_observation()['state_observation']
        obs = self.env.render_obs()
        self._frames.append(obs)

        obs = self._get_obs()

        if self.distractor is not None:
            #self.distractor.step()
            obs = self.distractor.augment(obs)
            #goal = distractor.goal_augment(goal)

        #return obs, goal, extra

        return obs, reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=2).transpose((2, 0, 1))
        #return np.concatenate(list(self._frames), axis=0).transpose()

    #def get_goal_obs(self):
        #return self._get_goal_obs()

    def _get_goal_obs(self):
        done = False
        count = 0
        keep = self.env.demo_action_variance
        self.env.demo_action_variance = 0.05
        while not done and count < 200:
            action = self.env.get_demo_action()
            observation, reward, done, info = self.env.step(action)
            count += 1

        self.env.demo_action_variance = keep
        img = self.env.render_obs()
        
        imgs = [img]
        for i in range(self._k - 1):
            imgs.append(img.copy())
        img = np.concatenate(imgs, axis = 2).transpose((2, 0, 1))
        #img = np.concatenate(imgs, axis = 0).transpose()
        return img