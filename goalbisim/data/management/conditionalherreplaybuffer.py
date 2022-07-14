import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
from goalbisim.data.management.herreplaybuffer import HERReplayBuffer
from goalbisim.data.management.goalreplaybuffer import GoalReplayBuffer
import random
import pickle as pkl


class ConditionalHERReplayBuffer(HERReplayBuffer):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, state_shape, action_shape, discount = 0.99, capacity=100000, batch_size=256, reward_strategy=None, device=None, transform=None, num_goals=2):
        #import pdb; pdb.set_trace()
        super().__init__(obs_shape, state_shape, action_shape, discount = discount, capacity = capacity, batch_size = batch_size, device=device, transform=transform)
        
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.init_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

    def load(self, save_path, start = 0, end = None):
        filehandler = open(save_path, "rb")
        payload = pkl.load(filehandler)
        assert self.idx == 0, "Should be Empty Replay Buffer!"
        total = payload[0]
        if end is None:
            end = total

        self.init_obses[start:end] = payload[1][payload[11][start:end].squeeze()]

        self.obses[start:end] = payload[1][start:end]
        self.states[start:end] = payload[2][start:end]
        self.next_states[start:end] = payload[3][start:end]
        self.next_obses[start:end] = payload[4][start:end]
        self.actions[start:end] = payload[5][start:end]
        self.rewards[start:end] = payload[6][start:end]
        self.curr_rewards[start:end] = payload[7][start:end]
        self.trajectory_mark[start:end] = payload[8][start:end]
        self.goals[start:end] = payload[9][start:end]
        self.not_dones[start:end] = payload[10][start:end]
        self.trajectory_start_idx[start:end] = payload[11][start:end]
        self.trajectory_end_idx[start:end] = payload[12][start:end]
        self.reward_to_go[start:end] = payload[13][start:end]
        self.temporal_distance[start:end] = payload[14][start:end]

        self.sample_start = start

        self.idx = end

    def sample(self, batch_size = None, fetch_states = False):
        if batch_size:
            used_batch_size = batch_size
        else:
            used_batch_size = self.batch_size
        idxs = np.random.randint(
            self.sample_start, self.capacity if self.full else self.idx - self.current_trajectory_length, size=used_batch_size
        )
        
        obses = np.concatenate((self.obses[idxs], self.init_obses[idxs]), axis=1)
        next_obses = np.concatenate((self.next_obses[idxs], self.init_obses[idxs]), axis=1)
        goals = np.concatenate((self.goals[idxs], self.init_obses[idxs]), axis=1)
        pos = obses.copy()

        #TODO: Need to ask when to apply transformations and how to concat...
        #Also do you concatenate State/Goal to observation, such that the encoder 
        #import pdb; pdb.set_trace()

        if self.transform:
            try:
                outs = self.transform([obses, next_obses, goals, pos], device = self.device) #Should Normalize for you...
                obses = outs[0]
                next_obses = outs[1]
                goals = outs[2]
                pos = outs[3]
            except:
                import pdb; pdb.set_trace()
        else:
            obses = torch.as_tensor(obses, device=self.device).float().contiguous() / 255
            next_obses = torch.as_tensor(next_obses, device=self.device).float().contiguous() / 255
            goals = torch.as_tensor(goals, device=self.device).float().contiguous() / 255
        
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rtg_rewards = torch.as_tensor(self.reward_to_go[idxs], device = self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        tds = torch.as_tensor(self.temporal_distance[idxs], device = self.device)
        #cum_reward = torch.as_tensor(self.curr_rewards[idxs], device=self.device)


        if fetch_states:
            states = self.states[idxs]
            next_states = self.next_states[idxs]
            kwargs = {'states' : states, 'next_states' : next_states, 'rtg' : rtg_rewards, 'td' : tds, 'idxs' : idxs, 'pos' : pos}
        else:
            kwargs = {'rtg' : rtg_rewards, 'td' : tds, 'idxs' : idxs, 'pos' : pos}

        return obses, actions, rewards, next_obses, not_dones, goals, kwargs
