import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
from goalbisim.data.management.replaybuffer import ReplayBuffer
#from goalbisim.data.manipulation.transform import apply_transforms
import random
import pickle as pkl



class GoalAnalogyReplayBuffer(ReplayBuffer):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, state_shape, action_shape, discount = 0.99, capacity=100000, batch_size=256, device=None, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.discount = discount
        #self.num_goals = num_goals

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.analogy_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.analogy_next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.analogy_actions = np.empty((capacity, *action_shape), dtype=np.float32)

        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.analogy_rewards = np.empty((capacity, 1), dtype=np.float32)

        self.not_dones = np.empty((capacity, 1), dtype=np.float32) #To mark if a goal was met in this step
        self.analogy_not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.trajectory_mark = np.empty((capacity, 1), dtype=np.float32) #To mark if a trejectory finished in this step regardless if goal was met
        self.trajectory_start_idx = np.empty((capacity, 1), dtype=np.int64) #Mark idx of beginning of trajectory
        self.trajectory_end_idx = np.empty((capacity, 1), dtype=np.int64) #Mark idx of end of trajectory

        self.match_trajectory_start = np.empty((capacity, 1), dtype=np.int64)
        #self.match_trajectory_start_inv = np.empty((capacity, 1), dtype=np.int64)

        self.analogy_trajectory_mark = np.empty((capacity, 1), dtype=np.float32) #To mark if a trejectory finished in this step regardless if goal was met
        self.analogy_trajectory_start_idx = np.empty((capacity, 1), dtype=np.int64) #Mark idx of beginning of trajectory
        self.analogy_trajectory_end_idx = np.empty((capacity, 1), dtype=np.int64) #Mark idx of end of trajectory
        #self.analogy_state = np.empty((capacity, *obs_shape), dtype = obs_dtype)
        #self.analogy_goal = np.empty((capacity, *obs_shape), dtype = obs_dtype)
        self.goals = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.analogy_goals = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        #self.reward_to_go = np.empty((capacity, 1), dtype=np.float32)
        #self.temporal_distance = np.empty((capacity, 1), dtype=np.float32)

        self.init_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        if transform:
            self.transform = transform
        else:
            self.transform = False

        # #Idxs of the start of all trajectories stored in replay

        self.idx = 0 #Current index of the replay buffer
        self.trajectory_idx = 0 #Index of first step of current trajectory in replay buffer, because of relabeling, not every 
        self.current_trajectory_length = 0


        self.analogy_idx = 0 #Current index of the replay buffer
        self.analogy_trajectory_idx = 0 #Index of first step of current trajectory in replay buffer, because of relabeling, not every 
        self.analogy_current_trajectory_length = 0


        self.full = False
        self.analogy_full = False
        #self.analogy_trajectory = False

        self.obs_shape = obs_shape

        self.sample_start = 0

    def overlay(self, distractor):
        
        for i in range(self.sample_start, self.idx):
            if not self.not_dones[i]:
                distractor.step()

            self.obses[i] = distractor.augment(self.obses[i])
            self.next_obses[i] = distractor.next_augment(self.next_obses[i])
            self.goals[i] = distractor.goal_augment(self.goals[i])

        distractor.step()

        for i in range(self.analogy_sample_start, self.analogy_idx):
            if not self.analogy_not_dones[i]:
                distractor.step()

            self.analogy_obses[i] = distractor.augment(self.analogy_obses[i])
            self.analogy_next_obses[i] = distractor.next_augment(self.analogy_next_obses[i])
            self.analogy_goals[i] = distractor.goal_augment(self.analogy_goals[i])

    '''
    Use add trajectory if you want to use HER
    '''
    def add(self, obs, action, reward, next_obs, done, trajectory_done, goal):

        self.current_trajectory_length += 1
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.trajectory_start_idx[self.idx], self.trajectory_idx)
        np.copyto(self.trajectory_mark[self.idx], trajectory_done)
        np.copyto(self.goals[self.idx], goal)
        np.copyto(self.match_trajectory_start[self.idx], self.analogy_trajectory_idx)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

        if trajectory_done:
            count = 0
            traj_idxs = np.arange(self.trajectory_idx, self.trajectory_idx + self.current_trajectory_length) % self.capacity
            for proper_idx in traj_idxs:

                np.copyto(self.trajectory_end_idx[proper_idx], (self.trajectory_idx + self.current_trajectory_length) % self.capacity)

                count += 1

            self.trajectory_idx = (self.idx) % self.capacity
            self.current_trajectory_length = 0

    def add_analogy(self, obs, action, reward, next_obs, done, trajectory_done, goal):
        self.analogy_current_trajectory_length += 1
        np.copyto(self.analogy_obses[self.analogy_idx], obs)
        np.copyto(self.analogy_actions[self.analogy_idx], action)
        np.copyto(self.analogy_rewards[self.analogy_idx], reward)
        np.copyto(self.analogy_next_obses[self.analogy_idx], next_obs)
        np.copyto(self.analogy_not_dones[self.analogy_idx], not done)
        np.copyto(self.analogy_trajectory_start_idx[self.analogy_idx], self.analogy_trajectory_idx)
        np.copyto(self.analogy_trajectory_mark[self.analogy_idx], trajectory_done)
        np.copyto(self.analogy_goals[self.analogy_idx], goal)

        self.analogy_idx = (self.analogy_idx + 1) % self.capacity
        self.analogy_full = self.analogy_full or self.analogy_idx == 0

        if trajectory_done:
            count = 0
            traj_idxs = np.arange(self.analogy_trajectory_idx, self.analogy_trajectory_idx + self.analogy_current_trajectory_length) % self.capacity
            for proper_idx in traj_idxs:

                np.copyto(self.analogy_trajectory_end_idx[proper_idx], (self.analogy_trajectory_idx + self.analogy_current_trajectory_length) % self.capacity)

                count += 1

            self.analogy_trajectory_idx = (self.analogy_idx) % self.capacity
            self.analogy_current_trajectory_length = 0

        

    def sample(self, batch_size = None, fetch_states = True):
        if batch_size:
            used_batch_size = batch_size
        else:
            used_batch_size = self.batch_size
        idxs = np.random.randint(
            self.sample_start, self.capacity if self.full else self.idx - self.current_trajectory_length, size=used_batch_size
        )
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        goals = self.goals[idxs]
        analogy_idxs = self.match_trajectory_start[idxs]
        analogy_obses = self.analogy_obses[analogy_idxs].squeeze()
        analogy_goals = self.analogy_goals[analogy_idxs].squeeze()
        init_obses = self.init_obses[idxs]
        pos = obses.copy()

        #TODO: Need to ask when to apply transformations and how to concat...
        #Also do you concatenate State/Goal to observation, such that the encoder 
        #import pdb; pdb.set_trace()

        if self.transform:
            try:
                outs = self.transform([obses, next_obses, goals, pos, init_obses, analogy_obses, analogy_goals], device = self.device) #Should Normalize for you...
                obses = outs[0]
                next_obses = outs[1]
                goals = outs[2]
                pos = outs[3]
                init_obses = outs[4]
                analogy_obses = outs[5]
                analogy_goals = outs[6]
            except:
                import pdb; pdb.set_trace()
        else:
            obses = torch.as_tensor(obses, device=self.device).float().contiguous() / 255
            next_obses = torch.as_tensor(next_obses, device=self.device).float().contiguous() / 255
            goals = torch.as_tensor(goals, device=self.device).float().contiguous() / 255
            init_obses = torch.as_tensor(init_obses, device=self.device).float().contiguous() / 255
        
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        #curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        #rtg_rewards = torch.as_tensor(self.reward_to_go[idxs], device = self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        #tds = torch.as_tensor(self.temporal_distance[idxs], device = self.device)


        kwargs = {'idxs' : idxs, 'pos' : pos, 'analogy_obses' : analogy_obses, 'analogy_goals' : analogy_goals, 'init_obs': init_obses}

        return obses, actions, rewards, next_obses, not_dones, goals, kwargs

    def sample_trajectory(self, fetch_states = False):
        #Try to sample one at a time as trajectories will have different lengths...

        idxs = np.random.randint(
            self.sample_start, self.sample_end, size=1
        )

        start_idx, end_idx = self.trajectory_start_idx[idxs], self.trajectory_end_idx[idxs]

        trajectory_idxs = np.arange(start_idx, end_idx) % self.capacity #Pad to make sure of no mistakes...

        traj_obses = self.obses[trajectory_idxs]
        traj_actions = self.actions[trajectory_idxs]
        traj_rewards = self.rewards[trajectory_idxs]
        traj_next_obses = self.next_obses[trajectory_idxs]
        traj_curr_reward = self.curr_rewards[trajectory_idxs]
        traj_not_dones = self.not_dones[trajectory_idxs]
        traj_goals = self.goals[trajectory_idxs]
        traj_tds = self.temporal_distance[trajectory_idxs]
        
        discounts = self.discount ** (np.arange(traj_rewards.shape[0]))
        traj_rtg = []
        for i in range(traj_rewards.shape[0]):
            if i != 0:
                traj_rtg.append(np.sum(traj_rewards[i:].squeeze() * discounts[:-i]))
            else:
                traj_rtg.append(np.sum(traj_rewards.squeeze() * discounts))

        traj_rtg = np.array(traj_rtg)

        if fetch_states:
            traj_states = self.states[trajectory_idxs]
            traj_next_states = self.next_states[trajectory_idxs]
            kwargs = {'states' : traj_states, 'next_states' : traj_next_states}
        else:
            kwargs = {}

        if self.transform:
            try:
                outs = self.transform([obses, next_obses, goals, pos, init_obses], device = self.device) #Should Normalize for you...
                obses = outs[0]
                next_obses = outs[1]
                goals = outs[2]
                pos = outs[3]
                init_obses = outs[4]
            except:
                import pdb; pdb.set_trace()
        else:
            obses = torch.as_tensor(obses, device=self.device).float().contiguous() / 255
            next_obses = torch.as_tensor(next_obses, device=self.device).float().contiguous() / 255
            goals = torch.as_tensor(goals, device=self.device).float().contiguous() / 255
            init_obses = torch.as_tensor(init_obses, device=self.device).float().contiguous() / 255

        kwargs['rtg'] = traj_rtg
        kwargs['td'] = traj_tds

        return traj_obses, traj_actions, traj_curr_reward, traj_rewards, traj_next_obses, traj_not_dones, traj_goals, kwargs    

    def save(self, save_dir, name):
        if self.full:
            total = self.capacity
        else:
            total = self.idx
        path = os.path.join(save_dir, name + '.pt')
        payload = [
            total,
            self.obses[:total],
            self.analogy_obses[:total],
            self.next_obses[:total],
            self.analogy_next_obses[:total],
            self.actions[:total],
            self.analogy_actions[:total],
            self.rewards[:total],
            self.analogy_rewards[:total],
            self.trajectory_mark[:total],
            self.analogy_trajectory_mark[:total],
            self.goals[:total],
            self.analogy_goals[:total],
            self.not_dones[:total],
            self.analogy_not_dones[:total],
            self.trajectory_start_idx[:total], 
            self.trajectory_end_idx[:total],
            self.analogy_trajectory_start_idx[:total], 
            self.analogy_trajectory_end_idx[:total],
            self.match_trajectory_start[:total]
        ]

        #self.last_save = self.idx
        filehandler = open(path, "wb")
        pkl.dump(payload, filehandler)

    def load(self, save_path, start = 0, end = None):
        filehandler = open(save_path, "rb")
        payload = pkl.load(filehandler)
        assert self.idx == 0, "Should be Empty Replay Buffer!"
        total = payload[0]
        if end is None:
            end = total

        self.init_obses[start:end] = payload[1][payload[15][start:end].squeeze()]

        self.match_trajectory_start[start:end] = payload[19][start:end]
        analogy_start = self.match_trajectory_start[start][0]
        analogy_end = self.match_trajectory_start[end - 1][0]

        self.obses[start:end] = payload[1][start:end]
        self.analogy_obses[analogy_start:analogy_end] = payload[2][analogy_start:analogy_end]
        #self.states[start:end] = payload[2][start:end]
        #self.next_states[start:end] = payload[3][start:end]
        self.next_obses[start:end] = payload[3][start:end]
        self.analogy_next_obses[analogy_start:analogy_end] = payload[4][analogy_start:analogy_end]

        self.actions[start:end] = payload[5][start:end]
        self.analogy_actions[analogy_start:analogy_end] = payload[6][analogy_start:analogy_end]

        self.rewards[start:end] = payload[7][start:end]
        self.analogy_rewards[analogy_start:analogy_end] = payload[8][analogy_start:analogy_end]

        #self.curr_rewards[start:end] = payload[7][start:end]
        self.trajectory_mark[start:end] = payload[9][start:end]
        self.analogy_trajectory_mark[analogy_start:analogy_end] = payload[10][analogy_start:analogy_end]

        self.goals[start:end] = payload[11][start:end]
        self.analogy_goals[analogy_start:analogy_end] = payload[12][analogy_start:analogy_end]

        self.not_dones[start:end] = payload[13][start:end]
        self.analogy_not_dones[analogy_start:analogy_end] = payload[14][analogy_start:analogy_end]

        self.trajectory_start_idx[start:end] = payload[15][start:end]
        self.trajectory_end_idx[start:end] = payload[16][start:end]

        self.analogy_trajectory_start_idx[analogy_start:analogy_end] = payload[17][analogy_start:analogy_end]
        self.analogy_trajectory_end_idx[analogy_start:analogy_end] = payload[18][analogy_start:analogy_end]
        #self.reward_to_go[start:end] = payload[13][start:end]
        #self.temporal_distance[start:end] = payload[14][start:end]

        self.sample_start = start
        self.analogy_sample_start = analogy_start
        self.analogy_sample_end = analogy_end
        self.sample_end = end

        self.idx = end
        self.analogy_idx = analogy_end
