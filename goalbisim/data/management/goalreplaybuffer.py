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



class GoalReplayBuffer(ReplayBuffer):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, state_shape, action_shape, discount = 0.99, capacity=100000, batch_size=256, device=None, transform=None, force_obs_shape=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.discount = discount
        #self.num_goals = num_goals

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        if force_obs_shape is not None:
            obs_shape = force_obs_shape

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32) #To mark if a goal was met in this step
        self.trajectory_mark = np.empty((capacity, 1), dtype=np.float32) #To mark if a trejectory finished in this step regardless if goal was met
        self.trajectory_start_idx = np.empty((capacity, 1), dtype=np.int64) #Mark idx of beginning of trajectory
        self.trajectory_end_idx = np.empty((capacity, 1), dtype=np.int64) #Mark idx of end of trajectory
        self.goals = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.reward_to_go = np.empty((capacity, 1), dtype=np.float32)
        self.temporal_distance = np.empty((capacity, 1), dtype=np.float32)

        if transform:
            self.transform = transform
        else:
            self.transform = False

        # #Idxs of the start of all trajectories stored in replay

        self.idx = 0 #Current index of the replay buffer
        self.trajectory_idx = 0 #Index of first step of current trajectory in replay buffer, because of relabeling, not every 
        self.current_trajectory_length = 0
        self.full = False

        self.obs_shape = obs_shape

        self.sample_start = 0

    def overlay(self, distractor):
        for i in range(self.sample_start, self.idx):
            if not self.not_dones[i]:
                distractor.step()

            self.obses[i] = distractor.augment(self.obses[i])
            self.next_obses[i] = distractor.next_augment(self.next_obses[i])
            self.goals[i] = distractor.goal_augment(self.goals[i])

    def video_populate(self, frames, sequence_length, frame_stack = 1): #TODO, Frame Stacking?

        assert self.idx == 0

        if isinstance(frames, list):

            for video in frames:
                step_size = video.shape[0] // sequence_length
                if step_size == 0:
                    continue #video frames less than sequence
                trajectory_start_idx = self.idx
                trajectory_end_idx = self.idx + sequence_length
                goal = video[sequence_length - 1]
                from skimage.transform import resize
                count = 0
                for frame in video:
                    #assert Fals
                    self.obses[self.idx] = frame.copy().transpose((2, 0, 1))
                    self.trajectory_start_idx[self.idx] = trajectory_start_idx
                    self.trajectory_end_idx[self.idx] = trajectory_end_idx

                    self.goals[self.idx] = goal.copy().transpose((2, 0, 1))

                    self.idx += 1

                    count += 1

                    if count >= sequence_length:
                        break
        else:

            for i in range(frames.shape[0]):
                trajectory_start_idx = i - (i % sequence_length)
                trajectory_end_idx = i - (-i % sequence_length)
                #if i % sequence_length == 0:
                    #self.trajectory_start_idx

                self.obses[self.idx] = frames[i].copy()
                self.trajectory_start_idx[self.idx] = trajectory_start_idx
                self.trajectory_end_idx[self.idx] = trajectory_end_idx
                self.goals[self.idx] = frames[trajectory_end_idx].copy()

                self.idx += 1




        
    '''
    Use add trajectory if you want to use HER
    '''
    def add(self, obs, state, action, curr_reward, reward, next_obs, next_state, done, trajectory_done, goal):
        self.current_trajectory_length += 1
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.trajectory_start_idx[self.idx], self.trajectory_idx)
        np.copyto(self.trajectory_mark[self.idx], trajectory_done)
        np.copyto(self.goals[self.idx], goal)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

        if trajectory_done:
            count = 0
            discounts = self.discount ** (np.arange(self.current_trajectory_length))
            traj_idxs = np.arange(self.trajectory_idx, self.trajectory_idx + self.current_trajectory_length) % self.capacity
            traj_rewards = self.rewards[traj_idxs]

            for proper_idx in traj_idxs:

                if count != 0:
                    RTG = np.sum(traj_rewards[count:].squeeze() * discounts[:-count])
                else:
                    RTG = np.sum(traj_rewards.squeeze() * discounts)

                np.copyto(self.trajectory_end_idx[proper_idx], (self.trajectory_idx + self.current_trajectory_length) % self.capacity)
                np.copyto(self.temporal_distance[proper_idx], self.trajectory_end_idx[proper_idx] - proper_idx)

                np.copyto(self.reward_to_go[proper_idx], RTG)

                count += 1

            self.trajectory_idx = (self.idx) % self.capacity
            self.current_trajectory_length = 0

        

    def sample(self, batch_size = None, fetch_states = False, contrastive_fetch = False):
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


        if fetch_states:
            states = None
            next_states = None
            kwargs = {'states' : states, 'next_states' : next_states, 'rtg' : rtg_rewards, 'td' : tds, 'idxs' : idxs, 'pos' : pos}
        else:
            kwargs = {'rtg' : rtg_rewards, 'td' : tds, 'idxs' : idxs, 'pos' : pos}

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
            outs = self.transform([traj_obses, traj_next_obses, traj_goals], device = self.device) #Should Normalize for you...
            traj_obses = outs[0]
            traj_next_obses = outs[1]
            traj_goals = outs[2]
        else:
            traj_obses = torch.as_tensor(traj_obses, device=self.device).float() / 255
            traj_next_obses = torch.as_tensor(traj_next_obses, device=self.device).float() / 255
            traj_goals = torch.as_tensor(traj_goals, device=self.device).float() / 255

        kwargs['rtg'] = traj_rtg
        kwargs['td'] = traj_tds

        return traj_obses, traj_actions, traj_curr_reward, traj_rewards, traj_next_obses, traj_not_dones, traj_goals, kwargs






    def dump(self):

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32) #To mark if a goal was met in this step
        self.trajectory_mark = np.empty((capacity, 1), dtype=np.float32) #To mark if a trejectory finished in this step regardless if goal was met
        self.goals = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        
        self.idx = 0
        self.trajectory_idx = 0
        self.last_save = 0
        self.full = False       

    def save(self, save_dir, name):
        if self.full:
            total = self.capacity
        else:
            total = self.idx
        path = os.path.join(save_dir, name + '.pt')
        payload = [
            total,
            self.obses[:total],
            self.states[:total],
            self.next_states[:total],
            self.next_obses[:total],
            self.actions[:total],
            self.rewards[:total],
            self.curr_rewards[:total],
            self.trajectory_mark[:total],
            self.goals[:total],
            self.not_dones[:total],
            self.trajectory_start_idx[:total], 
            self.trajectory_end_idx[:total],
            self.reward_to_go[:total],
            self.temporal_distance[:total]
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

        self.obses[start:end] = payload[1][start:end]
        #self.states[start:end] = payload[2][start:end]
        #self.next_states[start:end] = payload[3][start:end]
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
        self.sample_end = end

        self.idx = end
