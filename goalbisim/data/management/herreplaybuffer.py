import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
from goalbisim.data.management.goalreplaybuffer import GoalReplayBuffer
import random



class HERReplayBuffer(GoalReplayBuffer):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, state_shape, action_shape, discount = 0.99, capacity=100000, batch_size=256, reward_strategy=None, device=None, transform=None, num_goals=1):
        #import pdb; pdb.set_trace()
        super().__init__(obs_shape, state_shape, action_shape, discount = discount, capacity = capacity, batch_size = batch_size, device=device, transform=transform)

        self.num_goals = num_goals
        self.reward_strategy = reward_strategy

    def relabel(self):
        assert not self.full
        end = self.idx
        for idx in range(0, end):
            max_idx = int(self.trajectory_end_idx[idx])
            #for i in range(int(self.trajectory_start_idx[idx]), max_idx):

            for k in range(self.num_goals):
                #import pdb; pdb.set_trace()
                coin = np.random.uniform()
                if coin > 0.25:
                    coin = np.random.uniform()
                    if coin > 0.2:
                        j = np.random.randint(idx, int(self.trajectory_end_idx[idx])) #Future
                    else:
                        j = idx + 1
                else:
                    continue
                
                proper_i = idx % self.capacity
                proper_j = j % self.capacity

                np.copyto(self.obses[self.idx], self.obses[proper_i].copy())
                np.copyto(self.actions[self.idx], self.actions[proper_i].copy())
                np.copyto(self.curr_rewards[self.idx], self.curr_rewards[proper_i])
                np.copyto(self.states[self.idx], self.states[proper_i].copy())
                #Custom reward for pointmass....
                #TODO: Integrate Seperate Reward function for relabeling
                if self.reward_strategy:
                    new_reward, done_flag = self.reward_strategy(self.obses[proper_i], self.actions[proper_i],self.next_obses[proper_i], \
                    self.states[proper_i], self.next_states[proper_i], self.obses[proper_j],self.states[proper_j], proper_i, proper_j)
                else:
                    new_reward = int(proper_i + 1 == proper_j)
                    done_flag = (proper_i + 1 == proper_j) #0 or 1

                np.copyto(self.rewards[self.idx], new_reward)
                np.copyto(self.next_obses[self.idx], self.next_obses[proper_i].copy())
                np.copyto(self.next_states[self.idx], self.next_states[proper_i].copy())
                np.copyto(self.not_dones[self.idx], not done_flag)
                np.copyto(self.trajectory_start_idx[self.idx], self.trajectory_idx) #Place Trajectory it came from
                np.copyto(self.trajectory_end_idx[self.idx], (self.trajectory_idx + self.current_trajectory_length) % self.capacity) #Really Just Filler...
                np.copyto(self.trajectory_mark[self.idx], True) #True to avoid bugs, since no trajectory occured here anyway, should never really be touched...
                np.copyto(self.goals[self.idx], self.obses[proper_j].copy())
                np.copyto(self.reward_to_go[self.idx], self.reward_to_go[proper_j].copy())
                np.copyto(self.temporal_distance[self.idx], abs(proper_j - proper_i)) #Not Actually True...
                self.idx = (self.idx + 1) % self.capacity
                self.full = self.full or self.idx == 0

            self.trajectory_idx = (self.idx) % self.capacity
            self.current_trajectory_length = 0
        
        
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
                np.copyto(self.temporal_distance[proper_idx], self.trajectory_end_idx[proper_idx] - proper_idx) #Add Cost, but not sure tbh

                np.copyto(self.reward_to_go[proper_idx], RTG)

                count += 1

            max_idx = self.trajectory_idx + self.current_trajectory_length

            for i in range(self.trajectory_idx, max_idx):

                for k in range(self.num_goals):

                    coin = np.random.uniform()
                    if coin > 0.6:
                        coin = np.random.uniform()
                        if coin > 0.1:
                            try:
                                j = np.random.randint(i, int(self.trajectory_end_idx[i])) #Future
                            except:
                                j = i + 1
                        else:
                            j = i + 1
                    else:
                        continue

                    proper_i = i % self.capacity
                    proper_j = j % self.capacity

                    np.copyto(self.obses[self.idx], self.obses[proper_i].copy())
                    np.copyto(self.actions[self.idx], self.actions[proper_i].copy())
                    np.copyto(self.curr_rewards[self.idx], self.curr_rewards[proper_i])
                    np.copyto(self.states[self.idx], self.states[proper_i].copy())
                    #Custom reward for pointmass....
                    #TODO: Integrate Seperate Reward function for relabeling
                    if self.reward_strategy:
                        new_reward, done_flag = self.reward_strategy(self.obses[proper_i], self.actions[proper_i],self.next_obses[proper_i], \
                        self.states[proper_i], self.next_states[proper_i], self.obses[proper_j],self.states[proper_j], proper_i, proper_j)
                    else:
                        new_reward = int(proper_i + 1 == proper_j)
                        done_flag = (proper_i + 1 == proper_j) #0 or 1

                    np.copyto(self.rewards[self.idx], new_reward)
                    np.copyto(self.next_obses[self.idx], self.next_obses[proper_i].copy())
                    np.copyto(self.next_states[self.idx], self.next_states[proper_i].copy())
                    np.copyto(self.not_dones[self.idx], not done_flag)
                    np.copyto(self.trajectory_start_idx[self.idx], self.trajectory_idx) #Place Trajectory it came from
                    np.copyto(self.trajectory_end_idx[self.idx], (self.trajectory_idx + self.current_trajectory_length) % self.capacity) #Really Just Filler...
                    np.copyto(self.trajectory_mark[self.idx], True) #True to avoid bugs, since no trajectory occured here anyway, should never really be touched...
                    np.copyto(self.goals[self.idx], self.obses[proper_j].copy())
                    np.copyto(self.reward_to_go[self.idx], self.reward_to_go[proper_j].copy())
                    np.copyto(self.temporal_distance[self.idx], abs(proper_j - proper_i)) #Not Actually True...
                    self.idx = (self.idx + 1) % self.capacity
                    self.full = self.full or self.idx == 0




            self.trajectory_idx = (self.idx) % self.capacity
            self.current_trajectory_length = 0



