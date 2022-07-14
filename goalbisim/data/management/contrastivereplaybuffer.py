import numpy as np
import torch
from goalbisim.data.management.replaybuffer import ReplayBuffer
#from goalbisim.data.manipulation.transform import apply_transforms




class ContrastiveReplayBuffer(ReplayBuffer):

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device = None, transform=None):
        super().__init__(obs_shape, action_shape, capacity, batch_size, device, transform)

    
    def sample(self, batch_size = None):
        if batch_size:
            used_batch_size = batch_size
        else:
            used_batch_size = self.batch_size
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=used_batch_size
        )

      
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()
        if self.transform:
            outs = self.transform([obses, next_obses, pos], device = self.device)
            obses = outs[0]
            next_obses = outs[1]
            pos = outs[2]
        else:
            obses = torch.as_tensor(obses, device=self.device)
            next_obses = torch.as_tensor(next_obses, device=self.device)
            pos = torch.as_tensor(pos, device=self.device)
        
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        kwargs = {'anchor' : obses, 'positive' : pos, 'idxs' : idxs}

        return obses, actions, curr_rewards, rewards, next_obses, not_dones, kwargs

