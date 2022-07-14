import torch
import gym
import numpy as np
import random

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs
from collections import deque

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class GoalFrameStack(gym.Wrapper):
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
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs, extra = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), self._get_goal_obs(), extra

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

class ImageSource(object):
    """
    Source of natural images to be added to a simulated environment.
    """
    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """ Called when an episode ends. """
        pass

class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        self.total_frames = self.total_frames if self.total_frames else len(self.filelist)
        self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
        for i in range(self.total_frames):
            # if i % len(self.filelist) == 0: random.shuffle(self.filelist)
            fname = self.filelist[i % len(self.filelist)]
            if self.grayscale: im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)[..., None]
            else:              im = cv2.imread(fname, cv2.IMREAD_COLOR)
            self.arr[i] = cv2.resize(im, (self.shape[1], self.shape[0])) ## THIS IS NOT A BUG! cv2 uses (width, height)

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        return self.arr[self._loc]


class RandomVideoSource(ImageSource):
    def __init__(self, shape, dataset_loc, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        filelist = [join(dataset_loc, f) for f in listdir(dataset_loc) if isfile(join(dataset_loc, f))]
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        if not self.total_frames:
            self.total_frames = 0
            self.arr = None
            random.shuffle(self.filelist)
            for fname in tqdm.tqdm(self.filelist, desc="Loading videos for natural", position=0):
                if self.grayscale: frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                else:              frames = skvideo.io.vread(fname)
                local_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
                for i in tqdm.tqdm(range(frames.shape[0]), desc="video frames", position=1):
                    local_arr[i] = cv2.resize(frames[i], (self.shape[1], self.shape[0])) ## THIS IS NOT A BUG! cv2 uses (width, height)
                if self.arr is None:
                    self.arr = local_arr
                else:
                    self.arr = np.concatenate([self.arr, local_arr], 0)
                self.total_frames += local_arr.shape[0]
        else:
            self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
            total_frame_i = 0
            file_i = 0
            with tqdm.tqdm(total=self.total_frames, desc="Loading videos for natural") as pbar:
                while total_frame_i < self.total_frames:
                    if file_i % len(self.filelist) == 0: random.shuffle(self.filelist)
                    file_i += 1
                    fname = self.filelist[file_i % len(self.filelist)]
                    if self.grayscale: frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                    else:              frames = skvideo.io.vread(fname)
                    for frame_i in range(frames.shape[0]):
                        if total_frame_i >= self.total_frames: break
                        if self.grayscale:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0]))[..., None] ## THIS IS NOT A BUG! cv2 uses (width, height)
                        else:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0])) 
                        pbar.update(1)
                        total_frame_i += 1


    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img