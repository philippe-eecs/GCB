import imageio
import os
import numpy as np
import glob
import wandb
import skvideo
#skvideo.setFFmpegPath('/global/scratch/users/hansenpmeche/KONDARAIL/bin/')
import skvideo.io as vid
from os import listdir
from os.path import isfile, join
from rlkit.core import logger

#from dmc2gym.natural_imgsource import RandomVideoSource


class MultiVideoRecorder(object):
    def __init__(self, dir_name = '', width=5, fps=30, white_pixel_pad = 0):
        self.dir_name = dir_name
        self.width = width
        self.fps = fps
        self.frames = None
        self.num_trajectories = None
        self.current_trajectory = 0
        self.white_pixel_pad = white_pixel_pad

    def init(self, num_trajectories = 10, max_trajectory_length = 75):
        self.frames = None
        self.num_trajectories = num_trajectories
        self.max_trajectory_length = max_trajectory_length
        self.current_trajectory = 0
        self.current_step = 0

    def step(self):
        if self.current_step != 0:
            x = self.current_trajectory // self.width
            y = self.current_trajectory % self.width
            if self.current_step < self.max_trajectory_length:
                self.frames[self.current_step: , :, x * self.img_height : (x + 1) * self.img_height, y * self.img_width : (y + 1) * self.img_width] = np.expand_dims(self.last_obs[self.channel_margin * 3 : (self.channel_margin + 1) * 3].copy(), axis = 0).repeat(self.max_trajectory_length - self.current_step,0) // 2
            self.current_trajectory += 1
            self.current_step = 0
        else:
            return

    def record(self, obs):
        #import pdb; pdb.set_trace()
        if self.frames is None:
            self.img_width = obs.shape[1]
            self.img_height = obs.shape[2]
            self.frames = np.ones((self.max_trajectory_length, 3, int(np.ceil(self.num_trajectories / self.width)) * (self.img_height) + int(np.ceil(self.num_trajectories / self.width) - 1) * (self.white_pixel_pad), self.width * self.img_width + (self.width - 1) * self.white_pixel_pad), dtype=np.uint8)
            self.channel_margin = (obs.shape[0] // 3) // 2
            
        x = self.current_trajectory // self.width
        y = self.current_trajectory % self.width
        self.frames[self.current_step, :, x * self.img_height : (x + 1) * self.img_height, y * self.img_width : (y + 1) * self.img_width] = (obs[self.channel_margin * 3 : (self.channel_margin + 1) * 3].copy())
        self.last_obs = obs
        self.current_step += 1


        
    def save(self, file_name, step):
        #stats = {
        #"train_step" : step,
        #file_name : wandb.Video(self.frames, fps = self.fps, format = 'gif')
        #}

        logger.logging_tool.save_gif(step, file_name, self.frames, self.fps)

        #wandb.log(stats)
    
    def save_img(self, file_name, step):
        #stats = {
        #"train_step" : step,
        #file_name : wandb.Video(self.frames, fps = self.fps, format = 'gif')
        #}
        #import pdb; pdb.set_trace()
        from PIL import Image
        im = Image.fromarray(self.frames[0].transpose((1, 2, 0)))
        im.save(file_name)

        #wandb.log(stats)



class VideoDistractor(object):

    def __init__(self, env, details, replay_buffer = None, ratio_start = 0, ratio_end = .5):

        self.video_format = details['video_format']
        self.pixels_to_cut = details['pixels_to_cut']
        #self.frames = details['vide']
        #self.random_seed = random_seed
        self.ratio_start = ratio_start
        self.ratio_end = ratio_end
        self.augmentation_function = details['augmentation']

        if details['augmentation'] == 'identity':
            self.augmentation_function = lambda x: x
        elif details['augmentation'] == 'vertical_reflection':
            self.augmentation_function = lambda x: np.flip(x, 2) #(Color, Height, Width)
        elif details['augmentation'] == 'fixed_rotation':
            pass
        else:
            raise NotImplementedError



        if details['video_format'] == 'replay':
            try:
                details['dataset_loc']
                dataset_load = True
            except:
                dataset_load = False

            if replay_buffer is not None and not dataset_load:
                replay = replay_buffer
            else:
                from goalbisim.data.management.init_replay import init_replay
                replay = init_replay(env.observation_space.shape, env.action_space.shape, 'cuda', details, state_shape = env.state_space.shape)
                replay.load(details['dataset_loc'], start = details['number_training_points'])

            self.frames = replay.obses
            self.goals = replay.goals
            self.frames_start_idx = replay.trajectory_start_idx
            self.frames_end_idx = replay.trajectory_end_idx
            self.start_idx = replay.sample_start
            assert replay.idx > 1000
            self.max_idx = replay.idx - 100 #Slack for any issues end of replay

            seed = np.random.randint(self.start_idx, self.max_idx)
            self.current_idx = self.frames_start_idx[seed]


            if details['video_format'] == 'replay' and dataset_load:
                del replay

        elif details['video_format'] == 'single_mp4':
            import skvideo.io  
            videodata = skvideo.io.vread(details['dataset_loc'])  
            from goalbisim.data.management.init_replay import init_replay
            replay = init_replay(env.observation_space.shape, env.action_space.shape, 'cuda', details, state_shape = env.state_space.shape)
            replay.video_populate(videodata, details['sequence_length'])

            self.frames = replay.obses.copy()
            self.goals = replay.goals.copy()
            self.frames_start_idx = replay.trajectory_start_idx.copy()
            self.frames_end_idx = replay.trajectory_end_idx.copy()
            self.start_idx = replay.sample_start
            assert replay.idx > 1000
            self.max_idx = replay.idx #Slack for any issues end of replay
            seed = np.random.randint(self.start_idx, self.max_idx)
            self.current_idx = self.frames_start_idx[seed]

            del videodata
            del replay

        elif details['video_format'] == 'stitch_mp4':
            obs_shape = replay_buffer.obses[0].shape
            onlyfiles = [f for f in listdir(details['dataset_loc']) if isfile(join(details['dataset_loc'], f))]
            total = len(onlyfiles)
            start_idx = int(self.ratio_start * total)
            end_idx = int(self.ratio_end * total)
            onlyfiles = onlyfiles[start_idx:end_idx]
            videodata = []
            for file in onlyfiles:
                try:
                    video = vid.vread(join(details['dataset_loc'], file))
                    videodata.append(video)
                except:
                    continue


            from goalbisim.data.management.init_replay import init_replay
            replay = init_replay(env.observation_space.shape, env.action_space.shape, 'cuda', details['replay_kwargs'], state_shape = env.state_space.shape, transforms = False)
            replay.video_populate(videodata, 75)

            self.frames = replay.obses
            self.goals = replay.goals
            self.frames_start_idx = replay.trajectory_start_idx
            self.frames_end_idx = replay.trajectory_end_idx
            self.start_idx = replay.sample_start
            self.max_idx = replay.idx #Slack for any issues end of replay
            seed = np.random.randint(self.start_idx, self.max_idx)
            self.current_idx = self.frames_start_idx[seed]
            
            del videodata
            #del replay
            

        else:
            raise NotImplementedError

    def rebin(self, a, shape):
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)




    def augment(self, obs, step = True, force_goal = False):

        obs = obs.squeeze().copy() #Might Solve?

        if force_goal:
            back = self.goals[self.current_idx].squeeze().copy()
        else:
            back = self.frames[self.current_idx].squeeze().copy()

        if self.augmentation_function is not None:
            back = self.augmentation_function(back).squeeze()

        from skimage.transform import resize
        back = resize(back.transpose((1, 2, 0)), (100, 100)).transpose((2, 0, 1))

        obs_idx = None
        for pixel in self.pixels_to_cut:
            if obs.shape[0] == 9:
                pixel = pixel * 3

            pixel = np.expand_dims(np.expand_dims(pixel, 1), 1)

            try:
                locs = np.where(obs == pixel)
                temp = []
                for i in range(3):
                    temp.append(obs_idx[i])
                locs = temp

                for i in range(3):
                    obs_idx[i] = np.concatenate([obs_idx[i], locs[i]])
            except:
                obs_idx = np.where(obs == pixel)
                temp = []
                for i in range(3):
                    temp.append(obs_idx[i])
                obs_idx = temp

        obs[tuple(obs_idx)] = back[tuple(obs_idx)] #Might Overwrite things?

        if step and self.current_idx < (self.frames_end_idx[self.current_idx] - 1):
            self.current_idx += 1

        return obs

    def next_augment(self, obs):
        return self.augment(obs, step = False, force_goal = False)

    def goal_augment(self, obs):
        return self.augment(obs, step = False, force_goal = True)

    def step_idx(self):
        if self.current_idx < self.frames_end_idx[self.current_idx]:
            self.current_idx += 1
        #self.current_idx += 1

    def step(self):
        seed = np.random.randint(self.start_idx, self.max_idx)
        self.current_idx = self.frames_start_idx[seed]




        



















