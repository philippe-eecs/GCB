import roboverse as rv
import numpy as np
import skvideo.io
from PIL import Image
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--video_save_path', type=str, default='/2tb/home/patrickhaoy/data/test/')
parser.add_argument('--num_traj', type=int, default=1)
parser.add_argument('--obs_img_dim', type=int, default=256)
parser.add_argument('--seeds', type=int, default=256)
args = parser.parse_args()

video_save_path = args.video_save_path
num_traj = args.num_traj
obs_img_dim = args.obs_img_dim
seeds = args.seeds

env = rv.make(
    "SawyerRigAffordances-v1", 
    gui=False, 
    expl = False,
    random_color_p=1,
    max_episode_steps = 75,
    obs_img_dim = obs_img_dim,
    claw_spawn_mode ='fixed', #Maybe Uniform Instead, so distractor is more distracting....?
    drawer_yaw_setting = (0, 360),
    demo_action_variance = 0.2,
    color_range = (0, 255),
    drawer_bounding_x = [.46, .84],
    drawer_bounding_y = [-.19, .19],
    view_distance = 0.425,
    max_distractors = 4
)

for seed in seeds:
    ts = env.max_episode_steps
    observations = np.zeros((num_traj*ts, obs_img_dim, obs_img_dim, 3))

    count = 0
    for _ in range(num_traj):
        env.demo_reset(seed=seed)
        for t in range(ts):
            if count >= num_traj*ts:
                break
            img = np.uint8(env.render_obs())
            observations[count, :] = img
            count += 1
            action = env.get_demo_action(first_timestep=(t == 0), final_timestep=(t == ts - 1))
            next_observation, reward, done, info = env.step(action)
            if done:
                break
                #import pdb; pdb.set_trace()

    observations = observations[:count]

    writer = skvideo.io.FFmpegWriter(os.path.join(video_save_path, "rollout_seed{}.mp4".format(seed)))
    for i in range(count):
        writer.writeFrame(observations[i, :, :, :])
    writer.close()
