import roboverse as rv
import numpy as np
import skvideo.io

env = rv.make(
    "SawyerRigAffordances-v4", 
    gui=True, 
    expl = False,
    random_color_p=1,
    max_episode_steps = 75,
    obs_img_dim = 64,
    claw_spawn_mode='fixed', #Maybe Uniform Instead, so distractor is more distracting....?
    drawer_yaw_setting = (0, 360),
    demo_action_variance = 0.2,
    color_range = (0, 255),
    drawer_bounding_x = [.46, .84],
    drawer_bounding_y = [0, .19],
    view_distance = 0.55,
    max_distractors = 4

    # expl = False,
    # random_color_p=1,
    # max_episode_steps = 150,
    # obs_img_dim = 64,
    # claw_spawn_mode='fixed', #Maybe Uniform Instead, so distractor is more distracting....?
    # #drawer_yaw_setting = (0, 360),
    # demo_action_variance = 0.1,
    # color_range = (0, 255),
    # max_distractors = 4,
    # #reset_interval=1,
    # #test_env=True,
    # #env_type='top_drawer',
    # #swap_eval_task=True,
)

ts = 75
save_video = True

if save_video:
    video_save_path = '/2tb/home/patrickhaoy/data/test/'
    num_traj = 5
    observations = np.zeros((num_traj*ts, 64, 64, 3))

count = 0
while count < num_traj*ts:
    env.demo_reset()
    for t in range(ts):
        if count >= num_traj*ts:
            break
        if save_video:
            img = np.uint8(env.render_obs())
            observations[count, :] = img
            count += 1
        action = env.get_demo_action(first_timestep=(t == 0), final_timestep=(t == ts - 1))
        next_observation, reward, done, info = env.step(action)
        if done:
            break
            #import pdb; pdb.set_trace()
    print(count)

if save_video:
    writer = skvideo.io.FFmpegWriter(video_save_path + "debug.mp4")
    for i in range(num_traj*ts):
            writer.writeFrame(observations[i, :, :, :])
    writer.close()