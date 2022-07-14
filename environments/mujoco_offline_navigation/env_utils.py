import labmaze
from labmaze import assets as labmaze_assets
from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.arenas import labmaze_textures, mazes

#from robots import A1
from mujoco_offline_navigation.task import CarNavigate

TIME_LIMIT = 1000


class FixedWallTexture(labmaze_textures.WallTextures):

    def _build(self, style):
        labmaze_textures = labmaze_assets.get_wall_texture_paths(style)
        self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)
        self._textures = []
        texture_name , texture_path = list(labmaze_textures.items())[3]
        self._textures.append(self._mjcf_root.asset.add(
                'texture', type='2d', name=texture_name,
                file=texture_path.format(texture_name)))

class FixedFloorTexture(labmaze_textures.FloorTextures):

    def _build(self, style):
        labmaze_textures = labmaze_assets.get_wall_texture_paths(style)
        self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)
        self._textures = []
        # print(list(labmaze_textures.items()))
        texture_name , texture_path = list(labmaze_textures.items())[3]
        self._textures.append(self._mjcf_root.asset.add(
                'texture', type='2d', name=texture_name,
                file=texture_path.format(texture_name)))


def make_dmc_env(task, spawn_x, spawn_y, goal_x, goal_y):
    #robot = A1()

    if task == 'walk':
        pass
        #task = Walk(robot)
    else:
        D4RL_MAZE_LAYOUT = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

        start_i, start_j = 1, 1
        goal_i, goal_j = 6, 6

        MAZE_LAYOUT = ""
        for i in range(len(D4RL_MAZE_LAYOUT)):
            for j in range(len(D4RL_MAZE_LAYOUT[i])):
                MAZE_LAYOUT += [' ', '*'][D4RL_MAZE_LAYOUT[i][j]]
            MAZE_LAYOUT += '\n'

        maze = labmaze.FixedMazeWithRandomGoals(entity_layer=MAZE_LAYOUT,
                                                num_spawns=1,
                                                num_objects=1)

        skybox_texture = labmaze_textures.SkyBox(style='sky_03')
        wall_textures = FixedWallTexture(style='style_01')
        floor_textures = FixedFloorTexture(style='style_02')

        arena = mazes.MazeWithTargets(maze=maze,
                                      xy_scale=1.0,
                                      z_height=1.0,
                                      skybox_texture=skybox_texture,
                                      wall_textures=wall_textures,
                                      floor_textures=floor_textures)

        task = CarNavigate(arena, maze, spawn_x, spawn_y, goal_x, goal_y)
    return composer.Environment(task, raise_exception_on_physics_error=False)


# def make_env(task, seed, save_folder=None):
#     import gym
#     from gym.wrappers import RescaleAction
#     from jaxrl import wrappers
#     from jaxrl.wrappers import VideoRecorder

#     env = make_dmc_env(task)

#     env = wrappers.DMCEnv(env=env, task_kwargs={'random': seed})
#     env = gym.wrappers.TimeLimit(env, TIME_LIMIT)
#     env = gym.wrappers.FlattenObservation(env)

#     env = wrappers.EpisodeMonitor(env)

#     env = RescaleAction(env, -1.0, 1.0)

#     if save_folder is not None:
#         env = VideoRecorder(env, save_folder=save_folder)

#     env = wrappers.SinglePrecision(env)

#     env.seed(seed)
#     env.action_space.seed(seed)
#     env.observation_space.seed(seed)

#     return env