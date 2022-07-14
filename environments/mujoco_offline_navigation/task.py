from csv import Dialect
from os import dup

import random

from dm_control.utils.transformations import euler_to_quat, quat_to_euler

import numpy as np
from dm_control import composer
from dm_control.entities.props.duplo import Duplo
from dm_control.locomotion import arenas
from dm_robotics.manipulation.props import mesh_object
from dm_robotics.moma import prop
import os
import matplotlib.image as im
from PIL import Image

from mujoco_offline_navigation.car import Car

DEFAULT_CONTROL_TIMESTEP = 0.1
DEFAULT_PHYSICS_TIMESTEP = 0.001

_TEST_ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'models')

objects = [np.array([-1.5, 1.5, 0]),
            np.array([-1.5, 2.5, 0]),
            np.array([-1.5, -1.5, 0]),
            np.array([-1.5, -2.5, 0]),
            np.array([4.5, 4.5, 0]),
            np.array([4.5, 0.5, 0]),
            np.array([4.5, -1.5, 0])]

def calc_dist(objects, pos, orient):
    res = []
    pos = pos[:2]
    for object in objects:
        object = object[:2]
        temp = ((object[1] - pos[1]) / (object[0] - pos[0]))
        t1 = np.arctan(temp)
        t2 = quat_to_euler(orient)[-1]

        if(np.abs(t1 - t2) < np.deg2rad(21)):
            res.append(np.linalg.norm(object - pos))
        else:
            res.append(np.Inf)
    
    return min(res)

def _create_texture(img_size):
    # img = np.random.normal(loc=0, scale=10, size=img_size)
    # print(img)
    img = Image.new('RGB', img_size, (150, 50, 10))
    return img


def _create_texture_file(texture_filename):
    # create custom texture and save it.
    img_size = [4, 4]
    texture = _create_texture(img_size=img_size)
    texture.save(texture_filename, format="png")
    # im.imsave(texture_filename, texture)

#   with open(texture_filename, 'wb') as f:
#     f.write(texture)

class CarNavigate(composer.Task):

    def __init__(self,
                 maze_arena,
                 bmaze,
                 spawn_x,
                 spawn_y,
                 goal_x,
                 goal_y,
                 physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep: float = DEFAULT_CONTROL_TIMESTEP):

        self._maze_arena = maze_arena
        #self._floor = arenas.Floor(size=(64, 64))

        self._car = Car()
        self._maze_arena.add_free_entity(self._car)

        

        texture_file1 = os.path.join(_TEST_ASSETS_PATH, 'tmp.png')
        texture_file2 = os.path.join(_TEST_ASSETS_PATH, 'tmp1.png')
        texture_file3 = os.path.join(_TEST_ASSETS_PATH, 'tmp2.png')
        texture_file4 = os.path.join(_TEST_ASSETS_PATH, 'tmp3.png')
        texture_file5 = os.path.join(_TEST_ASSETS_PATH, 'tmp4.png')
        texture_file6 = os.path.join(_TEST_ASSETS_PATH, 'tmp5.png')
        texture_file7 = os.path.join(_TEST_ASSETS_PATH, 'tmp6.png')
        _create_texture_file(texture_file7)

        mesh_file = os.path.join(_TEST_ASSETS_PATH, 'tree_ob.obj')
        print(mesh_file)
        self.p = mesh_object.MeshProp(name='tree1', visual_meshes=[mesh_file], texture_file=texture_file1, size=[0.1, 0.1, 0.1])
        self.prop_1 = prop.WrapperProp(self.p, name = 'tree12')
        frame = self._maze_arena.attach(self.prop_1)
        # self.prop_1.set_freejoint(frame.freejoint)

        self.p1 = mesh_object.MeshProp(name='tree12', visual_meshes=[mesh_file], texture_file=texture_file3, size=[0.1, 0.1, 0.1])
        self.prop_2 = prop.WrapperProp(self.p1, name = 'tree123')
        frame = self._maze_arena.attach(self.prop_2)
        # self.prop_2.set_freejoint(frame.freejoint)

        self.p2 = mesh_object.MeshProp(name='tree123', visual_meshes=[mesh_file], texture_file=texture_file2, size=[0.1, 0.1, 0.1])
        self.prop_3 = prop.WrapperProp(self.p2, name = 'tree1234')
        frame = self._maze_arena.attach(self.prop_3)

        self.p22 = mesh_object.MeshProp(name='tree1232', visual_meshes=[mesh_file], texture_file=texture_file4, size=[0.1, 0.1, 0.1])
        self.prop_32 = prop.WrapperProp(self.p22, name = 'tree1234')
        frame = self._maze_arena.attach(self.prop_32)

        self.p5 = mesh_object.MeshProp(name='tree5', visual_meshes=[mesh_file], texture_file=texture_file5, size=[0.1, 0.1, 0.1])
        self.prop_5 = prop.WrapperProp(self.p5, name = 'tre5')
        frame = self._maze_arena.attach(self.prop_5)

        self.p6 = mesh_object.MeshProp(name='tree6', visual_meshes=[mesh_file], texture_file=texture_file6, size=[0.1, 0.1, 0.1])
        self.prop_6 = prop.WrapperProp(self.p6, name = 'tre6')
        frame = self._maze_arena.attach(self.prop_6)

        self.p7 = mesh_object.MeshProp(name='tree7', visual_meshes=[mesh_file], texture_file=texture_file7, size=[0.1, 0.1, 0.1])
        self.prop_7 = prop.WrapperProp(self.p7, name = 'tre7')
        frame = self._maze_arena.attach(self.prop_7)

        self.p8 = mesh_object.MeshProp(name='tree8', visual_meshes=[mesh_file], texture_file=texture_file4, size=[0.1, 0.1, 0.1])
        self.prop_8 = prop.WrapperProp(self.p8, name = 'tre8')
        frame = self._maze_arena.attach(self.prop_8)

        self.p9 = mesh_object.MeshProp(name='tree9', visual_meshes=[mesh_file], texture_file=texture_file2, size=[0.1, 0.1, 0.1])
        self.prop_9 = prop.WrapperProp(self.p9, name = 'tre9')
        frame = self._maze_arena.attach(self.prop_9)

        # self.p10 = mesh_object.MeshProp(name='tree10', visual_meshes=[mesh_file], texture_file=texture_file5, size=[0.1, 0.1, 0.1])
        # self.prop_10 = prop.WrapperProp(self.p10, name = 'tre10')
        # frame = self._maze_arena.attach(self.prop_10)


        self.bmaze = bmaze

        observables = (
                        [self._car.observables.realsense_camera] +
                        [self._car.observables.body_position] +
                        [self._car.observables.body_rotation] +
                        self._car.observables.kinematic_sensors)
        for observable in observables:
            observable.enabled = True

        self._spawn_x = spawn_x
        self._spawn_y = spawn_y
        self._goal_x = goal_x
        self._goal_y = goal_y
        self._spawn_pt = []
        self._goal_pt = []

        self._prev_action = [0, 0]

        self.stop_timestep = 0

        self.trees = [[-1.5, 1.5, 0], [-1.5, 2.5, 0], [-1.5, -1.5, 0], [-1.5, -2.5, 0], [4.5, 4.5, 0], [4.5, 0.5, 0], [4.5, -1.5, 0]]
        #self._duplo = Duplo()
        #self._maze_arena.add_free_entity(self._duplo)
        self.set_timesteps(control_timestep, physics_timestep)

    def get_stop_timestep(self):
        return self.stop_timestep

    def set_stop_timestep(self, step):
        self.stop_timestep = step
    
    def get_prev_action(self):
        return self._prev_action
    
    def get_spawn_goal(self):
        return self._spawn_pt, self._goal_pt

    def get_reward(self, physics):
        body = physics.bind(self._car.mjcf_model.find('body', 'buddy'))
        pos = body.xpos
        pos[2] = 0

        return -np.linalg.norm(pos - self._goal_pt)

    def initialize_episode_mjcf(self, unused_random_state):
        self._maze_arena.regenerate()

    def initialize_episode(self, physics, random_state):
        self._failure_termination = False
        # self.prop_1.set_pose(physics, np.array([-1.5, 1.5, 0]), np.array(euler_to_quat([0, 0, 0])))
        # self.prop_2.set_pose(physics, np.array([-1.5, 2.5, 0]), np.array(euler_to_quat([0, 0, 0])))


        super().initialize_episode(physics, random_state)
        self._respawn(physics, random_state)

    def _respawn(self, physics, random_state):
        sx = np.random.randint(low = self._spawn_x[0], high=self._spawn_x[1]) + 0.5
        sy = np.random.randint(low = self._spawn_y[0], high=self._spawn_y[1]) + 0.5
        gx = np.random.randint(low = self._goal_x[0], high=self._goal_x[1]) + 0.5
        gy = np.random.randint(low = self._goal_y[0], high=self._goal_y[1]) + 0.5

        self._spawn_pt = np.array([sx, sy, 0])
        self._goal_pt = np.array([gx, gy, 0])

        # self._spawn_pt = np.array([3, -6.5, 0])
        # self._goal_pt = np.array([gx, gy, 0])

        mid = []
        mid_spawn = [4.5, 3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5, -4.5]
        mid_spawn_y = [2.5]

        mid_spawn_sp = [3.5, 2.5, 1.5, 0.5, -0.5, -1.5]

        sel = random.choice(mid_spawn_sp)

        for pt in mid_spawn:
            if pt != sel:
                # x = random.choice(mid_spawn)
                y = random.choice(mid_spawn_y)
                mid.append(np.array([y, pt, 0]))

        # end = []
        # mid_spawn = [4.5, 3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5, -4.5]
        # mid_spawn_y = [4.5]
        # for _ in range(3):
        #     x = random.choice(mid_spawn)
        #     mid_spawn.remove(x)
        #     y = random.choice(mid_spawn_y)
        #     end.append(np.array([y, x, 0]))

        self.trees = mid

        # spawn_points = [[-1.5, 1.5, 0], [-1.5, 2.5, 0], [-1.5, -1.5, 0], [-1.5, -2.5, 0], [4.5, 4.5, 0], [4.5, 0.5, 0], [4.5, -1.5, 0]]
        # for p in spawn_points:
        #     # print(p)
        #     maze_point = self._maze_arena.world_to_grid_positions([p])
        #     print("point real to grid point: ", p, maze_point)

        print("task trees are: ", self.trees)
        print(self._spawn_pt, self._goal_pt)

        self.prop_1.set_pose(physics, self.trees[0], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))
        self.prop_2.set_pose(physics, self.trees[1], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))
        self.prop_3.set_pose(physics, self.trees[2], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))
        self.prop_32.set_pose(physics, self.trees[3], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))
        self.prop_5.set_pose(physics, self.trees[4], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))
        self.prop_6.set_pose(physics, self.trees[5], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))
        self.prop_7.set_pose(physics, self.trees[6], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))
        self.prop_8.set_pose(physics, self.trees[7], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))
        self.prop_9.set_pose(physics, self.trees[8], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))
        # self.prop_10.set_pose(physics, self.trees[9], np.array(euler_to_quat([np.pi/2, 0, 2*np.pi])))


        self._car.set_pose(physics, self._spawn_pt)

    def after_step(self, physics, random_state):
        body = physics.bind(self._car.mjcf_model.find('body', 'buddy'))
        pos = body.xpos
        pos[2] = 0
        # print("pos is: ", np.shape(pos))
        # print("spawn is: ", self._spawn_pt)
        # print("goal is: ", np.shape(self._goal_pt))
        # print("d is", np.linalg.norm(pos - self._goal_pt))

        self._success_termination = np.linalg.norm(pos - self._goal_pt) < 0.2

    def before_step(self, physics, action, random_state):
        self._prev_action = action
        self._car.apply_action(physics, action, random_state)
        pass

    def should_terminate_episode(self, physics):
        return self._success_termination

    def get_discount(self, physics):
        if self._failure_termination:
            return 0.0
        else:
            return 1.0

    @property
    def root_entity(self):
        return self._maze_arena
