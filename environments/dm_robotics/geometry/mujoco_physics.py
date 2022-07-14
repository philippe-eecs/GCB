# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Physics implemented through MuJoCo."""

from typing import Callable

from dm_control import composer
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
import dm_env
from dm_robotics.geometry import geometry
import numpy as np

mjlib = mjbindings.mjlib


def from_env(env: composer.Environment) -> geometry.Physics:
  return _MujocoPhysics(lambda: env.physics)


def from_getter(getter: Callable[[], mjcf.Physics]) -> geometry.Physics:
  return _MujocoPhysics(getter)


def wrap(physics: mjcf.Physics) -> geometry.Physics:
  return _MujocoPhysics(lambda: physics)


class _MujocoPhysics(geometry.Physics):
  """Exposes `mjcf.Physics` as a `geometry.Physics`.

  Supports body, geom, and site elements.
  """

  def __init__(self, physics_getter: Callable[[], mjcf.Physics]):
    super().__init__()
    self._physics_getter = physics_getter

  def sync_before_step(self, timestep: dm_env.TimeStep):
    """No-op for compatibility with Physics. Assumes synced elsewhere."""
    pass

  def world_pose(self,
                 frame: geometry.Grounding,
                 get_pos: bool = True,
                 get_rot: bool = True) -> geometry.Pose:
    """Return world pose of the provided frame.

    Args:
      frame: A frame identifier.
      get_pos: If False, drop position entries.
      get_rot: If False, drop rotation entries.

    Returns:
      A `geometry.Pose` containing the requested pose.
    """
    if not isinstance(frame, mjcf.Element):
      raise ValueError('bad frame: {}, expected mjcf.Element'.format(frame))
    physics = self._physics_getter()
    hmat_world_element = np.eye(4)

    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
    if get_rot:
      xmat = None
      if frame.tag == 'geom':
        xmat = physics.named.data.geom_xmat[frame.full_identifier]
      elif frame.tag == 'site':
        xmat = physics.named.data.site_xmat[frame.full_identifier]
      elif frame.tag == 'camera':
        xmat = physics.named.data.cam_xmat[frame.full_identifier]
      elif frame.tag == 'body':
        xmat = physics.named.data.xmat[frame.full_identifier]
      else:
        raise ValueError(f'Frame {frame} is a {frame.tag} not a geom/site/'
                         'camera/body')
      hmat_world_element[0:3, 0:3] = xmat.reshape(3, 3)

    if get_pos:
      xpos = None
      if frame.tag == 'geom':
        xpos = physics.named.data.geom_xpos[frame.full_identifier]
      elif frame.tag == 'site':
        xpos = physics.named.data.site_xpos[frame.full_identifier]
      elif frame.tag == 'camera':
        xpos = physics.named.data.cam_xpos[frame.full_identifier]
      elif frame.tag == 'body':
        xpos = physics.named.data.xpos[frame.full_identifier]
      else:
        raise ValueError(f'Frame {frame} is a {frame.tag} not a geom/site/'
                         'camera/body')
      hmat_world_element[0:3, 3] = xpos

    return geometry.Pose.from_hmat(hmat_world_element)
