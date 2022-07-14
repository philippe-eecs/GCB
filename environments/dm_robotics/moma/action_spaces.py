# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action spaces for manipulation."""

from typing import Callable, List, Optional, Sequence

from dm_control import mjcf  # type: ignore
from dm_env import specs
from dm_robotics import agentflow as af
from dm_robotics.agentflow import spec_utils
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics
import numpy as np


def action_limits(
    cartesian_translation_action_limit: List[float],
    cartesian_rotation_action_limit: List[float]) -> List[float]:
  """Returns the action limits for the robot in this subtask."""
  if len(cartesian_translation_action_limit) == 3:
    cartesian_vals = cartesian_translation_action_limit
  else:
    assert len(cartesian_translation_action_limit) == 1
    cartesian_vals = cartesian_translation_action_limit * 3

  if len(cartesian_rotation_action_limit) == 3:
    euler_vals = cartesian_rotation_action_limit
  else:
    assert len(cartesian_rotation_action_limit) == 1
    euler_vals = cartesian_rotation_action_limit * 3

  return cartesian_vals + euler_vals


class _DelegateActionSpace(af.ActionSpace[specs.BoundedArray]):
  """Delegate action space.

  Base class for delegate action spaces that exist to just give a specific type
  for an action space object (vs using prefix slicer directly).
  """

  def __init__(self, action_space: af.ActionSpace[specs.BoundedArray]):
    self._action_space = action_space

  @property
  def name(self):
    return self._action_space.name

  def spec(self) -> specs.BoundedArray:
    return self._action_space.spec()

  def project(self, action: np.ndarray) -> np.ndarray:
    assert len(action) == self._action_space.spec().shape[0]
    return self._action_space.project(action)


class ArmJointActionSpace(_DelegateActionSpace):
  """Arm joint action space.

  This is just giving a type name to a particular action space.
  I.e. while it just delegates to the underlying action space, the name
  tells us that this projects from an arm joint space to some other space.
  """
  pass


class GripperActionSpace(_DelegateActionSpace):
  """Gripper action space.

  This is just giving a type name to a particular action space.
  I.e. while it just delegates to the underlying action space, the name
  tells us that this projects from a gripper joint space to some other space.
  """
  pass


class CartesianTwistActionSpace(_DelegateActionSpace):
  """Cartesian Twist action space.

  This is just giving a type name to a particular action space.
  I.e. while it just delegates to the underlying action space, the name
  tells us that this projects from a cartesian twist space to some other space.
  """
  pass


class RobotArmActionSpace(_DelegateActionSpace):
  """Action space for a robot arm.

  This is just giving a type name to a particular action space.
  I.e. while it just delegates to the underlying action space, the name
  tells us that this projects an action for the arm. This should be used when
  users don't care about the nature of the underlying action space (for example,
  joint action space or cartesian action space).
  """
  pass


class ReframeVelocityActionSpace(af.ActionSpace):
  """Transforms a twist from one frame to another."""

  def __init__(self,
               spec: specs.BoundedArray,
               physics_getter: Callable[[], mjcf.Physics],
               input_frame: geometry.Frame,
               output_frame: geometry.Frame,
               name: str = 'ReframeVelocity',
               *,
               velocity_dims: Optional[Sequence[int]] = None):
    """Initializes ReframeVelocityActionSpace.

    Args:
      spec: The spec for the original un-transformed action.
      physics_getter: A callable returning a mjcf.Physics
      input_frame: The frame in which the input twist is defined.
      output_frame: The frame in which the output twist should be expressed.
      name: A name for this action_space.
      velocity_dims: Optional sub-dimensions of the full twist which the action
        will contain. This can be used to allow `ReframeVelocityActionSpace` to
        operate on reduced-DOF cartesian action-spaces, e.g. the 4D used by the
        RGB-stacking tasks. To use, velocity_dims should have the same shape as
        `spec.shape`, and contain the indices of the twist that that action
        space uses (e.g. [0, 1, 2, 5] for `Cartesian4dVelocityEffector`).
    """
    self._spec = spec
    self._physics_getter = physics_getter
    self._physics = mujoco_physics.from_getter(physics_getter)
    self._input_frame = input_frame
    self._output_frame = output_frame
    self._name = name

    if velocity_dims is not None and np.shape(velocity_dims) != spec.shape:
      raise ValueError(f'Expected velocity_dims to have shape {spec.shape} but '
                       f'it has shape {np.shape(velocity_dims)}.')
    self._velocity_dims = velocity_dims

  @property
  def name(self) -> str:
    return self._name

  def spec(self) -> specs.BoundedArray:
    return self._spec

  def project(self, action: np.ndarray) -> np.ndarray:
    if self._velocity_dims is not None:
      # If action isn't a full twist (e.g. 4-DOF cartesian action space), expand
      # it for the frame-transformation operation.
      full_action = np.zeros(6)
      full_action[self._velocity_dims] = action
      action = full_action

    input_twist = geometry.TwistStamped(action, self._input_frame)
    output_twist = input_twist.to_frame(
        self._output_frame, physics=self._physics)
    output_action = output_twist.twist.full

    if self._velocity_dims is not None:
      # If action was reduced-dof, slice out the original dims.
      output_action = output_action[self._velocity_dims]

    return spec_utils.shrink_to_fit(value=output_action, spec=self._spec)
