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

"""Defines Sawyer robot arm constants."""

import enum
import os


# Available actuation methods available for the Sawyer.
# In integrated velocity, the joint actuators receive a velocity that is
# integrated and a position controller is used to maintain the integrated
# joint configuration.
class Actuation(enum.Enum):
  INTEGRATED_VELOCITY = 0


# Number of degrees of freedom of the Sawyer robot arm.
NUM_DOFS = 7

# Joint names of Sawyer robot (without any namespacing). These names are defined
# by the Sawyer controller and are immutable.
JOINT_NAMES = ('right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4',
               'right_j5', 'right_j6')
WRIST_SITE_NAME = 'wrist_site'

# Effort limits of the Sawyer robot arm in Nm.
EFFORT_LIMITS = {
    'min': (-80, -80, -40, -40, -9, -9, -9),
    'max': (80, 80, 40, 40, 9, 9, 9),
}

JOINT_LIMITS = {
    'min': (-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124),
    'max': (3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124),
}

VELOCITY_LIMITS = {
    'min': (-1.74, -1.328, -1.957, -1.957, -3.485, -3.485, -4.545),
    'max': (1.74, 1.328, 1.957, 1.957, 3.485, 3.485, 4.545),
}

# Quaternion to align the attachment site with real
ROTATION_QUATERNION_MINUS_90DEG_AROUND_Z = (0.70711, 0, 0, 0.70711)

# Actuation limits of the Sawyer robot arm.
ACTUATION_LIMITS = {
    Actuation.INTEGRATED_VELOCITY: VELOCITY_LIMITS
}

# pylint: disable=line-too-long
_RETHINK_ASSETS_PATH = (os.path.join(os.path.dirname(__file__), '..',  '..', 'vendor', 'rethink', 'sawyer_description', 'mjcf'))
# pylint: enable=line-too-long

SAWYER_XML = os.path.join(_RETHINK_ASSETS_PATH, 'sawyer.xml')
SAWYER_PEDESTAL_XML = os.path.join(_RETHINK_ASSETS_PATH, 'sawyer_pedestal.xml')
