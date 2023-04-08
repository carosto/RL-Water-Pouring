import gymnasium as gym
from gymnasium import spaces

import numpy as np

from scipy.spatial.transform import Rotation as R

#from water_pouring.envs.pouring_env_base import PouringEnvBase

class XRotationWrapper(gym.ActionWrapper):

  def __init__(self, env):
    jug_start_position = [0, 0.2, -0.1]
    jug_start_rotation = R.from_euler('XYZ', [90, 180, 0], degrees=True)
    jug_start_position.extend(jug_start_rotation.as_quat())

    env.simulation.jug_start_position = jug_start_position

    super().__init__(env)

    # Define action and observation space
    # They must be gym.spaces objects
    # Action space: 1 dimensional vector
    # first vector = rotation on x axis
    self.action_space = spaces.Box(low=-1, high=1)

  def action(self, act):
          return [np.zeros(3), [act, 0, 0]]
