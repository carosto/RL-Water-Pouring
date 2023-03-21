import gym
from gym import spaces

import numpy as np

from scipy.spatial.transform import Rotation as R

#from water_pouring.envs.pouring_env_base import PouringEnvBase

class PouringEnvRotatingWrapper(gym.ActionWrapper):

  def __init__(self, env):
    jug_start_position = [0.1, 0.2, 0.1]
    jug_start_rotation = R.from_euler('XYZ', [90, 180, 0], degrees=True)
    jug_start_position.extend(jug_start_rotation.as_quat())

    env.simulation.jug_start_position = jug_start_position

    super().__init__(env)

    # Define action and observation space
    # They must be gym.spaces objects
    # Action space: 3 dimensional vector
    # first vector = direction + velocity of movement (x, y, z)
    self.action_space = spaces.Box(low=-1, high=1, shape=(3,))

  def action(self, act):
          return [np.zeros(3), act]

  """def step(self, action):
    # Execute one time step within the environment
    position_change = np.zeros(3)

    rotation_change = action

    self.env.simulation.next_position = [position_change, rotation_change]
    self.env.simulation.base.timeStepNoGUI()

    reward = self.env.__reward()
    observation = self.env.__observe()


    return observation, reward, self.done, {}"""

