import gymnasium as gym
from gymnasium import spaces

import numpy as np

from scipy.spatial.transform import Rotation as R

from pouring_env_base import PouringEnvBase

class PouringEnvRotating(PouringEnvBase):
  '''Custom Environment that follows gym interface'''
  metadata = {'render.modes': ['human']}

  def __init__(self, use_gui=False, spill_punish=1.5):
    jug_start_position = [0.1, 0.2, 0.1]
    jug_start_rotation = R.from_euler('XYZ', [90, 180, 0], degrees=True)
    jug_start_position.extend(jug_start_rotation.as_quat())

    super().__init__(use_gui, spill_punish, jug_start_position)

    # Define action and observation space
    # They must be gym.spaces objects
    # Action space: 3 dimensional vector
    # first vector = direction + velocity of movement (x, y, z)
    self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
    # observation space: position jug, position cup, particles
    self.observation_space = spaces.Tuple((spaces.Box(low=-5, high=5, shape=(7,)), # jug
                                          spaces.Box(low=-5, high=5, shape=(7,)), # cup
                                          spaces.Box(low=0, high=350, shape=(3,)))) # number of particles in cup, jug and spilled

  def step(self, action):
    # Execute one time step within the environment
    position_change = np.zeros(3)

    rotation_change = action

    self.simulation.next_position = [position_change, rotation_change]
    self.simulation.base.timeStepNoGUI()

    reward = self.__reward()
    observation = self.__observe()


    return observation, reward, self.done, {}
  
  def __observe(self): #TODO
    return None

  def __reward(self):
    n_particles_cup = self.simulation.n_particles_cup
    print('Cup: ', n_particles_cup)
    n_particles_spilled = self.simulation.n_particles_spilled
    print('Spilled: ', n_particles_spilled)

    reward = n_particles_cup - self.spill_punish * n_particles_spilled

    #TODO add distance metric between cup and jug?

    has_collided = self.simulation.collision

    if has_collided:
      print("collision")
      reward -= 1000
    
    return reward

  def reset(self):
    # Reset the state of the environment to an initial state
    # TODO maybe cleanup if reset?

    self.done = False

    # clear the command queue

    return self.jug_start_position #return initial state
    
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return NotImplementedError

if __name__ == "__main__":
  env = PouringEnvRotating(use_gui=False)
  #obs = env.reset(use_gui=True)

  while True:
      # Take a random action
      action = env.action_space.sample()
      print('Action: ', action)
      obs, reward, done, info = env.step(action)
      print('Observation space shape: ', env.observation_space.shape)
      print('Reward: ', reward)
      
      
      if done == True:
          break

  env.close()

