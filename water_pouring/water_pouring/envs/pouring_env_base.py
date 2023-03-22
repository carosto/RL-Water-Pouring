import gymnasium as gym
from gymnasium import spaces

import numpy as np

from scipy.spatial.transform import Rotation as R

from water_pouring.envs.simulation import Simulation

class PouringEnvBase(gym.Env):
  '''Custom Environment that follows gym interface'''
  metadata = {'render.modes': ['human']}

  def __init__(self, use_gui=False, spill_punish=1.5, jug_start_position=None):

    self.cup_position = [0, 0, 0] 
    cup_rotation = R.from_euler('XYZ', [-90, 0, 0], degrees=True)
    self.cup_position.extend(cup_rotation.as_quat())

    if jug_start_position is None:
      self.jug_start_position = [0.5, 0, 0.5]
      jug_start_rotation = R.from_euler('XYZ', [90, 180, 0], degrees=True)
      self.jug_start_position.extend(jug_start_rotation.as_quat())
    else:
      self.jug_start_position = jug_start_position

    self.output_directory = '../SimulationOutput'

    self.use_gui = use_gui

    self.spill_punish = spill_punish # factor to punish spilled particles

    # Define action and observation space
    # They must be gym.spaces objects
    self.action_space = spaces.Tuple((spaces.Box(low=-1, high=1, shape=(3,)),
                                      spaces.Box(low=-1, high=1, shape=(3,)))) # action space needs to be implemented for everything to run

    self.observation_space = spaces.Tuple((spaces.Box(low=-5, high=5, shape=(7,)), # jug
                                          spaces.Box(low=-5, high=5, shape=(7,)), # cup
                                          spaces.Box(low=0, high=10350, shape=(3,)))) # number of particles in jug, cup and spilled

    self.simulation = Simulation(self.use_gui, self.output_directory, self.jug_start_position,
                                    self.cup_position)

  def step(self, action):
    # Execute one time step within the environment
    """movement_vector = action[0][:3]
    movement_speed = action[0][3]
    normalized_movement = movement_vector / np.linalg.norm(movement_vector)
    position_change = normalized_movement * movement_speed

    rotation_vector = action[1][:3]
    rotation_speed = action[1][3]
    normalized_rotation = rotation_vector / np.linalg.norm(rotation_vector)
    rotation_change = normalized_rotation * rotation_speed"""

    position_change = action[0]
    rotation_change = action[1]

    self.simulation.next_position = [position_change, rotation_change]
    self.simulation.base.timeStepNoGUI()

    reward = self.__reward()
    observation = self.__observe()


    return observation, reward, self.terminated, self.truncated, {}
  
  def __observe(self):
    jug_position = self.simulation.get_object_position(0)
    cup_position = self.simulation.get_object_position(1)

    n_particles_cup = self.simulation.n_particles_cup
    n_particles_jug = self.simulation.n_particles_jug
    n_particles_spilled = self.simulation.n_particles_spilled

    return (jug_position, cup_position, [n_particles_jug, n_particles_cup, n_particles_spilled])

  def __reward(self): # currently identical to base
    n_particles_cup = self.simulation.n_particles_cup
    #print('Cup: ', n_particles_cup)
    n_particles_spilled = self.simulation.n_particles_spilled
    #print('Spilled: ', n_particles_spilled)

    reward = n_particles_cup - self.spill_punish * n_particles_spilled

    #TODO add distance metric between cup and jug?

    has_collided = self.simulation.collision

    if has_collided:
      print("collision")
      reward -= 1000
    
    return reward

  def reset(self, options=None, seed=None):
    # Reset the state of the environment to an initial state
    if self.simulation.is_initialized:
      self.simulation.cleanup()
    
    self.simulation.init_simulation()

    self.terminated = False
    self.truncated = False

    return (self.__observe(), {})#self.jug_start_position #return initial state
    
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return NotImplementedError

