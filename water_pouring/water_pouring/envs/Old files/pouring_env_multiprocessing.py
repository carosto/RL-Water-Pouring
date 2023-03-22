import time
import gymnasium as gym
from gymnasium import spaces

import numpy as np

from scipy.spatial.transform import Rotation as R

import multiprocessing

from water_pouring.envs.simulation_multiprocessing import Simulation

class PouringEnv(gym.Env):
  '''Custom Environment that follows gym interface'''
  metadata = {'render.modes': ['human']}

  def __init__(self, use_gui=False, spill_punish=1.5):
    #super(CustomEnv, self).__init__()

    self.cup_position = [0, 0, 0] 
    cup_rotation = R.from_euler('XYZ', [-90, 0, 0], degrees=True)
    self.cup_position.extend(cup_rotation.as_quat())

    self.jug_start_position = [0.5, 0, 0.5]
    jug_start_rotation = R.from_euler('XYZ', [90, 180, 0], degrees=True)
    self.jug_start_position.extend(jug_start_rotation.as_quat())

    self.output_directory = '../SimulationOutput'

    self.use_gui = use_gui

    self.spill_punish = spill_punish # factor to punish spilled particles

    # Define action and observation space
    # They must be gym.spaces objects
    # Action space: 2 4-dimensional vectors
    # first vector = direction + velocity of movement (x, y, z, velocity)
    # second vector = direction of rotation (x, y, z, velocity)
    # TODO: set low and high (independent bounds are possible, given as list)
    self.action_space = spaces.Tuple((
                          spaces.Box(low=-0.1, high=0.1, shape=(4,)),
                          spaces.Box(low=0, high=1, shape=(4,))))

    self.observation_space = spaces.Tuple((
                              spaces.Box(low=-5, high=5, shape=(3,)),
                              spaces.Box(low=-10, high=10, shape=(4,)))) # TODO create useful observations (currently just using position and rotation of jug)

    #multiprocessing.set_start_method('spawn') # default on Windows, Linux default is fork
    self.simulation_manager = multiprocessing.Manager()
    self.command_queue = self.simulation_manager.Queue()
    self.communication_manager = self.simulation_manager.dict() #TODO check ob das entfernt werden kann!
    self.make_next_step = self.simulation_manager.Event()

    self.simulation_process = multiprocessing.Process(target=Simulation, 
                                    args=(self.use_gui, self.output_directory, self.jug_start_position,
                                    self.cup_position, self.command_queue, self.communication_manager, self.make_next_step))
    self.simulation_process.daemon = True # make sure the simulation process exits when the main process ends

    self.reset()
  
  def seed(self, seed):
    np.random.seed(seed) #TODO needed for the TD3 implementation?

  def step(self, action):
    # wait for the simulation if it is still executing a step
    self.make_next_step.wait()
    self.make_next_step.clear()

    # Execute one time step within the environment
    movement_vector = action[0][:3]
    movement_speed = action[0][3]
    normalized_movement = movement_vector / np.linalg.norm(movement_vector)
    position_change = normalized_movement * movement_speed

    rotation_vector = action[1][:3]
    rotation_speed = action[1][3]
    normalized_rotation = rotation_vector / np.linalg.norm(rotation_vector)
    rotation_change = normalized_rotation * rotation_speed

    self.communication_manager['step'] = [position_change, rotation_change]
    self.command_queue.put('STEP')

    self.make_next_step.wait() # to make sure the correct values are used for the reward
    reward = self.__reward()
    observation = self.__observe()


    self.make_next_step.clear()
    return observation, reward, self.done, {}
  
  def __observe(self): #TODO
    return self.communication_manager['current_jug_pose']

  def __reward(self):
    n_particles_cup = self.communication_manager['n_particles_cup']
    n_particles_spilled = self.communication_manager['n_particles_spilled']

    reward = n_particles_cup - self.spill_punish * n_particles_spilled

    #TODO add distance metric between cup and jug?

    has_collided = self.communication_manager['has_collided']

    if has_collided:
      print("collision")
      reward -= 1000
    
    return reward

  def reset(self):
    # Reset the state of the environment to an initial state
    # TODO maybe cleanup if reset?
    if self.simulation_process.is_alive():
      self.stop_simulation_process()

    self.done = False
    self.communication_manager.clear()

    # clear the command queue
    while not self.command_queue.empty():
      self.command_queue.get()

    self.simulation_process.start()

    return self.jug_start_position #return initial state
    
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return NotImplementedError

  def stop_simulation_process(self):
    self.command_queue.put('STOP')

    # allowing the simulation process to shut down within 5 seconds, otherwise it is terminated by the main process 
    start = time.time()
    while time.time() - start <= 5:
      # if the simulation process has ended within the time limit
      if not self.simulation_process.is_alive():
        break
    else: # If the process was not terminated within the time limit, it is terminated manually
      self.simulation_process.terminate()
      self.simulation_process.join()


if __name__ == "__main__":
  env = PouringEnv(use_gui=False)
  #obs = env.reset(use_gui=True)

  print(env.observation_space.shape)
  while True:
      # Take a random action
      action = env.action_space.sample()
      obs, reward, done, info = env.step(action)
      print(env.observation_space.shape)
      print(reward)
      
      
      if done == True:
          break

  env.close()

