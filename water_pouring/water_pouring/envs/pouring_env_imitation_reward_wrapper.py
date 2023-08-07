import gymnasium as gym
from gymnasium import spaces

import numpy as np

from scipy.spatial.transform import Rotation as R

class ImitationRewardWrapper(gym.RewardWrapper):

  def __init__(self, env, trajectory_path, weight_task_objective, weight_imitation, weight_position, weight_rotation):
    super().__init__(env)

    self.trajectory_path = trajectory_path
    self.trajectory = np.load(trajectory_path)
    
    assert weight_task_objective + weight_imitation == 1, "Weights for task goal and imitation do not sum up to 1."
    self.weight_task_objective = weight_task_objective
    self.weight_imitation = weight_imitation

    assert weight_position + weight_rotation == 1, "Weights for position and rotation do not sum up to 1."
    self.weight_position = weight_position
    self.weight_rotation = weight_rotation

    self.step_trajectory = 0

  def reward(self, reward):
    reference_pose = self.trajectory[self.step_trajectory]
    current_pose = self.env.simulation.get_object_position(0)

    # calculate quaternion difference of the two poses (-> rotation difference)
    rot_reference = R.from_quat(reference_pose[3:])
    rot_current = R.from_quat(current_pose[3:])

    r_rotation = np.exp(-2 * (rot_current * rot_reference.inv()).magnitude() ** 2)

    # calculate difference between positions 
    position_reference = reference_pose[:3]
    position_current = current_pose[:3]

    r_position = np.exp(-40 * np.linalg.norm(position_current - position_reference) ** 2)

    # calculate total imitation reward 
    r_imitation = self.weight_position * r_position + self.weight_rotation * r_rotation

    # calculate total reward
    r_total = self.weight_task_objective * reward + self.weight_imitation * r_imitation

    return r_total
    
