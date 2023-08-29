# script to generate jug positions from trajectory (required for deep mimic)

from scipy.spatial.transform import Rotation as R
import numpy as np
import os
from water_pouring.envs.simulation import Simulation

def get_pose_trajectory_from_actions(action_trajectory):
    trajectory_actions = np.load(action_trajectory)

    pose_trajectory = []

    jug_start_position = [0.5, 0, 0.5]
    jug_start_rotation = R.from_euler("XYZ", [90, 180, 0], degrees=True)
    jug_start_position.extend(jug_start_rotation.as_quat())

    cup_position = [0, 0, 0]
    cup_rotation = R.from_euler("XYZ", [-90, 0, 0], degrees=True)
    cup_position.extend(cup_rotation.as_quat())

    simulation = Simulation(
            False, "SimulationOutput", jug_start_position, cup_position, "scene.json"
        )
    simulation.init_simulation()

    for action in trajectory_actions:
        try:
            _ = action[0]
        except:
            action = [np.zeros(3), [action, 0, 0]]
        position_change = action[0]
        rotation_change = action[1]

        step_position_change = np.array(position_change) / 1
        step_rotation_change = np.array(rotation_change) / 1

        for _ in range(1):
            simulation.next_position = [step_position_change, step_rotation_change]

            simulation.base.timeStepNoGUI()
        
        jug_pose = simulation.get_object_position(0) 
        pose_trajectory.append(jug_pose)
    simulation.cleanup()
    return pose_trajectory

path = "PoseTrajectories"

if not os.path.exists(f"./{path}"):
    os.makedirs(f"./{path}")

trajectory = get_pose_trajectory_from_actions("/home/carola/bachelorthesis/TD3/RandomTrajectories/random_trajectory_10.npy")

trajectory = np.array(trajectory)
#np.save(f"{path}/x_rotation_10.npy", trajectory)

