import gymnasium as gym
from gymnasium import spaces

import numpy as np
import random

from scipy.spatial.transform import Rotation as R

from water_pouring.envs.simulation import Simulation


class PouringEnvBase(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        use_gui=False,
        spill_punish=0.1,
        hit_reward=0.1,
        jerk_punish=0.1,
        particle_explosion_punish=0.1,
        max_timesteps=500,
        jug_start_position=None,
        scene_file="scene.json",
        output_directory="SimulationOutput",
    ):
        self.cup_position = [0, 0, 0]
        cup_rotation = R.from_euler("XYZ", [-90, 0, 0], degrees=True)
        self.cup_position.extend(cup_rotation.as_quat())

        self.jug_upright_rotation = R.from_euler("XYZ", [90, 180, 0], degrees=True).as_euler("XYZ", degrees=True)
        self.jug_upright_rotation_internal = np.array(
            [0, 0, 0]
        )  # rotation at which the jug is upright (needed as reference for rotation normalization)
        self.initial_position_internal = self.jug_upright_rotation_internal.copy()
        self.current_rotation_internal = self.initial_position_internal.copy()

        if jug_start_position is None:
            self.jug_start_position = [0.5, 0, 0.5]
            jug_start_rotation = R.from_euler("XYZ", [90, 180, 0], degrees=True)
            self.jug_start_position.extend(jug_start_rotation.as_quat())
        else:
            self.jug_start_position = jug_start_position

        self.output_directory = f"../{output_directory}"

        self.use_gui = use_gui

        self.scene_file = scene_file

        self.spill_punish = spill_punish  # factor to punish spilled particles
        self.hit_reward = hit_reward  # factor to reward particles in the cup
        self.jerk_punish = jerk_punish  # factor to punish jerky movements
        self.particle_explosion_punish = (
            particle_explosion_punish  # factor to punish exploding particles (high acceleration)
        )
        self.time_step_punish = 1

        self.max_timesteps = max_timesteps

        self.max_spilled_particles = 20

        self.simulation = Simulation(
            self.use_gui, self.output_directory, self.jug_start_position, self.cup_position, self.scene_file
        )

        self.time_step = 0

        self.movement_bounds = [-0.5, 0.5]  # ((-0.5, 0.5), (0, 0.5), (-0.5, 0.5))

        self.rotation_bounds = [-180, 180]

        self.particle_bounds = [self.movement_bounds[0] - 1, self.movement_bounds[1] + 1]

        """self.last_poses = [
            self.jug_start_position,
            self.jug_start_position,
            self.jug_start_position,
        ]  # required for calculation of jerk"""

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Tuple(
            (spaces.Box(low=-1, high=1, shape=(3,)), spaces.Box(low=-1, high=1, shape=(3,)))
        )  # action space needs to be implemented for everything to run

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    low=-1,
                    high=1,
                    shape=(6,),
                    dtype=np.float64,
                ),  # jug
                spaces.Box(low=-1, high=1, shape=(350, 6), dtype=np.float64),  # position and velocity of particles
            )
        )  # spaces.Box(low=0, high=self.max_timesteps, shape=(1,), dtype=np.float64)))
        # self.observation_space = spaces.utils.flatten_space(self.observation_space)

        self.last_hit_reward_results = 0
        self.last_spill_punish_results = 0

        self.steps_per_action = 1

    def seed(self, seed):
        """
        Seeding might not work perfectly for this environment
        because the simulator does not behave exactly the same
        every time.
        Args:
            seed: An integer seed for the random number generator.
        """
        np.random.seed(seed)
        random.seed(seed)

    def step(self, action):
        # Execute one time step within the environment
        self.last_action = action

        position_change = action[0]
        rotation_change = action[1]

        self.current_rotation_internal += np.array(rotation_change)

        step_position_change = np.array(position_change) / self.steps_per_action
        step_rotation_change = np.array(rotation_change) / self.steps_per_action

        for _ in range(self.steps_per_action):
            self.simulation.next_position = [step_position_change, step_rotation_change]

            self.simulation.base.timeStepNoGUI()

        # self.simulation.next_position = [position_change, rotation_change]

        # self.simulation.base.timeStepNoGUI()

        reward = self.__reward()
        observation = self.__observe()

        self.time_step += 1

        # check if time step limit was reached
        if self.time_step >= self.max_timesteps:
            print("Reached step limit")
            self.terminated = True

        # check if jug was moved outside the bounds
        obs_position = observation[0][:3]
        if any(abs(x) >= 1 for x in obs_position):
            print("Out of movement bounds")
            self.terminated = True

        """self.last_poses.pop(0)  # remove the oldest position, keep the list always at length 3
        self.last_poses.append(jug_position)"""

        # check if jug was rotated outside the bounds
        obs_rotation = observation[0][3:]
        if any(abs(x) >= 1 for x in obs_rotation):
            print("Out of rotation bounds")
            self.terminated = True

        # done when all particles have poured out
        if self.simulation.n_particles_jug == 0 and self.simulation.n_particles_pouring == 0:
            print("Poured all particles")
            self.terminated = True

        # done if too many particles have been spilled
        if self.simulation.n_particles_spilled >= self.max_spilled_particles:
            print("Too many particles spilled")
            self.terminated = True

        return observation, reward, self.terminated, self.truncated, {}

    def __observe(self):
        jug_position = self.simulation.get_object_position(0)

        # normalize the jug position and rotation for the observation
        pos = jug_position[:3]
        normalized_position = list(np.interp(pos, self.movement_bounds, [-1, 1]))

        rot_diff = self.current_rotation_internal - self.jug_upright_rotation_internal
        normalized_rotation = np.interp(rot_diff, self.rotation_bounds, [-1, 1])

        normalized_position.extend(normalized_rotation)

        # normalize the particle positions and velocities
        particle_positions, particle_velocities = self.simulation.get_particle_positions_velocities()
        normalized_particle_positions = np.interp(particle_positions, self.particle_bounds, [-1, 1])
        particle_velocities_clipped = np.clip(particle_velocities, -1, 1)  # normalize particle velocities

        normalized_particle_data = np.concatenate([normalized_particle_positions, particle_velocities_clipped], axis=1)

        time_step = self.time_step / self.max_timesteps

        # observation = np.append(jug_position, cup_position)
        # observation = np.append(observation, np.array([n_particles_jug, n_particles_cup, n_particles_spilled, n_particles_pouring]))
        # observation = [jug_position, cup_position, [n_particles_jug, n_particles_cup, n_particles_spilled, n_particles_pouring]]
        # observation = [jug_position, particle_positions, time_step]

        observation = (np.array(normalized_position), normalized_particle_data)  # , np.array([time_step]))
        # observation = (np.array(jug_pos), particle_positions_clipped)
        return observation

    def __reward(self):  # currently identical to base
        n_particles_cup = self.simulation.n_particles_cup
        # print('Cup: ', n_particles_cup)
        # print('Jug: ', self.simulation.n_particles_jug)
        n_particles_spilled = self.simulation.n_particles_spilled
        # print('Spilled: ', n_particles_spilled)
        # print('Pouring: ', self.simulation.n_particles_pouring)

        # calculate jerk
        current_pose = self.simulation.get_object_position(0)
        current_position = current_pose[:3]
        current_rotation = R.from_quat(current_pose[3:]).as_euler("XYZ", degrees=True)

        """last_positions = [x[:3] for x in self.last_poses]
        last_rotations = [R.from_quat(x[3:]).as_euler("XYZ", degrees=True) for x in self.last_poses]"""

        # penalize high acceration for particles (-> exploding)
        particle_accelerations = self.simulation.get_particle_accelerations()
        acceleration_magnitudes = np.linalg.norm(particle_accelerations, axis=1)
        average_acceleration = np.mean(acceleration_magnitudes)

        # jerk for position
        # jerk_position = np.linalg.norm(self.approx_3rd_derivative(current_position, last_positions, self.simulation.time_step_size))
        jerk = np.linalg.norm(self.last_action) ** 2  # [0]**2

        # jerk for rotation
        # jerk_rotation = np.linalg.norm(self.approx_3rd_derivative(current_rotation, last_rotations, self.simulation.time_step_size))
        # TODO taken from yannik, check! (reward of previous step is taken into account)
        # reward: only newly spilled/hit particles are counted
        hit_reward_result = self.hit_reward * n_particles_cup

        spill_punish_result = self.spill_punish * n_particles_spilled

        reward = (
            (hit_reward_result - self.last_hit_reward_results)
            - (spill_punish_result - self.last_spill_punish_results)
            - self.jerk_punish * jerk
        ) - self.time_step_punish  # - self.particle_explosion_punish * average_acceleration#(jerk_position + jerk_rotation)

        # reward = hit_reward_result - spill_punish_result - self.jerk_punish * jerk - self.time_step_punish

        if np.isnan(reward):
            print(n_particles_cup)
            print(n_particles_spilled)
            print(jerk)
            print(np.array(particle_accelerations).shape)
            print(acceleration_magnitudes)
            print(average_acceleration)
        # TODO add distance metric between cup and jug?
        """
        has_collided = self.simulation.collision

        if has_collided:
            print("collision")
            reward -= 1000"""

        self.last_hit_reward_results = hit_reward_result
        self.last_spill_punish_results = spill_punish_result
        return reward

    def reset(self, options=None, seed=None):
        # Reset the state of the environment to an initial state
        if self.simulation.is_initialized:
            self.simulation.cleanup()

        self.simulation.init_simulation()
        self.max_particles = self.simulation.get_number_of_particles()
        """self.last_poses = [
            self.jug_start_position,
            self.jug_start_position,
            self.jug_start_position,
        ]  # required for calculation of jerk"""
        self.time_step = 0

        self.last_hit_reward_results = 0
        self.last_spill_punish_results = 0

        self.terminated = False
        self.truncated = False

        self.current_rotation_internal = self.initial_position_internal.copy()

        return (self.__observe(), {})

    def approx_3rd_derivative(self, current_value, previous_values, time):
        # required for jerk calculation
        # using backward difference (https://de.mathworks.com/matlabcentral/answers/496527-how-calculate-the-second-and-third-numerical-derivative-of-one-variable-f-x)
        # print(np.array(current_value) - np.array(previous_values[2]) + -3 * np.array(previous_values[0]) + 3 * np.array(previous_values[1]))
        # print((current_value - 3 * np.array(previous_values[0]) + 3 * np.array(previous_values[1]) - previous_values[2]))
        # (current_value - 3 * np.array(previous_values[0]) + 3 * np.array(previous_values[1]) - previous_values[2]) / (time**3)
        return np.around(
            (
                np.array(current_value)
                - np.array(previous_values[2])
                + -3 * np.array(previous_values[0])
                + 3 * np.array(previous_values[1])
            )
            / (time**3),
            2,
        )

    def change_start_position(self, new_position):
        self.jug_start_position = new_position

        rot = R.from_quat(new_position[3:]).as_euler("XYZ", degrees=True)
        diff = rot - self.jug_upright_rotation
        self.initial_position_internal = self.jug_upright_rotation_internal + diff

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        return NotImplementedError
