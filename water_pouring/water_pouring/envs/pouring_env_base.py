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
        jerk_punish=0,
        action_punish=0,
        particle_explosion_punish=0,
        max_timesteps=500,
        use_fill_limit=False,
        jug_start_position=None,
        scene_file="scene.json",
        output_directory="SimulationOutput",
    ):
        self.cup_position = [0, 0, 0]
        cup_rotation = R.from_euler("XYZ", [-90, 0, 0], degrees=True)
        self.cup_position.extend(cup_rotation.as_quat())

        self.jug_upright_rotation = R.from_euler("XYZ", [90, 180, 0], degrees=True).as_euler("XYZ", degrees=True)
        self.jug_upright_rotation_internal = np.array(
            [0., 0., 0.]
        )  # rotation at which the jug is upright (needed as reference for rotation normalization)
        self.initial_position_internal = self.jug_upright_rotation_internal.copy()
        self.current_rotation_internal = self.initial_position_internal.copy()

        if jug_start_position is None:
            self.jug_start_position = [-0.04144097, -0.1583946, -0.22223671] # average taken from human data
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
        self.action_punish = action_punish
        self.particle_explosion_punish = particle_explosion_punish  # factor to punish exploding particles (high acceleration)
        self.time_step_punish = 0.1

        self.max_timesteps = max_timesteps

        self.max_spilled_particles = 350#100

        self.max_particles_cup = 175 # maximum amount of particles that fit into the cup

        self.use_fill_limit = use_fill_limit

        if self.use_fill_limit:
            percentage_fill = random.uniform(0.5, 1)
            self.max_fill = int(self.max_particles_cup * percentage_fill)
        else:
            self.max_fill = 350

        self.simulation = Simulation(
            self.use_gui, self.output_directory, self.jug_start_position, self.cup_position, self.scene_file
        )

        self.time_step = 0

        self.movement_bounds = [-0.5, 0.5]  # ((-0.5, 0.5), (0, 0.5), (-0.5, 0.5))

        self.rotation_bounds = [-180, 180]

        self.particle_bounds = [self.movement_bounds[0] - 0.1, self.movement_bounds[1] + 0.1]

        self.last_actions = [[[0,0,0], [0,0,0]], [[0,0,0], [0, 0, 0]], [[0,0,0],[0, 0, 0]]] # required for calculation of jerk

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=np.array([-0.005, -0.005, -0.005, -1., -1., -1.]), high=np.array([0.005, 0.005, 0.005, 1., 1., 1.]), shape=(6,))  # action space needs to be implemented for everything to run

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    low=-1,
                    high=1,
                    shape=(6,),
                    dtype=np.float64,
                ),  # jug
                spaces.Box(low=-1, high=1, shape=(350, 6), dtype=np.float64),  # position of particles
                #spaces.Box(low=-1, high=1, shape=(350, 3), dtype=np.float64),  # velocities of particles
                spaces.Box(low=-1, high=1, shape=(350,), dtype=np.float64), #distances from jug
                spaces.Box(low=-1, high=1, shape=(350,), dtype=np.float64), #distances from cup
                spaces.Box(low=-1, high=1, shape=(6,) if self.use_fill_limit else (5,), dtype=np.float64), # other features
            )
        )  # spaces.Box(low=0, high=self.max_timesteps, shape=(1,), dtype=np.float64)))
        # self.observation_space = spaces.utils.flatten_space(self.observation_space)

        self.last_particles_cup = 0
        self.last_particles_spilled = 0

        self.steps_per_action = 1

        self.max_jerk = 200000 #500000 (for full action space)

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
        action = [action[:3], action[3:]]
        # add the current action to the list and remove the oldest one (length always kept at 3)
        self.last_actions.insert(0, action)
        self.last_actions.pop()

        position_change = action[0]
        rotation_change = action[1]

        old_rotation_jug = R.from_quat(self.simulation.get_object_position(0)[3:]).as_euler("XYZ", degrees=True)

        step_position_change = np.array(position_change) / self.steps_per_action
        step_rotation_change = np.array(rotation_change) / self.steps_per_action

        for _ in range(self.steps_per_action):
            self.simulation.next_position = [step_position_change, step_rotation_change]

            self.simulation.base.timeStepNoGUI()
        
        new_rotation_jug = R.from_quat(self.simulation.get_object_position(0)[3:]).as_euler("XYZ", degrees=True)

        if not np.array_equal(old_rotation_jug, new_rotation_jug):
            self.current_rotation_internal += np.array(rotation_change)

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
        normalized_position = list(np.interp(jug_position[:3], self.movement_bounds, [-1, 1]))

        rot_diff = self.current_rotation_internal - self.jug_upright_rotation_internal
        normalized_rotation = np.interp(rot_diff, self.rotation_bounds, [-1, 1])

        normalized_position.extend(normalized_rotation)

        # normalize the particle positions and velocities
        particle_positions, particle_velocities = self.simulation.get_particle_positions_velocities()

        # sort the particles by euclidian distance to fixed point (corner of the particle boundary)
        reference_point = [self.particle_bounds[0], self.particle_bounds[0], self.particle_bounds[0]]
        distances = [np.linalg.norm(p - reference_point) for p in particle_positions]
        sorted_indices = np.argsort(distances)

        particle_positions = particle_positions[sorted_indices]
        particle_velocities = particle_velocities[sorted_indices]

        noise = np.random.normal(-0.001, 0.001)

        particle_positions += noise 
        particle_velocities += noise

        normalized_particle_positions = np.interp(particle_positions, self.particle_bounds, [-1, 1])
        particle_velocities_clipped = np.clip(particle_velocities, -1, 1)  # normalize particle velocities

        normalized_particle_data = np.concatenate([normalized_particle_positions, particle_velocities_clipped], axis=1)

        # append jug position to each particle
        #normalized_particle_data = np.hstack((normalized_particle_data, np.tile(normalized_position, (350, 1))))

        # calculate the distance of particles from jug and cup
        max_distance = np.linalg.norm(np.array([self.movement_bounds[1], self.movement_bounds[1], self.movement_bounds[1]]) - np.array([self.movement_bounds[0], self.movement_bounds[0], self.movement_bounds[0]]))
        distances_jug = [np.linalg.norm(p - jug_position[:3]) for p in particle_positions]
        normalized_distances_jug = np.interp(distances_jug, [0, max_distance], [-1, 1])

        cup_position = self.simulation.get_object_position(1)
        distances_cup = [np.linalg.norm(p - cup_position[:3]) for p in particle_positions]
        normalized_distances_cup = np.interp(distances_cup, [0, max_distance], [-1, 1])
        
        # calculate normalized timestep
        time_step = 2 * (self.time_step / self.max_timesteps) - 1

        other_features = np.array([
            2 * (self.simulation.n_particles_jug / self.max_particles) - 1, 
            2 * (self.simulation.n_particles_cup / self.max_particles) - 1, 
            2 * (self.simulation.n_particles_spilled / self.max_particles) - 1,
            2 * (self.simulation.n_particles_pouring / self.max_particles) - 1,
            time_step,
            ])

        if self.use_fill_limit:
            other_features = np.append(other_features, 2 * (self.max_fill / self.max_particles) - 1)

        # observation = np.append(jug_position, cup_position)
        # observation = np.append(observation, np.array([n_particles_jug, n_particles_cup, n_particles_spilled, n_particles_pouring]))
        # observation = [jug_position, cup_position, [n_particles_jug, n_particles_cup, n_particles_spilled, n_particles_pouring]]
        # observation = [jug_position, particle_positions, time_step]

        observation = (np.array(normalized_position), normalized_particle_data, np.array(normalized_distances_jug), np.array(normalized_distances_cup), other_features)  # , np.array([time_step]))
        #observation = (np.array(normalized_position), normalized_particle_positions, particle_velocities_clipped)
        # observation = (np.array(jug_pos), particle_positions_clipped)
        return observation

    def __reward(self):
        n_particles_cup = self.simulation.n_particles_cup
        n_particles_spilled = self.simulation.n_particles_spilled

        # calculate jerk
        jerk = self.calculate_jerk(self.last_actions)

        # punish actions
        action_magnitude = np.linalg.norm(self.last_actions[0])**2
        
        reward = self._calc_reward(n_particles_cup, n_particles_spilled, jerk, action_magnitude)

        if self.use_fill_limit:
            min_reward = min(self._calc_reward(0,self.max_particles, 10, np.linalg.norm(self.action_space.high)**2), self._calc_reward(self.max_particles_cup, self.max_particles, 10, np.linalg.norm(self.action_space.high)**2))

            # scale reward to range
            reward = np.interp(reward, [min_reward, 0], [-1,0])

        return reward
    
    def _calc_reward(self, n_particles_cup, n_particles_spilled, jerk, action_magnitude):
        reward = (
            self.hit_reward * (n_particles_cup / self.max_particles)
            - self.spill_punish * (n_particles_spilled / self.max_particles)
            - self.action_punish * action_magnitude
            - self.jerk_punish * jerk
            - self.time_step_punish
        )

        # keep range of rewards the same for all target fill levels (-> min-max normalization)
        if self.use_fill_limit:
            min_reward = 0
            max_reward = max((((self.max_fill - 0)/self.max_particles) ** 2), (((self.max_fill - self.max_particles_cup)/self.max_particles) ** 2))

            max_fill_reward = 50 * ((((self.max_fill - n_particles_cup)/self.max_particles) ** 2) - min_reward) / (max_reward - min_reward)
            #max_fill_reward = 200 * ((self.max_fill - n_particles_cup)/self.max_particles) ** 2
            reward -= max_fill_reward

        return reward

    def reset(self, options=None, seed=None):
        # Reset the state of the environment to an initial state
        if self.simulation.is_initialized:
            self.simulation.cleanup()

        if options is not None and "cleanup" in options.keys(): # required when opening and closing multiple environments in one script, otherwise the remaining simulation will cause a segmentation fault
            return None

        self.simulation.init_simulation()
        self.max_particles = self.simulation.get_number_of_particles()
        self.last_actions = [[[0,0,0], [0,0,0]], [[0,0,0], [0, 0, 0]], [[0,0,0],[0, 0, 0]]] # required for calculation of jerk

        self.time_step = 0

        self.last_particles_cup = 0
        self.last_particles_spilled = 0

        self.terminated = False
        self.truncated = False

        self.current_rotation_internal = self.initial_position_internal.copy()

        if self.use_fill_limit:
            if options is not None:
                self.max_fill = options['fixed fill goal']
            else:
                percentage_fill = random.uniform(0.5, 1)
                self.max_fill = int(self.max_particles_cup * percentage_fill)
            print('Target fill: ', self.max_fill)

        return (self.__observe(), {})

    def calculate_jerk(self, actions):
        timestep_size = self.simulation.time_step_size * self.steps_per_action

        actions = np.array(actions)
        acc_0 = (actions[0] - actions[1])/timestep_size
        acc_1 = (actions[1] - actions[2])/timestep_size

        jerk = (acc_0 - acc_1)/timestep_size
        
        jerk = np.linalg.norm(jerk)
        # scaling jerk to a useful size (would be in the thousands otherwise)
        jerk = np.interp(jerk, [0, self.max_jerk], [0, 10])

        return np.linalg.norm(jerk)

    def change_start_position(self, new_position):
        self.jug_start_position = new_position

        rot = R.from_quat(new_position[3:]).as_euler("XYZ", degrees=True)
        diff = rot - self.jug_upright_rotation
        self.initial_position_internal = self.jug_upright_rotation_internal + diff

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        return NotImplementedError
