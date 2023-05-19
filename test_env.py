from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper
import gymnasium as gym
import numpy as np
import sys


env = gym.make("WaterPouringEnvBase-v0")
wrapped_env = XRotationWrapper(env)
#env = PouringEnvBase(use_gui=False)
obs = wrapped_env.reset()

print("The initial observation is {}".format(obs))

s = 0
angle = 90

while True:
    # Take a random action
    if s < 10:
        action = np.array([0])
    else:
        action = np.array([0.1])#wrapped_env.action_space.sample()

    s += 1
    angle += action[0]
    print(angle)

    if angle > 140:
        #wrapped_env.simulation.save_particles()
        break
    print('Action: ', action)
    print(wrapped_env.simulation.n_particles_jug)
    print(wrapped_env.simulation.n_particles_cup)
    print(wrapped_env.simulation.n_particles_pouring)
    print(wrapped_env.simulation.n_particles_spilled)
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print('Reward: ', reward)
    #print("The new observation is {}".format(obs))
    
    if terminated or truncated:
        print('Done')
        break

wrapped_env.close()