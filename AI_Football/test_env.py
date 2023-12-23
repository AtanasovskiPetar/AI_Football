import gymnasium as gym
import gym_envs.envs.grid_world
import gym_envs.envs.pendulum
import gym_envs.envs.ai_football

env1 = gym.make("gym_envs/GridWorld-v0")
env2 = gym.make("gym_envs/Pendulum-v0")
env2 = gym.make("AiFootball-v0")
env2 = gym.make("CartpoleCustom-v0")
print('Success')