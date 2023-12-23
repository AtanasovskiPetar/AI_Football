import gymnasium as gym
import gym_envs.envs.cartpole
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import pygame



env = gym.make("CartpoleCustom-v0", render_mode='human')
env.reset()
model = PPO.load("models/170000.zip", env=env)

env = model.get_env()
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # print(f'Rew: {rewards}')
    # print(f'Action: {action}')
    env.render()
