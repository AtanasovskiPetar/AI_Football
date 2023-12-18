import gymnasium as gym
import gym_envs.envs.ai_football
from stable_baselines3 import DDPG
import pygame

pygame.init()
pygame.display.set_mode((1366, 768), pygame.RESIZABLE)

env = gym.make("AiFootball", render_mode='human')
env.reset()
model = DDPG.load("models/ai_football_3/900000.zip", env=env)

env = model.get_env()
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(f'Action: {action}')
    env.render()
