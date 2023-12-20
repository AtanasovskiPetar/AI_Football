import gymnasium as gym
import gym_envs.envs.ai_football
from stable_baselines3 import PPO
import pygame

pygame.init()
pygame.display.set_mode((1366, 768), pygame.RESIZABLE)

env = gym.make("AiFootball-v0", render_mode='human')
env.reset()
model = PPO.load("models/PPO/ai_football_1/4000k", env=env)

env = model.get_env()
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(f'Rew: {rewards}')
    print(f'Action: {action}')
    env.render()
