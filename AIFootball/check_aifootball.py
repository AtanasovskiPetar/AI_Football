import gymnasium as gym
import gym_envs.envs.ai_football
import time
import pygame

pygame.init()
pygame.display.set_mode((1366, 768), pygame.RESIZABLE)

env = gym.make("AiFootball")
env.reset()
# done = False
start = time.time()
episodes = 20
max_rew = -10
while True:
    action = env.action_space.sample()
    _, rewards, done, _, info = env.step(action)
    max_rew = max(rewards, max_rew)
    if done:
        print(f'Time: {time.time() - start} MaxReward: {max_rew}')
        max_rew = -10
        start = time.time()