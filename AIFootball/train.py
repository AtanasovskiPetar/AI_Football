import numpy as np
import gymnasium as gym
import gym_envs.envs.ai_football
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make("AiFootball-v0")
env.reset()
# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# Model initialization
# model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="logs")

model = DDPG.load("models/ai_football_3/900000.zip", env=env, action_noise=action_noise)

TIMESTAMPS = 10000
i = 91
while True:
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="ai_football_7")
    if i % 5 == 0:
        model.save(f'models/ai_football_3/{TIMESTAMPS * i}')
    i += 1