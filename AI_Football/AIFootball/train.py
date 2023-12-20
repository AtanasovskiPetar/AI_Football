import numpy as np
import gymnasium as gym
import gym_envs.envs.ai_football
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env

# vec_env = make_vec_env('AiFootball-v0', n_envs=4)
# vec_env.reset()

env = gym.make("AiFootball-v0")
env.reset()
# Model initialization
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="PPO_logs", ent_coef=0.1)
# model = DDPG.load("models/ai_football_3/900000.zip", env=env, action_noise=action_noise)

TIMESTAMPS = 100000
i = 1
while True:
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="ai_football_1")
    model.save(f'models/PPO/ai_football_1/{i}00k')
    i += 1