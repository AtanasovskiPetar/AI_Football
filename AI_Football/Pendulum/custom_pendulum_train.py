import numpy as np
import gymnasium as gym
import gym_envs.envs.pendulum
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make("gym_envs/Pendulum-v0")
env.reset()

# The noise objcts for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="pendulum/PPO_logs")
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="PPO_logs")

TIMESTAMPS = 10000
i = 1
while True:
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="pendulum_custom")
    model.save(f'models/pendulum_custom/{TIMESTAMPS*i}')
    i += 1