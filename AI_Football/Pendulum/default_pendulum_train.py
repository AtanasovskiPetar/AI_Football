import gymnasium as gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make("Pendulum-v1", render_mode="rgb_array")
env.reset()

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="pendulum/PPO_logs")

TIMESTAMPS = 10000
i = 1
while True:
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="pendulum_default")
    model.save(f'pendulum/models/pendulum_default/{TIMESTAMPS*i}')
    i += 1