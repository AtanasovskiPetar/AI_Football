import gymnasium as gym
import gym_envs.envs.cartpole
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartpoleCustom-v0", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="logs")
model.learn(total_timesteps=25000)

TIMESTAMPS = 5000
i = 1
while True:
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="cartpole_0")
    model.save(f'models/{TIMESTAMPS*i}')
    i += 1
