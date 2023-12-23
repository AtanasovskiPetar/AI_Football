import gymnasium as gym
import gym_envs.envs.ai_football
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env('AiFootball-v0', n_envs=16)
vec_env.reset()

# Model initialization
model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log="logs/Azure_PPO_logs")

TIMESTAMPS = 1000000
i = 1
while True:
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="ai_football_azure_logs")
    if i % 5 == 0:
        model.save(f'models/Azure_PPO/{TIMESTAMPS*i}')
    i += 1