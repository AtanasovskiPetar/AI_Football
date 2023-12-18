import gymnasium as gym
import gym_envs.envs.pendulum
from stable_baselines3 import DDPG

env = gym.make("gym_envs/Pendulum-v0", render_mode="human")
env.reset()
model = DDPG.load("models/pendulum_custom/40000.zip", env=env)

# env = gym.make("Pendulum-v1", render_mode="rgb_array")
# env.reset()
# model = DDPG.load("models/pendulum_default/40000.zip", env=env)

env = model.get_env()
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")