from gymnasium.envs.registration import register

register(
    id="gym_envs/GridWorld-v0",
    entry_point="gym_envs.envs:GridWorldEnv",
)

register(
    id="gym_envs/Pendulum-v0",
    entry_point="gym_envs.envs:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="AiFootball-v0",
    entry_point="gym_envs.envs:AiFootballEnv",
    max_episode_steps=1700,
)

register(
    id="CartpoleCustom-v0",
    entry_point="gym_envs.envs:CartPoleEnv",
    max_episode_steps=1700,
)