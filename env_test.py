from gymnasium.envs.registration import register
from environment import HydroponicEnv

register(
    id='HydroponicEnv-v0',
    entry_point='environment:HydroponicEnv',
)

import gymnasium as gym

env = gym.make('HydroponicEnv-v0')
obs, _ = env.reset()
env.render()

action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
while True:
    env.render()
    import time
    time.sleep(0.1)