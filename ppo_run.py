from stable_baselines3 import PPO
from gym_env import HydroponicEnv

env = HydroponicEnv()
model = PPO.load("ppo_hydroponic_model")

obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    # env.render()  # only if your env supports rendering
