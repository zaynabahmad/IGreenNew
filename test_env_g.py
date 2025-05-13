import numpy as np
from gym_env import HydroponicEnv


import time

def run_random_agent(env, num_episodes=5, steps_per_episode=90, delay=0.5):
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        obs, _ = env.reset()
        total_reward = 0

        for step in range(steps_per_episode):
            action = {key: env.action_space[key].sample() for key in env.action_space.spaces}
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            print(f"Step {step + 1}:")
            print(f"  Action: {action}")
            print(f"  Observation: {obs}")
            print(f"  Reward = {reward:.4f}, Total Reward = {total_reward:.4f}\n")

            time.sleep(delay)  

            if done:
                print(f"Episode ended early at step {step + 1}")
                break

        print(f"Episode {episode + 1} completed with total reward: {total_reward:.4f}\n")

if __name__ == "__main__":
    env = HydroponicEnv()
    run_random_agent(env, num_episodes=3, steps_per_episode=90, delay=0.5)  

