import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_env import HydroponicEnv  # make sure HydroponicEnv is gym-compliant
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor



# Create the environment
env = HydroponicEnv()
eval_env = Monitor(HydroponicEnv())


# Optional: Check if environment follows Gym API
# check_env(env, warn=True)

# Instantiate the PPO model
model = PPO(
    "MultiInputPolicy",  # Use MultiInputPolicy if your observation space is a dict or complex
    env,
    verbose=1,
    tensorboard_log="./ppo_hydroponic_tensorboard/",
    n_steps=1024,  # 2048
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    ent_coef=0.01,
    learning_rate=3e-4,
)


# Train the model
print("Starting PPO training...")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=5000,  # evaluate every 5000 steps
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

model.learn(total_timesteps=20000 , callback=eval_callback)

# Save the model
model.save("ppo_hydroponic_model")



# Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# Optionally: evaluate the model
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
