import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pygame

class HydroponicEnv(gym.Env):
    def __init__(self):
        super(HydroponicEnv, self).__init__()

        # Observation space
        self.observation_space = spaces.Dict({
            "plant_type": spaces.Discrete(3),         # Category: 0, 1, 2 (3 types)
            "plant_stage": spaces.Discrete(3),        # Stage: 0 (seed), 1 (growing), 2 (mature)
            "temp": spaces.Box(low=10, high=40, shape=(1,), dtype=np.float32),
            "humidity": spaces.Box(low=30, high=90, shape=(1,), dtype=np.float32),
            "light": spaces.Box(low=0, high=20000, shape=(1,), dtype=np.float32),
            "ec": spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32),
            "ph": spaces.Box(low=5.0, high=8.0, shape=(1,), dtype=np.float32)
        })

        # Action space
        self.action_space = spaces.Dict({
            "watering_cycles": spaces.Discrete(11),   # 0 to 10
            "watering_period": spaces.Box(low=30, high=1440, shape=(1,), dtype=np.float32),
            "light": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            "temp": spaces.Box(low=10, high=40, shape=(1,), dtype=np.float32),
            "ec": spaces.Box(low=0.0, high=6.0, shape=(1,), dtype=np.float32),
            "ph": spaces.Box(low=5.0, high=8.0, shape=(1,), dtype=np.float32),
            "humidity" : spaces.Box(low =20 , high = 100 , shape=(1,) , dtype=np.float32)  # zaynap
        })

        self.state = None
        self.episode_length = 100   # mmkn ab2a a5yrha l 90
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly initialize the environment state
        self.state = {
            "plant_type": self.observation_space["plant_type"].sample(),
            "plant_stage": self.observation_space["plant_stage"].sample(),
            "temp": np.array([np.random.uniform(10, 40)], dtype=np.float32),
            "humidity": np.array([np.random.uniform(30, 90)], dtype=np.float32),
            "light": np.array([np.random.uniform(0, 1000)], dtype=np.float32),
            "ec": np.array([np.random.uniform(0.0, 5.0)], dtype=np.float32),
            "ph": np.array([np.random.uniform(5.0, 8.0)], dtype=np.float32)
        }
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # Dummy reward function: you’ll replace this later with a real optimization goal
        reward = np.random.rand()

        terminated = self.current_step >= self.episode_length
        truncated = False
        self.last_action = action
        self.last_reward = reward
        # Apply action to the state (later we can define how it affects plant growth)
        # For now, we skip state transitions to keep it simple

        return self.state, reward, terminated, truncated, {}

    def render(self, mode="human"):
        if not hasattr(self, 'window'):
            pygame.init()
            self.window_size = (800, 600)
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Hydroponic Farm RL")
            self.font = pygame.font.SysFont(None, 24)

            # Fixed icon size
            self.icon_size = (32, 32)
            self.icons = {
                "temp": pygame.transform.scale(pygame.image.load(os.path.join("assets", "Temp.png")), self.icon_size),
                "humidity": pygame.transform.scale(pygame.image.load(os.path.join("assets", "Humidity.png")), self.icon_size),
                "light": pygame.transform.scale(pygame.image.load(os.path.join("assets", "Light.png")), self.icon_size),
                "ec": pygame.transform.scale(pygame.image.load(os.path.join("assets", "EC.png")), self.icon_size),
                "ph": pygame.transform.scale(pygame.image.load(os.path.join("assets", "PH.png")), self.icon_size),
            }

            self.plant_drawings = {
                "0_0": pygame.image.load(os.path.join("assets", "phase1.png")),
                "0_1": pygame.image.load(os.path.join("assets", "phase2.png")),
                "0_2": pygame.image.load(os.path.join("assets", "phase3.png")),
                "1_0": pygame.image.load(os.path.join("assets", "phase1.png")),
                "1_1": pygame.image.load(os.path.join("assets", "phase2.png")),
                "1_2": pygame.image.load(os.path.join("assets", "phase3.png")),
                "2_0": pygame.image.load(os.path.join("assets", "phase1.png")),
                "2_1": pygame.image.load(os.path.join("assets", "phase2.png")),
                "2_2": pygame.image.load(os.path.join("assets", "phase3.png")),
            }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.window.fill((245, 255, 250))  # Soft background color

        # --- Observations Panel ---
        labels = {
            "temp": f"Temp: {self.state['temp'][0]:.1f}°C",
            "humidity": f"Humidity: {self.state['humidity'][0]:.1f}%",
            "light": f"Light: {self.state['light'][0]:.1f} lx",
            "ec": f"EC: {self.state['ec'][0]:.2f}",
            "ph": f"pH: {self.state['ph'][0]:.2f}",
        }

        y = 10
        for key in labels:
            icon = self.icons[key]
            self.window.blit(icon, (10, y))

            text_surface = self.font.render(labels[key], True, (0, 0, 0))
            text_y = y + (self.icon_size[1] - text_surface.get_height()) // 2
            self.window.blit(text_surface, (10 + self.icon_size[0] + 10, text_y))

            y += self.icon_size[1] + 10

        # --- Plant Visualization ---
        plant_key = f"{self.state['plant_type']}_{self.state['plant_stage']}"
        plant_img = self.plant_drawings.get(plant_key)
        if plant_img:
            center_x = self.window_size[0] // 2 - plant_img.get_width() // 2
            center_y = self.window_size[1] // 2 - plant_img.get_height() // 2
            self.window.blit(plant_img, (center_x, center_y))
        else:
            missing = self.font.render("No image for this plant stage.", True, (255, 0, 0))
            self.window.blit(missing, (self.window_size[0] // 2 - 100, self.window_size[1] // 2))

        # --- Actions & Reward Display ---
        if hasattr(self, 'last_action') and self.last_action is not None:
            formatted_action = []
            for k, v in self.last_action.items():
                if isinstance(v, np.ndarray):
                    formatted = f"{k}: {v[0]:.2f}"
                else:
                    formatted = f"{k}: {v}"
                formatted_action.append(formatted)
            action_str = " | ".join(formatted_action)
        else:
            action_str = "N/A"

        action_text = self.font.render(f"Last Action: {action_str}", True, (0, 100, 0))
        reward_val = getattr(self, 'last_reward', 0)
        reward_text = self.font.render(f"Reward: {reward_val:.2f}", True, (0, 100, 0))

        self.window.blit(action_text, (0, 550))
        self.window.blit(reward_text, (0, 570))

        pygame.display.flip()

