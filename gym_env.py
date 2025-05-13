import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pygame
import math

class HydroponicEnv(gym.Env):
    def __init__(self):
        super(HydroponicEnv, self).__init__()

        # Observation space
        self.observation_space = spaces.Dict({
            "plant_stage": spaces.Discrete(5),   
            "day":  spaces.Discrete(365),
            "watering_cycles": spaces.Discrete(11), 
            "watering_period": spaces.Discrete(21),  
            "temp": spaces.Discrete(51),
            "RH": spaces.Discrete(61),
            "light_intensity": spaces.Discrete(41),
            "light_duration": spaces.Discrete(49),
            "ec": spaces.Discrete(51),
            "ph": spaces.Discrete(51)
        })

        # Action space
        self.action_dims = {
            'RH': 61,
            'ec': 51,
            'light_duration': 49,
            'light_intensity': 41,
            'ph': 51,
            'temp': 51,
            'watering_cycles': 11,
            'watering_period': 21
        }

        # Flattened MultiDiscrete action space
        self.action_space = spaces.MultiDiscrete(list(self.action_dims.values()))

        self.state = None
        self.episode_length = 1000
        self.current_step = 0
        self.max_days = 150
        self.height = 0
        self.biomass = 0
        self.max_biomass = 450  #lesa hn7dedha
        #pygame init 
        pygame.init()
        self.screen = pygame.display.set_mode((1440, 900))
        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0
        self.font_path = os.path.join('assets', 'Lucida_Handwriting_Italic.ttf')
        self.font = pygame.font.Font(self.font_path, 32)
        self.x_space = 310
        self.y_space = 83
        self.mid_position = pygame.Vector2(self.screen.get_width() / 2, self.screen.get_height() / 2)
        self.background = pygame.image.load(os.path.join("assets", "background.png")).convert()
        self.progress_bar_full = pygame.image.load(os.path.join("assets", "progress_bar_full.png")).convert()
        self.plant_stages_assets = [pygame.image.load(os.path.join("assets", "stage-1.png")).convert(),
                             pygame.image.load(os.path.join("assets", "stage-2.png")).convert(),
                             pygame.image.load(os.path.join("assets", "stage-3.png")).convert(),
                             pygame.image.load(os.path.join("assets", "stage-4.png")).convert(),
                             pygame.image.load(os.path.join("assets", "stage-5.png")).convert()]
        self.plant_dead_asset = pygame.image.load(os.path.join("assets", "dead.png")).convert()
        self.GREEN = (44, 149, 65)
        self.PURPLE = (143,125,183)
        # Rectangle properties
        self.rect_width = 38
        self.max_height = 405  
        self.current_height = 0  
        self.fixed_bottom = 670 
        self.rect_x = 1357

    def reset(self, seed=None, options=None):
        self.day=0
        self.Done = False
        self.plant_died = False
        super().reset(seed=seed)
        self.state = {
            "plant_stage": 0,                                  # min=0 (Discrete(5): 0-4
            "day": 0,                                          # min=0 (Discrete(365): 0-364
            "watering_cycles": 0,  # min=0 (Discrete(11): 0-10
            "watering_period": 0,  # min=0 (Discrete(1411): 0-49
            "temp": 0,             # min=0 (Discrete(51): 0-50 → maps to 10°C
            "RH": 0,               # min=0 (Discrete(61): 0-60 → maps to 30% RH
            "light_intensity": 0,  # min=0 (Discrete(41): 0-41 → maps to 0 units
            "light_duration": 0,   # min=0 (Discrete(1441): 0-49
            "ec": 0,               # min=0 (Discrete(51): 0-50 → maps to 0.0
            "ph": 0                # min=0 (Discrete(51): 0-50 → maps to 5.0
        }
        self.current_step = 0
        return self.state, {}

    def retreive_data(self):
        
        self.day += 1

        self.plant_stage = self.calculate_stage()
        
        self.watering_cycles = self.state['watering_cycles']  # Direct mapping (0-10)

        # Watering period: 0-24 → maps to 0-24 hr (discrete 30-minute intervals)
        self.watering_period = 30 * self.state['watering_period']

        # Temperature: 0-50 → maps to 10°C-60°C (0.5°C increments)
        self.temp = 10 + self.state['temp']

        # Relative Humidity: 0-60 → maps to 30%-90% (1% increments)
        self.RH = 30 + self.state['RH']

        # Light Intensity: 0-40 → maps to 0-20000 µmol/m²/s (500 units per step)
        self.light_intensity = self.state['light_intensity'] * 500

        # Light Duration: 0-49 → maps to 0-24 hours (30-minute increments)
        self.light_duration = self.state['light_duration'] * 30  

        # EC (Electrical Conductivity): 0-50 → maps to 0.0-5.0 dS/m (0.1 increments)
        self.ec = self.state['ec'] * 0.1

        # pH: 0-50 → maps to 4.0-9 (0.1 increments)
        self.ph = 4.0 + (self.state['ph'] * 0.1)

    def _get_observation_state(self):
      return {
          'plant_stage': self.plant_stage,
          'day': self.day,
          'watering_cycles': self.watering_cycles,
          'watering_period': self.watering_period // 30,
          'temp': self.temp - 10,
          'RH': self.RH - 30,
          'light_intensity': self.light_intensity // 500,
          'light_duration': self.light_duration // 30,
          'ec': int((self.ec *10 )),
          'ph': int((self.ph - 4.0) * 10)
    }

    def calculate_stage(self):
        if self.day < 10: return 0
        elif 10 <= self.day < 30: return 1
        elif 30 <= self.day < 60: return 2
        elif 60 <= self.day < 90: return 3
        else: return 4

    def _apply_actions(self, action):
      self.state['watering_cycles'] = action['watering_cycles']
      self.state['watering_period'] = action['watering_period']
      self.state['temp'] = action['temp']
      self.state['RH'] = action['RH']
      self.state['light_intensity'] = action['light_intensity']
      self.state['light_duration'] = action['light_duration']
      self.state['ec'] = action['ec']
      self.state['ph'] = action['ph']

    def calc_growth (self) : 
        growth = 0 # initial growth 

        #constants optimal 
        dli_optimal =8
        tempreture_optimal = 25 
        ph_optimal = 6.4
        ec_optimal = 1.4
        rh_optimal = 55
        water_duration_optimal = 1.5 # 1.5hr
        n_cycles_optimal = 5


        # constants sigma 
        dli_sigma = 3
        tempreture_sigma =8
        ph_sigma = 0.7
        ec_sigma = 0.8
        rh_sigma = 15.8
        water_duration_sigma =0.8 
        n_cycles_sigma =  1.5

        #light 
        ppfd = self.state['light_intensity']*0.0185
        DLI = ppfd*self.state['light_duration']*60/1000000

        f_light = self.factor_function(dli_optimal,DLI,dli_sigma)

        # tempreture 
        f_temp = self.factor_function(tempreture_optimal ,self.state['temp'] , tempreture_sigma)
        
        #ph 
        f_ph = self.factor_function(ph_optimal , self.state['ph'], ph_sigma)

        #ec
        f_ec = self.factor_function(ec_optimal , self.state['ec'] , ec_sigma)

        # rh 
        f_rh = self.factor_function(rh_optimal , self.state['RH'] , rh_sigma)

        #water 
        f_water_duration = self.factor_function(water_duration_optimal ,self.state['watering_period'] , water_duration_sigma)
        f_n_cycles = self.factor_function (n_cycles_optimal , self.state['watering_cycles'] , n_cycles_sigma)
        RUE = 3
        growth = RUE * f_light * f_temp * f_ph * f_ec * f_rh * f_water_duration * f_n_cycles  
        print("growth" + str(growth))
        return growth


    def factor_function (self,x_optimal , x , sigma_x): 
        return np.exp(-((x-x_optimal)**2)/(2*(sigma_x)**2))
    
    def damage_loss (self,decay_coff=0,biomass=0): 
        D_t = (self.d_t(self.state['light_intensity'],"light_I")+ self.d_t((self.state['light_duration']/60),"light_D")+ self.d_t(self.state['temp'],"temp")+
                  self.d_t(self.state['RH'],"humidity")
                +self.d_t(self.state['ph'],"ph")+ self.d_t(self.state['watering_period']*self.state['watering_cycles'],"TWD"))

        # print(type(D_t))
        print("D_t" + str(D_t))
        return D_t

    def d_t(self,condition,condition_name:str):
        conditions_factors={
        # light indensity
        "light_I":{
        "light_I_optimal": 10000,
        "light_I_low" : [3000, -500,3] ,
        "light_I_high" : [65000,100000,5.2]},

        # light duration
        "light_D":{
        "light_D_optimal": 10,
        "light_D_low" : [5,-1,6.4] ,
        "light_D_high" : [14,30,2.3]},

        # temp
        "temp":{
        "temp_optimal": 25,
        "temp_low" : [7,-1,3.3] ,
        "temp_high" : [40,54,3.3]},
        
        # humidity
        "humidity":{
        "humidity_optimal": 55,
        "humidity_low" : [200,400,1] ,
        "humidity_high" : [200,400,1]},

        # ph
        "ph":{
        "ph_optimal": 6.4,
        "ph_low" : [4.4,1.8,2.0] ,
        "ph_high" : [8.2,10,2.2]},

        # ec
        "ec":{
        "ec_optimal": 1.4,
        "ec_low" : [0,0,2] ,
        "ec_high" : [12,12,2]},

        # total water duration (number of water cycels * hours per one cycle )
        "TWD":{
        "TWD_optimal": 7.5,
        "TWD_low" : [5,-5,2.0] ,
        "TWD_high" : [12,30,2]}

        }
        flag = 0
        if condition < conditions_factors[condition_name][condition_name+"_optimal"] and condition < conditions_factors[condition_name][condition_name+"_low"][0] : 
            x_critical = conditions_factors[condition_name][condition_name+"_low"][0]
            x_max = conditions_factors[condition_name][condition_name+"_low"][1]
            gamma = conditions_factors[condition_name][condition_name+"_low"][2]
            flag = 1
            # print(condition_name, {x_critical})
        elif condition > conditions_factors[condition_name][condition_name+"_optimal"] and condition > conditions_factors[condition_name][condition_name+"_high"][0]:
            x_critical=conditions_factors[condition_name][condition_name+"_high"][0]
            x_max = conditions_factors[condition_name][condition_name+"_high"][1]
            gamma = conditions_factors[condition_name][condition_name+"_high"][2]
            flag = 1

            
        if flag== 1: 
            f_x = min(1,max(0,((condition - x_critical) / (x_max - x_critical)) ** gamma))
        else : 
            f_x =0

        return f_x 

    def calculate_reward(self):
        #El model Hykoon hena 
        # calculate_growth_rate()
        # calculate_penality()
        # calculate_height()
        # reward logic
        if (self.damage_loss()>= 1 ): 
            self.plant_died = True
            return -10.0
        if (self.damage_loss() != None ):
            return (self.calc_growth() - ( 0.03 * self.damage_loss() * self.calc_growth()))
        else:
            return self.calc_growth()



    def step(self, action):
        self.current_step += 1

        action_dict = dict(zip(self.action_dims.keys(), action))

        self._apply_actions(action_dict)
        self.retreive_data()
        self.reward = self.calculate_reward()
        self.state = self._get_observation_state()

        #implement it inside the calculate reward function
        if self.plant_died:
            self.Done = True
            self.reward = -10
            self.reset()


        terminated = self.current_step >= self.episode_length
        truncated = False
        self.last_action = action
        self.last_reward = self.reward
        # Apply action to the state (later we can define how it affects plant growth)
        # For now, we skip state transitions to keep it simple

        #pygame loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
        #Background and plant
        self.screen.blit(self.background, (0,0))
        self.screen.blit(self.plant_stages_assets[self.plant_stage], self.mid_position-pygame.Vector2(100,200))
        if self.plant_died:
              self.screen.blit(self.plant_dead_asset, self.mid_position-pygame.Vector2(100,200))

        #Progress bar
        self.current_height = min((self.biomass/self.max_biomass)*self.max_height, self.max_height)
        self.rect_y = self.fixed_bottom - self.current_height
        self.growing_rect = pygame.Rect(self.rect_x, self.rect_y, self.rect_width, self.current_height)
        pygame.draw.rect(self.screen, self.PURPLE, self.growing_rect)
        pygame.draw.line(self.screen, self.PURPLE, (self.rect_x, self.fixed_bottom), 
                        (self.rect_x + self.rect_width, self.fixed_bottom), 3)
        if self.current_height == self.max_height:
            self.screen.blit(self.progress_bar_full, (1356,250))


        #Environment
        # Convert MultiDiscrete array -> dict
        action_dict = dict(zip(self.action_dims.keys(), action))
        #1 Temp
        self.temp_ts = self.font.render(f"{self.temp} C", True, self.GREEN)
        self.temp_tr = self.temp_ts.get_rect()
        self.temp_tr.center = (pygame.Vector2(250,60))
        self.screen.blit(self.temp_ts, self.temp_tr)

        #2 RH
        self.rh_ts = self.font.render(f"{self.RH}%", True, self.GREEN)
        self.rh_tr = self.rh_ts.get_rect()
        self.rh_tr.center = (pygame.Vector2(250,60+self.y_space))
        self.screen.blit(self.rh_ts, self.rh_tr)

        #3 PH
        self.ph_ts = self.font.render(f"{self.ph}", True, self.GREEN)
        self.ph_tr = self.ph_ts.get_rect()
        self.ph_tr.center = (pygame.Vector2(250+self.x_space,60))
        self.screen.blit(self.ph_ts, self.ph_tr)

        #4 EC
        self.ec_ts = self.font.render(f"{self.ec}", True, self.GREEN)
        self.ec_tr = self.ec_ts.get_rect()
        self.ec_tr.center = (pygame.Vector2(250+self.x_space,60+self.y_space))
        self.screen.blit(self.ec_ts, self.ec_tr)

        #5 Light Intensity
        self.light_intensity_ts = self.font.render(f"{self.light_intensity} LUX", True, self.GREEN)
        self.light_intensity_tr = self.light_intensity_ts.get_rect()
        self.light_intensity_tr.center = (pygame.Vector2(250+2*self.x_space,60))
        self.screen.blit(self.light_intensity_ts, self.light_intensity_tr)

        #6 Light duration
        self.light_duration_ts = self.font.render(f"{self.light_duration} mins", True, self.GREEN)
        self.light_duration_tr = self.light_duration_ts.get_rect()
        self.light_duration_tr.center = (pygame.Vector2(250+2*self.x_space,60+self.y_space))
        self.screen.blit(self.light_duration_ts, self.light_duration_tr)

        #7 Water duration/period
        self.water_duration_ts = self.font.render(f"{self.watering_period} mins", True, self.GREEN)
        self.water_duration_tr = self.water_duration_ts.get_rect()
        self.water_duration_tr.center = (pygame.Vector2(250+3*self.x_space,60))
        self.screen.blit(self.water_duration_ts, self.water_duration_tr)

        #8 Num of water periods
        self.water_periods_ts = self.font.render(f"{self.watering_cycles}", True, self.GREEN)
        self.water_periods_tr = self.water_periods_ts.get_rect()
        self.water_periods_tr.center = (pygame.Vector2(250+3*self.x_space,60+self.y_space))
        self.screen.blit(self.water_periods_ts, self.water_periods_tr)

        #9 Day
        self.day_ts = self.font.render(f"{self.day}", True, self.GREEN)
        self.day_tr = self.water_periods_ts.get_rect()
        self.day_tr.center = (pygame.Vector2(725,221))
        self.screen.blit(self.day_ts, self.day_tr)

        #Actions
        #1 Temp
        self.action_temp_ts = self.font.render(f"{action_dict['temp']} C", True, self.GREEN)   # +10 
        self.action_temp_tr = self.action_temp_ts.get_rect()
        self.action_temp_tr.center = (pygame.Vector2(250,765))
        self.screen.blit(self.action_temp_ts, self.action_temp_tr)

        #2 RH
        self.action_rh_ts = self.font.render(f"{action_dict['RH']}%", True, self.GREEN)  # +30 
        self.action_rh_tr = self.action_rh_ts.get_rect()
        self.action_rh_tr.center = (pygame.Vector2(250,765+self.y_space))
        self.screen.blit(self.action_rh_ts, self.action_rh_tr)

        #3 PH
        self.action_ph_ts = self.font.render(f"{4.0 + (action_dict['ph'] )}", True, self.GREEN) # *0.1
        self.action_ph_tr = self.action_ph_ts.get_rect()
        self.action_ph_tr.center = (pygame.Vector2(250+self.x_space,765))
        self.screen.blit(self.action_ph_ts, self.action_ph_tr)

        #4 EC
        self.action_ec_ts = self.font.render(f"{action_dict['ec'] }", True, self.GREEN)  # *0.1
        self.action_ec_tr = self.action_ec_ts.get_rect()
        self.action_ec_tr.center = (pygame.Vector2(250+self.x_space,765+self.y_space))
        self.screen.blit(self.action_ec_ts, self.action_ec_tr)

        #5 Light Intensity
        self.action_light_intensity_ts = self.font.render(f"{action_dict['light_intensity'] }", True, self.GREEN) #*500
        self.action_light_intensity_tr = self.action_light_intensity_ts.get_rect()
        self.action_light_intensity_tr.center = (pygame.Vector2(250+2*self.x_space,765))
        self.screen.blit(self.action_light_intensity_ts, self.action_light_intensity_tr)

        #6 Light duration
        self.action_light_duration_ts = self.font.render(f"{action_dict['light_duration'] }", True, self.GREEN) # *30
        self.action_light_duration_tr = self.action_light_duration_ts.get_rect()
        self.action_light_duration_tr.center = (pygame.Vector2(250+2*self.x_space,765+self.y_space))
        self.screen.blit(self.action_light_duration_ts, self.action_light_duration_tr)

        #7 Water duration/period
        self.action_water_duration_ts = self.font.render(f"{action_dict['watering_period'] }", True, self.GREEN)  # *30
        self.action_water_duration_tr = self.action_water_duration_ts.get_rect()
        self.action_water_duration_tr.center = (pygame.Vector2(250+3*self.x_space,765))
        self.screen.blit(self.action_water_duration_ts, self.action_water_duration_tr)

        #8 Num of water periods
        self.action_water_periods_ts = self.font.render(f"{action_dict['watering_cycles']}", True, self.GREEN)
        self.action_water_periods_tr = self.action_water_periods_ts.get_rect()
        self.action_water_periods_tr.center = (pygame.Vector2(250+3*self.x_space,765+self.y_space))
        self.screen.blit(self.action_water_periods_ts, self.action_water_periods_tr)

        #Equations
        #1 Biomass
        self.biomass_ts = self.font.render(f"{self.biomass} KG", True, self.GREEN)
        self.biomass_tr = self.biomass_ts.get_rect()
        self.biomass_tr.center = (pygame.Vector2(192,370))
        self.screen.blit(self.biomass_ts, self.biomass_tr)

        #2 Height
        self.height_ts = self.font.render(f"{self.height} CM", True, self.GREEN)
        self.height_tr = self.height_ts.get_rect()
        self.height_tr.center = (pygame.Vector2(192,516))
        self.screen.blit(self.height_ts, self.height_tr)

        pygame.display.flip()

        self.dt = self.clock.tick(60) / 1000
        return self.state, self.reward, terminated, truncated, {}