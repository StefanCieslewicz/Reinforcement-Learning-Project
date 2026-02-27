import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Env():
    def __init__(self):
        super(Env, self).__init__()
        
        self.action_space = spaces.Discrete(...)
        self.observation_space = ...
        self.state = None
        
    def reset(self):
        self.state = ...
        return self.state, {}
    
    def action1(self):
        ...
        
    def __str__(self):
        return f'{self.state}, ...'

