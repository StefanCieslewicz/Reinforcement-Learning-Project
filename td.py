from env import FakeMinecraftEnv
from collections import defaultdict
import numpy as np
import gymnasium as gym
import pygame
import plot_policy


class TemporalDifferenceAgent():
    def __init__(self, env: gym.Env, initial_epsilon:float =0.98, epsilon_decay:float = 0.9, final_epsilon:float = 0.01 , gamma:float =0.95, alpha:float =0.1):
        self.env = env.unwrapped
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.alpha = alpha
        self.gamma = gamma
        
    def e_greedy(self, next_state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)    
        return np.argmax(self.Q[next_state])

    def sarsa(self, n_episodes:int = 200):
        
        rewards_per_episode = []
        ep_with_diamonds = {}
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            state = tuple(state)
            action = self.e_greedy(state)
            
            total_reward = 0
            terminated = False 
            step = 0
            
            while not terminated:
                step += 1
                next_state, reward, terminated, _, _ = self.env.step(action)
                next_state = tuple(next_state)
                next_action = self.e_greedy(next_state)
                
                td_target = reward + self.gamma * self.Q[next_state][next_action] * (not terminated)
                td_error = td_target - self.Q[state][action]

                self.Q[state][action] += self.alpha * td_error

                state = next_state
                action = next_action
                total_reward += reward
                
            if reward == 10:
                ep_with_diamonds[episode] = step
            
            rewards_per_episode.append(total_reward)
            
            self.epsilon = np.maximum(self.final_epsilon, self.epsilon*self.epsilon_decay)
        
        return self.Q, ep_with_diamonds, rewards_per_episode


    def q_learning(self, n_episodes:int = 200):
        
        rewards_per_episode = []
        ep_with_diamonds = {}
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            state = tuple(state)  

            total_reward = 0
            terminated = False 
            step = 0
            
            while not terminated:
                step += 1
                action = self.e_greedy(state)
                
                next_state, reward, terminated, _, _ = self.env.step(action)
                next_state = tuple(next_state)            
                
                td_target = reward + self.gamma * np.max(self.Q[next_state]) * (not terminated)
                td_error = td_target - self.Q[state][action]

                self.Q[state][action] += self.alpha * td_error

                state = next_state
                total_reward += reward
            
            if reward == 10:
                ep_with_diamonds[episode] = step
            
            rewards_per_episode.append(total_reward)
            
            self.epsilon = np.maximum(self.final_epsilon, self.epsilon*self.epsilon_decay)
        
        return self.Q, ep_with_diamonds, rewards_per_episode
    
    
    def reset_q(self):
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        return

if __name__ == '__main__':
    env = gym.make('FakeMinecraft-v1', render_mode="human" ) #  
    pygame.init()
    
    agent = TemporalDifferenceAgent(env)
    num_episodes = 200
    
    #q_,e, _ = agent.sarsa(num_episodes)
    #plot_policy.plot_policy(q_)
    
    
    q_, e, _  = agent.q_learning(num_episodes)
    plot_policy.plot_policy(q_)
    
    #print(e)