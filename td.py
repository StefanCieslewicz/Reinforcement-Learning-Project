from env import FakeMinecraftEnv
import numpy as np
import gymnasium as gym
import pygame

def e_greedy(Q, next_state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)    
    return int(np.argmax(Q[next_state]))



def sarsa(
            env, 
            alpha: float = 0.1, 
            epsilon: float = 1.0,
            gamma: float = 0.8, 
            n_episodes:int = 1 
        ):
    
    
    Q = np.zeros((12*12, 4))
    rewards_per_episode = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        state = state[0]*12 + state[1]
        action = e_greedy(Q, state, epsilon)
        

        total_reward = 0
        terminated = False 
        
        while not terminated:
            
            next_state, reward, terminated, _, _ = env.step(action)
            
            next_state = next_state[0]*12 + next_state[1]
            next_action = e_greedy(Q, next_state, epsilon)
            
            
            
            td_target = reward + gamma * Q[next_state, next_action] * (not terminated)
            td_error = td_target - Q[state, action]

            Q[state, action] += alpha * td_error

            state = next_state
            action = next_action
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        
        epsilon = np.maximum(0.01, epsilon-0.001)
    
    return Q, rewards_per_episode


def q_learning(
                env, 
                alpha: float = 0.1, 
                epsilon: float = 1.0,
                gamma: float = 0.8, 
                n_episodes:int = 100 
            ):
    
    Q = np.zeros((12*12, 4))
    rewards_per_episode = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        state = state[0]*12 + state[1]        

        total_reward = 0
        terminated = False 
        
        while not terminated:
            
            action = e_greedy(Q, state, epsilon)
            
            next_state, reward, terminated, _, _ = env.step(action)
            next_state = next_state[0]*12 + next_state[1]            
            
            td_target = reward + gamma * np.max(Q[next_state]) * (not terminated)
            td_error = td_target - Q[state, action]

            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        
        epsilon = np.maximum(0.01, epsilon-0.001)
    
    
    
    return Q, rewards_per_episode

if __name__ == '__main__':
    env = gym.make('FakeMinecraft-v1') #  render_mode="human"
    pygame.init()
    
    print(q_learning(env))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    env.close()