from env import FakeMinecraftEnv
import numpy as np
import gymnasium as gym
import pygame

def e_greedy(Q, next_state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)    
    return int(np.argmax(Q[next_state]))



def sarsa(env, alpha: float = 0.1, epsilon: float = 1.0, n_episodes:int = 1, gamma: float = 0.8):
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
                
            if reward == 10:
                print(f"Ep: {episode}; got diamond \nSteps: {step}")
            
            rewards_per_episode.append(total_reward)
            
            self.epsilon = np.maximum(self.final_epsilon, self.epsilon*self.epsilon_decay)
        
        return self.Q


    def q_learning(self, n_episodes:int = 200):
        
        rewards_per_episode = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            state = tuple(state)  

            total_reward = 0
            terminated = False 
            step = 0
            
            while not terminated:
                step += 1
                action = self.e_greedy(state)
                
                next_state, reward, terminated, _, _ = env.step(action)
                next_state = tuple(next_state)            
                
                td_target = reward + self.gamma * np.max(self.Q[next_state]) * (not terminated)
                td_error = td_target - self.Q[state][action]

                self.Q[state][action] += self.alpha * td_error

                state = next_state
                total_reward += reward
            
            if reward == 10:
                print(f"Ep: {episode}; got diamond \nSteps: {step}")
            
            rewards_per_episode.append(total_reward)
            
            self.epsilon = np.maximum(self.final_epsilon, self.epsilon*self.epsilon_decay)
        
        return self.Q
    
    
    def reset_q(self):
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        return


if __name__ == '__main__':
    env = gym.make('FakeMinecraft-v1', render_mode="human")
    pygame.init()
    
    print(sarsa(env))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    env.close()