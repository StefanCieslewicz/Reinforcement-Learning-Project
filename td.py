from env import FakeMinecraft
import numpy as np


def e_greedy(Q, next_state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)    
    return int(np.argmax(Q[next_state]))



def sarsa(env, alpha: float = 0.1, epsilon: float = 1.0, n_episodes:int = 100, gamma: float = 0.8):
    Q = env.get_Q()
    rewards_per_episode = []
    
    for episode in range(n_episodes):
        state = env.reset()
        state = state[0]*12 + state[1]
        action = e_greedy(Q, state, epsilon)
        

        total_reward = 0
        terminated = False 
        
        while not terminated:
            
            reward, terminated = env.step(action)
            
            next_state = env.get_location()
            next_state = next_state[0]*12 + next_state[1]
            next_action = e_greedy(Q, next_state, epsilon)
            
            
            
            td_target = reward + gamma * Q[next_state, next_action] * (not terminated)
            td_error = td_target - Q[state, action]

            Q[state, action] += + alpha * td_error

            state = next_state
            action = next_action
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        
        epsilon = np.maximum(0.01, epsilon-0.001)
    
    return Q





def q_learning():
    ...
    
    
    
    
    
if __name__ == '__main__':
    environment = FakeMinecraft()
    
    print(sarsa(environment))