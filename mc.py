from env import FakeMinecraftEnv
from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import plot_policy

class MonteCarloAgent:
    def __init__(self, env: gym.Env, initial_epsilon, epsilon_decay, final_epsilon, discount_factor=0.95):
        self.env = env
        self.discount_factor = discount_factor
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns = defaultdict(list)
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
    
    def get_action(self, state):
        state = tuple(state)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[state])
    
    def update(self, episode):
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            state = tuple(state)
            G = self.discount_factor * G + reward
            if (state, action) not in visited:
                self.returns[(state, action)].append(G)
                self.q_values[state][action] = np.mean(self.returns[(state, action)])
                visited.add((state, action))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    env = gym.make("FakeMinecraft-v1")
    agent = MonteCarloAgent(env, initial_epsilon=1.0, epsilon_decay=0.9995, final_epsilon=0.05, discount_factor=0.95)
    num_episodes = 2000

    for episode_idx in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode = []
        optimalish = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, action, reward))

            if reward == 100: 
                print(f"Episode {episode_idx}: reached goal in {len(episode)} steps!")

            state = next_state
            done = terminated or truncated

        agent.update(episode)
        agent.decay_epsilon()

    plot_policy.plot_policy(agent.q_values)