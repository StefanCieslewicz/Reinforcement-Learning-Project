import numpy as np
import gymnasium as gym
from env import FakeMinecraftEnv

class DPAlgorithm:
    """
    It encapsulates the Dynamic Programming algorithms (Policy Iteration and Value Iteration).
    I used a class here to share the transition model and parameters between both algorithms,
    which makes the code cleaner and more object-oriented.
    """
    # it initializes the solver with the environment and hyperparameters
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env.unwrapped
        self.gamma = gamma
        self.theta = theta
        self.size = self.env.size
        self.n_states = self.size * self.size
        self.n_actions = self.env.action_space.n
        
        # it builds the transition model right away so we can use it for both algorithms
        self.P = self._build_transition_model()

    def _state_to_idx(self, pos):
        # it converts a 2D position array into a single integer state index (0 to 143)
        return pos[0] * self.size + pos[1]

    def _idx_to_state(self, idx):
        # it converts a single integer state index back to a 2D position array [row, col]
        return np.array([idx // self.size, idx % self.size])

    def _is_terminal(self, pos):
        # it checks if the current position is a terminal state (lava or diamond)
        # by looking at the environment's actual variables instead of hardcoding
        is_lava = any(np.array_equal(pos, lava) for lava in self.env.lava)
        is_target = np.array_equal(pos, self.env._target_location)
        return is_lava or is_target

    def _build_transition_model(self):
        """
        It creates a dictionary to store transitions: P[state][action] = [(prob, next_state, reward, terminated)]
        Instead of hardcoding rewards and walls, it uses the environment's step() function to
        discover the rules. This ensures it's perfectly synced with env.py.
        """
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        
        # it saves the original agent location so we don't mess up the environment during simulation
        original_location = self.env._agent_location.copy()
        
        for state in range(self.n_states):
            pos = self._idx_to_state(state)
            
            # it handles terminal states by making them absorbing states with 0 reward (Slide 18)
            if self._is_terminal(pos):
                for a in range(self.n_actions):
                    P[state][a] = [(1.0, state, 0.0, True)]
                continue
            
            # it simulates taking each action from the current state
            for a in range(self.n_actions):
                # it sets the environment to the specific state
                self.env._agent_location = pos.copy()
                
                # it takes a step in the environment to see what happens (extracts reward dynamically)
                obs, reward, terminated, truncated, info = self.env.step(a)
                next_state = self._state_to_idx(obs)
                
                # since the environment is deterministic, probability is always 1.0
                P[state][a] = [(1.0, next_state, reward, terminated)]
                
        # it restores the original agent location to keep the environment untouched
        self.env._agent_location = original_location
        return P

    def policy_evaluation(self, policy):
        """
        It evaluates a given policy (from Slide 20).
        """
        # it starts with an array of zeros for the value function
        V = np.zeros(self.n_states)
        
        while True:
            delta = 0
            for s in range(self.n_states):
                v_old = V[s]
                new_v = 0
                
                # it calculates the expected value using the Bellman Expectation Equation
                for a, action_prob in enumerate(policy[s]):
                    if action_prob > 0:
                        for prob, next_state, reward, terminated in self.P[s][a]:
                            # if it's terminated, the future value is 0 (Slide 18)
                            new_v += action_prob * prob * (reward + self.gamma * V[next_state] * (not terminated))
                
                V[s] = new_v
                delta = max(delta, abs(v_old - V[s]))
                
            # it stops when the changes are smaller than our threshold
            if delta < self.theta:
                break
                
        return V

    def policy_iteration(self):
        """
        Policy Iteration algorithm (from Slide 38).
        Alternates between evaluating a policy and improving it.
        """
        # it initializes a uniform random policy (25% chance for each action)
        policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        policy_stable = False
        iteration = 0
        
        while not policy_stable:
            iteration += 1
            print(f"  [PI] Iteration {iteration}: Evaluating policy...")
            
            # it evaluates the current policy
            V = self.policy_evaluation(policy)
            
            policy_stable = True
            changes = 0
            
            # it improves the policy by acting greedily
            for s in range(self.n_states):
                old_action = np.argmax(policy[s])
                
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for prob, next_state, reward, terminated in self.P[s][a]:
                        action_values[a] += prob * (reward + self.gamma * V[next_state] * (not terminated))
                
                # it uses a small tolerance to break ties and prevent infinite loops between equally good actions
                best_val = np.max(action_values)
                if action_values[old_action] >= best_val - 1e-8:
                    best_action = old_action
                else:
                    best_action = np.argmax(action_values)
                
                # it creates the new greedy policy for this state
                new_policy_s = np.zeros(self.n_actions)
                new_policy_s[best_action] = 1.0
                policy[s] = new_policy_s
                
                if old_action != best_action:
                    policy_stable = False
                    changes += 1
                    
            print(f"  [PI] Iteration {iteration}: Policy improved. {changes} states changed action.")
            
            # failsafe limit to avoid infinite loops
            if iteration > 50:
                print("  [PI] Warning: Reached 50 iterations, breaking to avoid infinite loop.")
                break
                
        return V, policy

    def value_iteration(self):
        """
        Value Iteration algorithm (from Slide 48).
        """
        # it initializes the value function to zeros
        V = np.zeros(self.n_states)
        
        while True:
            delta = 0
            for s in range(self.n_states):
                v_old = V[s]
                
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for prob, next_state, reward, terminated in self.P[s][a]:
                        action_values[a] += prob * (reward + self.gamma * V[next_state] * (not terminated))
                
                # it applies the Bellman Optimality Equation (takes the max immediately)
                V[s] = np.max(action_values)
                delta = max(delta, abs(v_old - V[s]))
                
            # it stops when the values converge
            if delta < self.theta:
                break
                
        # it extracts the final optimal policy from the optimal value function
        policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_state, reward, terminated in self.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * V[next_state] * (not terminated))
            
            best_action = np.argmax(action_values)
            policy[s][best_action] = 1.0
            
        return V, policy

# wrapper functions to maintain compatibility with the rest of the project (main.ipynb)
def policy_iteration(env, gamma=0.9, theta=1e-6):
    solver = DPAlgorithm(env, gamma, theta)
    return solver.policy_iteration()

def value_iteration(env, gamma=0.9, theta=1e-6):
    solver = DPAlgorithm(env, gamma, theta)
    return solver.value_iteration()

if __name__ == '__main__':
    # it tests the code to ensure it runs correctly
    env = gym.make('FakeMinecraft-v1')
    
    print("Running Value Iteration...")
    V_vi, policy_vi = value_iteration(env, gamma=0.9)
    print("Value Iteration complete. Optimal Value Function shape:", V_vi.shape)
    
    print("Running Policy Iteration...")
    V_pi, policy_pi = policy_iteration(env, gamma=0.9)
    print("Policy Iteration complete. Optimal Value Function shape:", V_pi.shape)
    
    # it verifies that both methods yield roughly the same values
    diff = np.max(np.abs(V_vi - V_pi))
    print(f"Max difference between VI and PI value functions: {diff:.6f}")
    
    env.close()
