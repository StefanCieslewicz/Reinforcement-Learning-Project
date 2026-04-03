import numpy as np
import gymnasium as gym
from env import FakeMinecraftEnv

class DPAlgorithm:
    """
    It encapsulates DP algorithms. Using a class helps us share the environment 
    knowledge (the transition model P) between Policy and Value Iteration.
    """
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env, self.gamma, self.theta = env.unwrapped, gamma, theta
        self.n_s, self.n_a = self.env.size**2, self.env.action_space.n
        self.P = self._build_model()

    def _build_model(self):
        # it builds the MDP model by simulating steps from every state
        P = {s: {a: [] for a in range(self.n_a)} for s in range(self.n_s)}
        orig = self.env._agent_location.copy()
        
        for s in range(self.n_s):
            pos = np.array([s // self.env.size, s % self.env.size])
            # it handles terminal states (lava/diamond) as absorbing (Slide 18)
            if any(np.array_equal(pos, l) for l in self.env.lava) or np.array_equal(pos, self.env._target_location):
                for a in range(self.n_a): P[s][a] = [(1.0, s, 0.0, True)]
                continue
            for a in range(self.n_a):
                self.env._agent_location, (ns_pos, r, term, _, _) = pos.copy(), self.env.step(a)
                ns = ns_pos[0] * self.env.size + ns_pos[1]
                P[s][a] = [(1.0, ns, r, term)]
        self.env._agent_location = orig
        return P

    def _get_v(self, s, a, V):
        # it calculates the expected value of an action using the Bellman equation
        return sum(prob * (r + self.gamma * V[ns] * (not term)) for prob, ns, r, term in self.P[s][a])

    def policy_iteration(self):
        V, policy, stable = np.zeros(self.n_s), np.ones((self.n_s, self.n_a)) / self.n_a, False
        while not stable:
            while True: # Evaluation
                delta = 0
                for s in range(self.n_s):
                    v, V[s] = V[s], sum(policy[s,a] * self._get_v(s, a, V) for a in range(self.n_a))
                    delta = max(delta, abs(v - V[s]))
                if delta < self.theta: break
            stable = True # Improvement
            for s in range(self.n_s):
                old_a, vals = np.argmax(policy[s]), [self._get_v(s, a, V) for a in range(self.n_a)]
                best_a = old_a if vals[old_a] >= max(vals) - 1e-8 else np.argmax(vals)
                policy[s] = np.eye(self.n_a)[best_a]
                if old_a != best_a: stable = False
        return V, policy

    def value_iteration(self):
        # it updates V using max action value directly
        V = np.zeros(self.n_s)
        while True:
            delta = 0
            for s in range(self.n_s):
                v, V[s] = V[s], max(self._get_v(s, a, V) for a in range(self.n_a))
                delta = max(delta, abs(v - V[s]))
            if delta < self.theta: break
        # it extracts the policy
        policy = np.zeros((self.n_s, self.n_a))
        for s in range(self.n_s):
            policy[s, np.argmax([self._get_v(s, a, V) for a in range(self.n_a)])] = 1.0
        return V, policy

def policy_iteration(env, gamma=0.9, theta=1e-6): return DPAlgorithm(env, gamma, theta).policy_iteration()
def value_iteration(env, gamma=0.9, theta=1e-6): return DPAlgorithm(env, gamma, theta).value_iteration()

if __name__ == '__main__':
    env = gym.make('FakeMinecraft-v1')
    V, _ = value_iteration(env); print(f"Value Iteration complete. V shape: {V.shape}")
    V, _ = policy_iteration(env); print(f"Policy Iteration complete. V shape: {V.shape}")
    env.close()
