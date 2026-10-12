import gymnasium as gym
import numpy as np

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False
)

n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))

print("Number of states:", n_states)
print("Number of actions:", n_actions)
print("Q-table shape:", Q.shape)
print()
print(Q)

env.close()