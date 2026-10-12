import gymnasium as gym

env = env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False
)

print("Environment:")
print(env)

print("\nObservation space:")
print(env.observation_space)

print("\nAction space:")
print(env.action_space)

print("\nNumber of states:")
print(env.observation_space.n)

print("\nNumber of actions:")
print(env.action_space.n)


state, info = env.reset()

print("\nInitial state:")
print(state)

print("\nInfo:")
print(info)

action = 2

next_state , reward , terminated , trunctated , info = env.step(action)

print("\nNext state:")
print(next_state)

print("\nReward:")
print(reward)

print("\nTerminated:")
print(terminated)

print("\nTruncated:")
print(trunctated)

print("\nInfo:")
print(info)
    
env.close()