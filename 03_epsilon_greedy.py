import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

ACTION_SYMBOLS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

LAKE = [
    ["S", "F", "F", "F"],
    ["F", "H", "F", "H"],
    ["F", "F", "F", "H"],
    ["H", "F", "F", "G"]
]


def train_q_table(
    is_slippery,
    num_episodes=50000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
    seed=None
):

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=is_slippery
    )

    if seed is not None:
        env.reset(seed=seed)
        np.random.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    print("\nNumber of states :", n_states)
    print("Number of actions:", n_actions)

    Q = np.zeros((n_states, n_actions))

    episode_rewards = []

    def choose_action(state):

        if np.random.random() < epsilon:
            return env.action_space.sample()

        max_value = np.max(Q[state])

        best_actions = np.flatnonzero(
            Q[state] == max_value
        )

        return np.random.choice(best_actions)

    for episode in range(num_episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not terminated and not truncated:

            action = choose_action(state)

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            current_q = Q[state, action]

            if terminated:
                target = reward
            else:
                best_next_q = np.max(Q[next_state])
                target = reward + gamma * best_next_q

            td_error = target - current_q

            Q[state, action] = (
                current_q + alpha * td_error
            )

            state = next_state
            total_reward += reward

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        episode_rewards.append(total_reward)

        if (episode + 1) % 1000 == 0:

            recent_rewards = episode_rewards[-1000:]

            success_rate = np.mean(recent_rewards)

            print(
                f"Episode: {episode + 1:5d} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Success rate: {success_rate:.3f}"
            )

    env.close()

    return Q, episode_rewards


def display_q_table(Q):

    print("\nQ-TABLE")

    for state in range(len(Q)):

        print(
            f"State {state:2d}: "
            f"L={Q[state,0]:.4f} | "
            f"D={Q[state,1]:.4f} | "
            f"R={Q[state,2]:.4f} | "
            f"U={Q[state,3]:.4f}"
        )


def display_policy(Q):

    policy = np.argmax(Q, axis=1)

    print("\nLearned Policy")

    for row in range(4):

        for col in range(4):

            state = row * 4 + col
            tile = LAKE[row][col]

            if tile == "H":
                print(" H ", end=" ")

            elif tile == "G":
                print(" G ", end=" ")

            else:
                action = policy[state]

                print(
                    f" {ACTION_SYMBOLS[action]} ",
                    end=" "
                )

        print()


def evaluate(
    Q,
    is_slippery,
    num_episodes=10000,
    seed=None
):

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=is_slippery
    )

    successes = 0

    for episode in range(num_episodes):

        if seed is not None:

            state, info = env.reset(
                seed=seed + episode
            )

        else:

            state, info = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:

            max_value = np.max(Q[state])

            best_actions = np.flatnonzero(
                Q[state] == max_value
            )

            action = np.random.choice(best_actions)

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            state = next_state

            if reward == 1:
                successes += 1

    env.close()

    return successes / num_episodes


if __name__ == "__main__":

    print("\n")
    print("        DETERMINISTIC FROZENLAKE")

    Q_deterministic, deterministic_rewards = train_q_table(
        is_slippery=False,
        num_episodes=50000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=42
    )

    deterministic_success = evaluate(
        Q_deterministic,
        is_slippery=False,
        num_episodes=10000,
        seed=100
    )

    print("\nDeterministic evaluation:")

    print(
        f"Success rate = "
        f"{deterministic_success:.3f}"
    )

    print("\n   SLIPPERY FROZENLAKE")

    Q_slippery, slippery_rewards = train_q_table(
        is_slippery=True,
        num_episodes=50000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=42
    )

    slippery_success = evaluate(
        Q_slippery,
        is_slippery=True,
        num_episodes=10000,
        seed=100
    )

    print("\nSlippery evaluation:")

    print(
        f"Success rate = "
        f"{slippery_success:.3f}"
    )

    episodes = np.arange(
        1,
        len(deterministic_rewards) + 1
    )

    window = 500

    deterministic_average = np.convolve(
        deterministic_rewards,
        np.ones(window) / window,
        mode="valid"
    )

    slippery_average = np.convolve(
        slippery_rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        deterministic_rewards,
        alpha=0.15,
        label="Deterministic"
    )

    plt.plot(
        episodes,
        slippery_rewards,
        alpha=0.15,
        label="Slippery"
    )

    plt.plot(
        episodes[window - 1:],
        deterministic_average,
        linewidth=2,
        label="Deterministic average"
    )

    plt.plot(
        episodes[window - 1:],
        slippery_average,
        linewidth=2,
        label="Slippery average"
    )

    plt.xlabel("No. of episodes")
    plt.ylabel("Total reward")
    plt.title("Tabular Q-Learning Training Curve")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    print("\nFinal Q-table - Deterministic:")

    display_q_table(
        Q_deterministic
    )

    print("\nPolicy - Deterministic:")

    display_policy(
        Q_deterministic
    )

    print("\nFinal Q-table - Slippery:")

    display_q_table(
        Q_slippery
    )

    print("\nPolicy - Slippery:")

    display_policy(
        Q_slippery
    )

    print("\nFINAL COMPARISON")

    print(
        f"Deterministic : "
        f"{deterministic_success:.3f}"
    )

    print(
        f"Slippery      : "
        f"{slippery_success:.3f}"
    )