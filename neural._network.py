import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):

    def __init__(self, n_states, n_actions):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, state):

        return self.network(state)


def train_q_network(
    is_slippery,
    num_episodes=20000,
    alpha=0.001,
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
        torch.manual_seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = QNetwork(
        n_states,
        n_actions
    ).to(device)

    optimizer = optim.Adam(
        Q.parameters(),
        lr=alpha
    )

    loss_function = nn.MSELoss()

    episode_rewards = []
    losses = []

    def one_hot(state):

        state_input = torch.zeros(
            n_states,
            device=device
        )

        state_input[state] = 1

        return state_input

    def choose_action(state):

        if np.random.random() < epsilon:
            return env.action_space.sample()

        state_input = one_hot(state)

        with torch.no_grad():

            q_values = Q(state_input)

        return torch.argmax(q_values).item()

    for episode in range(num_episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        # NEW:
        # Store total loss for this episode
        episode_loss = 0
        episode_steps = 0

        while not terminated and not truncated:

            action = choose_action(state)

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            state_input = one_hot(state)
            next_state_input = one_hot(next_state)

            q_values = Q(state_input)

            current_q = q_values[action]

            with torch.no_grad():

                next_q_values = Q(
                    next_state_input
                )

                best_next_q = torch.max(
                    next_q_values
                )

                # Same logic, but simplified
                if terminated:

                    target = torch.tensor(
                        reward,
                        dtype=torch.float32,
                        device=device
                    )

                else:

                    target = torch.tensor(
                        reward,
                        dtype=torch.float32,
                        device=device
                    ) + gamma * best_next_q

            loss = loss_function(
                current_q,
                target
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # NEW:
            # Accumulate loss for this episode
            episode_loss += loss.item()
            episode_steps += 1

            state = next_state
            total_reward += reward

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        episode_rewards.append(
            total_reward
        )

        # NEW:
        # Store average loss for this episode
        average_episode_loss = (
            episode_loss / episode_steps
        )

        losses.append(
            average_episode_loss
        )

        if (episode + 1) % 1000 == 0:

            recent_rewards = episode_rewards[-1000:]

            success_rate = np.mean(
                recent_rewards
            )

            print(
                f"Episode: {episode + 1:5d} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Success rate: {success_rate:.3f}"
            )

    env.close()

    return Q, episode_rewards, losses


def evaluate(
    Q,
    is_slippery,
    num_episodes=20000,
    seed=None
):

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=is_slippery
    )

    n_states = env.observation_space.n

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

            state_input = torch.zeros(
                n_states,
                device=device
            )

            state_input[state] = 1

            with torch.no_grad():

                q_values = Q(
                    state_input
                )

                action = torch.argmax(
                    q_values
                ).item()

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


def moving_average(values, window):

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nUsing device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    print("\nDETERMINISTIC FROZENLAKE")

    Q_deterministic, deterministic_rewards, deterministic_losses = train_q_network(
        is_slippery=False,
        num_episodes=20000,
        alpha=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=42
    )

    deterministic_success = evaluate(
        Q_deterministic,
        is_slippery=False,
        num_episodes=20000,
        seed=100
    )

    print(
        f"\nDeterministic success rate = "
        f"{deterministic_success:.3f}"
    )


    print("\nSLIPPERY FROZENLAKE")

    Q_slippery, slippery_rewards, slippery_losses = train_q_network(
        is_slippery=True,
        num_episodes=20000,
        alpha=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=42
    )

    slippery_success = evaluate(
        Q_slippery,
        is_slippery=True,
        num_episodes=20000,
        seed=100
    )

    print(
        f"\nSlippery success rate = "
        f"{slippery_success:.3f}"
    )


    # ==========================================================
    # REWARD GRAPH
    # ==========================================================

    window = 500

    deterministic_average = moving_average(
        deterministic_rewards,
        window
    )

    slippery_average = moving_average(
        slippery_rewards,
        window
    )


    plt.figure(figsize=(10, 5))

    plt.plot(
        np.arange(
            window,
            len(deterministic_rewards) + 1
        ),
        deterministic_average,
        linewidth=2,
        label="Deterministic"
    )

    plt.plot(
        np.arange(
            window,
            len(slippery_rewards) + 1
        ),
        slippery_average,
        linewidth=2,
        label="Slippery"
    )

    plt.xlabel("No. of episodes")
    plt.ylabel("Total reward")

    plt.title(
        "Neural Q-Learning Training Curve"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


    # ==========================================================
    # LOSS GRAPH
    # ==========================================================

    loss_window = 500

    deterministic_loss_average = moving_average(
        deterministic_losses,
        loss_window
    )

    slippery_loss_average = moving_average(
        slippery_losses,
        loss_window
    )


    plt.figure(figsize=(10, 5))

    # NEW:
    # X-axis is now episodes
    plt.plot(
        np.arange(
            loss_window,
            len(deterministic_losses) + 1
        ),
        deterministic_loss_average,
        linewidth=2,
        label="Deterministic"
    )

    # NEW:
    # X-axis is now episodes
    plt.plot(
        np.arange(
            loss_window,
            len(slippery_losses) + 1
        ),
        slippery_loss_average,
        linewidth=2,
        label="Slippery"
    )

    plt.xlabel("No. of episodes")
    plt.ylabel("Average TD Loss")

    plt.title(
        "Neural Q-Learning Loss Curve"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


    # ==========================================================
    # FINAL COMPARISON
    # ==========================================================

    print("\nFINAL COMPARISON")

    print(
        f"Deterministic : "
        f"{deterministic_success:.3f}"
    )

    print(
        f"Slippery      : "
        f"{slippery_success:.3f}"
    )