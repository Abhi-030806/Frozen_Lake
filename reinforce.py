"""
Vanilla Policy Gradient (REINFORCE) on Gymnasium FrozenLake-v1.

This is intentionally a small, readable implementation:
- State input: one-hot vector of length 16 (same representation as your DQN).
- Policy network: 16 -> 64 -> 4 logits.
- Actions are sampled from a Categorical distribution.
- Train on batches of complete episodes using reward-to-go.
- No critic/value network: this is vanilla REINFORCE, not actor-critic.

Run inside your existing ML virtual environment:
    python reinforce_frozenlake.py
"""

import random
import csv
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# -------------------------
# 1. Reproducibility/device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -------------------------
# 2. Hyperparameters
# -------------------------
ENV_NAME = "FrozenLake-v1"
IS_SLIPPERY = True

STATE_SIZE = 16
ACTION_SIZE = 4
HIDDEN_SIZE = 64

GAMMA = 0.99
LEARNING_RATE = 1e-2
EPISODES_PER_UPDATE = 32
NUM_UPDATES = 1500

EVAL_EVERY = 10
EVAL_EPISODES = 1000

OUTPUT_DIR = Path("reinforce_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------
# 3. Policy network
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, ACTION_SIZE),
        )

    def forward(self, state_one_hot):
        # Return logits, not probabilities.
        return self.network(state_one_hot)


def encode_state(state):
    """Convert integer state 0..15 to a 16-element one-hot vector."""
    return torch.nn.functional.one_hot(
        torch.tensor(int(state)), num_classes=STATE_SIZE
    ).float().to(DEVICE)


# -------------------------
# 4. Sample one episode
# -------------------------
def collect_episode(env, policy):
    states = []
    actions = []
    rewards = []

    state, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        state_vector = encode_state(state)

        # The network produces one logit per action.
        logits = policy(state_vector)

        # Softmax is applied internally by Categorical(logits=...).
        action_distribution = Categorical(logits=logits)

        # Sample an action, rather than always taking argmax.
        action = action_distribution.sample()

        states.append(state_vector)
        actions.append(action)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(float(reward))
        state = next_state

    return states, actions, rewards


# -------------------------
# 5. Calculate reward-to-go
# -------------------------
def calculate_returns(rewards, gamma):
    """
    G_t = r_(t+1) + gamma*r_(t+2) + gamma^2*r_(t+3) + ...

    Returns one discounted return for each action in the episode.
    """
    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return returns


# -------------------------
# 6. Update policy (REINFORCE)
# -------------------------
def update_policy(policy, optimizer, batch):
    all_log_probs = []
    all_returns = []

    for states, actions, rewards in batch:
        episode_returns = calculate_returns(rewards, GAMMA)

        for state_vector, action, return_t in zip(states, actions, episode_returns):
            logits = policy(state_vector)
            distribution = Categorical(logits=logits)

            # Log-probability of the action actually sampled.
            all_log_probs.append(distribution.log_prob(action))
            all_returns.append(return_t)

    returns_tensor = torch.tensor(
        all_returns, dtype=torch.float32, device=DEVICE
    )

    # Normalize across the whole batch, not separately within each episode.
    # This can reduce gradient variance. If the batch has no variation,
    # leave the returns unchanged.
    if returns_tensor.numel() > 1 and returns_tensor.std(unbiased=False) > 1e-8:
        returns_tensor = (
            returns_tensor - returns_tensor.mean()
        ) / (returns_tensor.std(unbiased=False) + 1e-8)

    log_probs_tensor = torch.stack(all_log_probs)

    # Minimize negative weighted log-probability == maximize weighted log-probability.
    loss = -(log_probs_tensor * returns_tensor).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


# -------------------------
# 7. Evaluate a trained policy
# -------------------------
def evaluate_policy(policy, episodes=300, greedy=False, seed=10_000):
    """
    greedy=False: sample actions from the learned stochastic policy.
    greedy=True: choose the highest-probability action (argmax).
    """
    env = gym.make(ENV_NAME, is_slippery=IS_SLIPPERY)
    successes = 0

    policy.eval()
    with torch.no_grad():
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            terminated = False
            truncated = False

            while not (terminated or truncated):
                state_vector = encode_state(state)
                logits = policy(state_vector)

                if greedy:
                    action = int(torch.argmax(logits).item())
                else:
                    action = int(Categorical(logits=logits).sample().item())

                state, reward, terminated, truncated, _ = env.step(action)

            if reward == 1.0:
                successes += 1

    env.close()
    policy.train()
    return successes / episodes


# -------------------------
# 8. Main training loop
# -------------------------
def main():
    env = gym.make(ENV_NAME, is_slippery=IS_SLIPPERY)

    policy = PolicyNetwork().to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    training_successes = []
    eval_updates = []
    eval_stochastic = []
    eval_greedy = []
    update_losses = []

    print(f"Environment: {ENV_NAME}, is_slippery={IS_SLIPPERY}")
    print(f"Updates: {NUM_UPDATES}; episodes/update: {EPISODES_PER_UPDATE}")
    print(f"Total training episodes: {NUM_UPDATES * EPISODES_PER_UPDATE}")
    print("-" * 72)

    for update in range(1, NUM_UPDATES + 1):
        batch = []
        batch_successes = 0

        for _ in range(EPISODES_PER_UPDATE):
            episode = collect_episode(env, policy)
            batch.append(episode)

            if sum(episode[2]) > 0:
                batch_successes += 1

        loss = update_policy(policy, optimizer, batch)
        update_losses.append(loss)
        training_successes.append(batch_successes / EPISODES_PER_UPDATE)

        if update == 1 or update % EVAL_EVERY == 0:
            stochastic_rate = evaluate_policy(
                policy, episodes=EVAL_EPISODES, greedy=False
            )
            greedy_rate = evaluate_policy(
                policy, episodes=EVAL_EPISODES, greedy=True
            )

            eval_updates.append(update)
            eval_stochastic.append(stochastic_rate)
            eval_greedy.append(greedy_rate)

            print(
                f"Update {update:4d}/{NUM_UPDATES} | "
                f"batch success={batch_successes / EPISODES_PER_UPDATE:.3f} | "
                f"eval sampled={stochastic_rate:.3f} | "
                f"eval greedy={greedy_rate:.3f} | "
                f"loss={loss:.4f}"
            )

    env.close()

    # Save evaluation history at full floating-point precision.
    csv_path = OUTPUT_DIR / "reinforce_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["update", "stochastic_success_rate", "greedy_success_rate"])
        writer.writerows(zip(eval_updates, eval_stochastic, eval_greedy))

    # Save per-update training data (no rounding or smoothing).
    training_csv_path = OUTPUT_DIR / "reinforce_training.csv"
    with training_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["update", "loss", "batch_success_rate"])
        writer.writerows(
            (i, loss_value, success_rate)
            for i, (loss_value, success_rate) in enumerate(
                zip(update_losses, training_successes), start=1
            )
        )

    # Plot measured evaluation points. Markers show each actual evaluation;
    # no moving average, interpolation, or smoothing is applied.
    plt.figure(figsize=(10, 6))
    plt.plot(
        eval_updates, eval_stochastic,
        marker="o", markersize=3, linewidth=1,
        label="Sampled policy"
    )
    plt.plot(
        eval_updates, eval_greedy,
        marker="s", markersize=3, linewidth=1,
        label="Greedy (argmax) policy"
    )
    plt.xlabel("Policy updates")
    plt.ylabel(f"Success rate over {EVAL_EPISODES} episodes")
    plt.title(f"REINFORCE on FrozenLake (slippery={IS_SLIPPERY})")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "reinforce_learning_curve.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Plot raw loss after every policy update (not smoothed).
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(update_losses) + 1),
        update_losses,
        linewidth=1,
    )
    plt.xlabel("Policy updates")
    plt.ylabel("REINFORCE loss")
    plt.title("REINFORCE Training Loss on FrozenLake")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = OUTPUT_DIR / "reinforce_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()

    model_path = OUTPUT_DIR / "reinforce_policy.pt"
    torch.save(policy.state_dict(), model_path)

    print("\nFinished.")
    print(f"Final sampled-policy success rate: {eval_stochastic[-1]:.6f}")
    print(f"Final greedy-policy success rate:  {eval_greedy[-1]:.6f}")
    print(f"Saved evaluation CSV: {csv_path}")
    print(f"Saved per-update training CSV: {training_csv_path}")
    print(f"Saved success-rate graph: {plot_path}")
    print(f"Saved loss graph: {loss_plot_path}")
    print(f"Saved policy weights: {model_path}")


if __name__ == "__main__":
    main()
