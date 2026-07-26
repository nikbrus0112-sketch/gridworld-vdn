import matplotlib.pyplot as plt
import numpy as np


def rolling_average(values, window=20):
    """Smooths a noisy metric with a simple moving average."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_metrics(losses, episode_rewards, episode_lengths, epsilons, window=20):
    """
    losses: list of loss values, one per training step
    episode_rewards: list of total reward per episode
    episode_lengths: list of steps-to-done per episode
    epsilons: list of epsilon value per episode (or per step, just be consistent)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curve (this one's usually the noisiest, since it's per-step not per-episode)
    axes[0, 0].plot(losses, alpha=0.3, color="tab:blue", label="raw")
    axes[0, 0].plot(
        range(window - 1, len(losses)), rolling_average(losses, window),
        color="tab:blue", label=f"{window}-step avg"
    )
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Loss (MSE)")
    axes[0, 0].legend()

    # Episode reward - the metric you actually care about most
    axes[0, 1].plot(episode_rewards, alpha=0.3, color="tab:green", label="raw")
    axes[0, 1].plot(
        range(window - 1, len(episode_rewards)), rolling_average(episode_rewards, window),
        color="tab:green", label=f"{window}-episode avg"
    )
    axes[0, 1].set_title("Episode Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Total reward")
    axes[0, 1].legend()

    # Episode length - should trend DOWN as the agent finds shorter paths to the goal
    axes[1, 0].plot(episode_lengths, alpha=0.3, color="tab:orange", label="raw")
    axes[1, 0].plot(
        range(window - 1, len(episode_lengths)), rolling_average(episode_lengths, window),
        color="tab:orange", label=f"{window}-episode avg"
    )
    axes[1, 0].set_title("Episode Length (steps to reach goal)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].legend()

    # Epsilon decay - sanity check that your exploration schedule is doing what you think
    axes[1, 1].plot(epsilons, color="tab:red")
    axes[1, 1].set_title("Epsilon (exploration rate)")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # quick sanity check with fake data, so you can confirm the plotting code itself works
    # before wiring it up to your real training loop
    fake_losses = np.abs(np.random.randn(2000)) * np.exp(-np.arange(2000) / 500)
    fake_rewards = np.clip(np.random.randn(300) * 0.3 + np.linspace(0, 1, 300), 0, 1)
    fake_lengths = np.clip(50 - np.linspace(0, 40, 300) + np.random.randn(300) * 5, 5, 50)
    fake_epsilons = np.maximum(0.05, 1.0 * (0.995 ** np.arange(300)))

    plot_training_metrics(fake_losses, fake_rewards, fake_lengths, fake_epsilons)
