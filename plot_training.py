import matplotlib.pyplot as plt
import numpy as np


def rolling_average(values, window=100):
    """Smooths a noisy metric with a simple moving average."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def rolling_percentile(values, window, percentile):
    """Computes a rolling percentile (0-100) over a sliding window."""
    values = np.asarray(values)
    if len(values) < window:
        return np.array([])
    return np.array(
        [
            np.percentile(values[i : i + window], percentile)
            for i in range(len(values) - window + 1)
        ]
    )


def plot_training_metrics(
    losses, episode_rewards, episode_lengths, epsilons, window=100
):
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
        range(window - 1, len(losses)),
        rolling_average(losses, window),
        color="tab:blue",
        label=f"{window}-step avg",
    )
    x_pct = range(window - 1, len(losses))
    axes[0, 0].plot(
        x_pct,
        rolling_percentile(losses, window, 50),
        color="black",
        linewidth=1.5,
        label=f"{window}-step median",
    )
    axes[0, 0].plot(
        x_pct,
        rolling_percentile(losses, window, 90),
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label=f"{window}-step p90",
    )
    axes[0, 0].plot(
        x_pct,
        rolling_percentile(losses, window, 10),
        color="tab:green",
        linestyle="--",
        linewidth=1,
        label=f"{window}-step p10",
    )
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Loss (MSE, log scale)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)

    # Episode reward - the metric you actually care about most
    axes[0, 1].plot(episode_rewards, alpha=0.3, color="tab:green", label="raw")
    axes[0, 1].plot(
        range(window - 1, len(episode_rewards)),
        rolling_average(episode_rewards, window),
        color="tab:green",
        label=f"{window}-episode avg",
    )
    axes[0, 1].set_title("Episode Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Total reward")
    axes[0, 1].legend()

    # Episode length - should trend DOWN as the agent finds shorter paths to the goal
    axes[1, 0].plot(episode_lengths, alpha=0.3, color="tab:orange", label="raw")
    axes[1, 0].plot(
        range(window - 1, len(episode_lengths)),
        rolling_average(episode_lengths, window),
        color="tab:orange",
        label=f"{window}-episode avg",
    )
    axes[1, 0].set_title("Episode Length (steps to reach goal)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].legend()

    # Relative loss dispersion — (p90 - p10) / p50, a scale-free noisiness measure.
    # Unlike a raw p90-p10 gap, dividing by the median stays meaningful even as loss
    # shrinks by orders of magnitude over training (a gap of 0.01 means very different
    # things when the median loss is 0.5 vs. when it's 0.001).
    p10 = rolling_percentile(losses, window, 10)
    p50 = rolling_percentile(losses, window, 50)
    p90 = rolling_percentile(losses, window, 90)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_spread = np.where(p50 > 0, (p90 - p10) / p50, np.nan)
    axes[1, 1].plot(x_pct, relative_spread, color="tab:purple")
    axes[1, 1].set_title("Loss Volatility: (p90 - p10) / p50")
    axes[1, 1].set_xlabel("Training step")
    axes[1, 1].set_ylabel("Relative spread")
    axes[1, 1].yaxis.tick_right()
    axes[1, 1].yaxis.set_label_position("right")

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # quick sanity check with fake data, so you can confirm the plotting code itself works
    # before wiring it up to your real training loop
    fake_losses = np.abs(np.random.randn(2000)) * np.exp(-np.arange(2000) / 500)
    fake_rewards = np.clip(np.random.randn(300) * 0.3 + np.linspace(0, 1, 300), 0, 1)
    fake_lengths = np.clip(
        50 - np.linspace(0, 40, 300) + np.random.randn(300) * 5, 5, 50
    )
    fake_epsilons = np.maximum(0.05, 1.0 * (0.995 ** np.arange(300)))

    plot_training_metrics(fake_losses, fake_rewards, fake_lengths, fake_epsilons)
