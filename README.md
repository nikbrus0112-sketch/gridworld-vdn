# gridworld-vdn

Cooperative multi-agent RL in a gridworld: DQN, Independent Q-Learning, and Value Decomposition Networks (VDN), comparing coordination under a shared team reward.

**Status:** Day 1 complete (single-agent DQN foundation). Day 2 (two-agent IQL vs. VDN comparison) in progress.

## Overview

This project builds up to a comparison between Independent Q-Learning (IQL) and Value Decomposition Networks (VDN) in a cooperative multi-agent gridworld, where two agents only receive reward if both reach a goal simultaneously. Before adding multi-agent complexity, Day 1 establishes a working single-agent DQN as the foundation both approaches will be built on.

## Environment

A 5x5 gridworld with no obstacles. An agent occupies one cell; a goal is placed randomly on reset. Four actions are available (up, down, left, right); moving into a wall is a no-op. The agent receives a reward of `+1` upon reaching the goal (terminating the episode) and `0` otherwise. State is represented as normalized `(x, y)` coordinates for both agent and goal, in `[0, 1]`.

## Day 1: Single-agent DQN

Standard DQN: a small MLP predicts Q-values for all four actions in one forward pass, trained via a replay buffer, a periodically-synced target network, and epsilon-greedy exploration decaying over the course of training.

**Network:** 2 hidden layers, 64 units each, ReLU activations, no normalization or residual connections — a plain feedforward MLP was appropriate here, not a simplification; the state space has no sequence structure, variable length, or need for positional information that would justify attention or recurrence.

### A hyperparameter finding worth noting

The standard `gamma = 0.99` default, common across most RL tutorials and papers, was actively harmful for this environment. With discounting this close to 1, the target-value gap between an optimal path and a path with one redundant step is roughly 1% (e.g., `0.99` vs. `0.9801`) — well within the network's normal approximation error. The result: the agent frequently took redundant steps near the goal, and in some cases settled into short back-and-forth cycles between two states with near-identical Q-values, since nothing in training meaningfully penalized the extra step.

Lowering `gamma` to `0.9` widened that same gap to roughly 9% (`0.9` vs. `0.81`), giving the network enough signal to reliably distinguish path lengths. This fully resolved both the redundant-step behavior and the cycling failures observed at `gamma = 0.99`.

### Evaluation methodology

Evaluation uses a fully greedy policy (epsilon forced to zero, no training updates) across fresh, randomly-generated start/goal configurations — separate from the noisier training curves, which reflect exploration in addition to policy quality. For each test episode, actual steps taken are compared against the optimal step count (Manhattan distance between start and goal, which equals shortest-path length on an obstacle-free 4-directional grid).

**Results:** 100% success rate across 1000 fresh evaluation episodes, with the agent consistently matching the optimal step count. (A single failure was observed in an earlier, smaller evaluation batch and did not reproduce across a subsequent 1000-episode run — attributed to a rare Q-value tie rather than a systematic issue.)

### Network size vs. training data tradeoff

A secondary experiment varied hidden layer size against the minimum episode count needed for consistent convergence (defined as <5 failures per 1000 evaluation episodes):

| Hidden units | Min. episodes for consistent convergence |
|---|---|
| 64 | ~150 |
| 32 | ~200 |
| 16 | ~250 |

Smaller networks were also noticeably less *reliable* at a given episode count, not just slower to converge on average — at 16 units, a 250-episode run would sometimes converge cleanly and sometimes train poorly, an inconsistency not observed at 64 units even at lower episode counts. This is consistent with smaller networks having less redundant capacity to absorb an unlucky weight initialization. The final Day 1 model uses 64 hidden units for training reliability, ahead of building the more complex two-agent version on top of it.

## Repo structure

```
gridworld-vdn/
├── environment.py       # gridworld env
├── train.py              # DQN training loop
├── plot_training.py       # training/eval metric plotting
├── day1_dqn.pt            # trained single-agent checkpoint
└── README.md
```

## Coming in Day 2

Extending to a two-agent cooperative gridworld (two agents, two goals, team-only reward requiring both agents to reach a goal simultaneously), comparing:
- **Independent Q-Learning (IQL):** each agent learns its own Q-function independently, with no shared training signal.
- **Value Decomposition Networks (VDN):** per-agent Q-values are summed into a joint value trained on the shared team reward, allowing credit assignment to propagate back to each agent individually.

The comparison is directly inspired by factored multi-agent RL approaches, particularly Prof. Chongjie Zhang's work on cooperative MARL with factorization structures.


## Hyperparameters Day 2 Multi Agent IQL
grid_size = 5
agents = 2
layer_size = 64
learning_rate = 3e-4
num_episodes = 1000
epsilon = 1.0
epsilon_end = 0.05
decay_rate = 0.99
buffer_size = 5000
warmup_period = 100
batch_size = 32
gamma = 0.90
target_update_frequency = 100
max_steps_per_episode = 75

Achieved 188 / 1000 successes in testing or 18.8% success rate