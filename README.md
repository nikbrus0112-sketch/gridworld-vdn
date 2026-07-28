# gridworld-vdn

Cooperative multi-agent RL in a gridworld: DQN, Independent Q-Learning (IQL), and Value Decomposition Networks (VDN), comparing coordination under a shared team reward.

**Status:** Day 1 and Day 2 complete. IQL vs. VDN comparison shows a statistically significant, mechanistically-explained VDN advantage.

## Overview

This project compares Independent Q-Learning (IQL) and Value Decomposition Networks (VDN) in a cooperative multi-agent gridworld, where two agents only receive a completion reward if both reach a goal simultaneously. Day 1 establishes a working single-agent DQN as the foundation; Day 2 extends this to two agents and directly compares how IQL and VDN handle credit assignment under an identical, shared reward signal.

## Environment

A 5x5 gridworld with no obstacles.

**Day 1 (single agent):** one agent, one goal, four actions (up/down/left/right), `+1` reward on reaching the goal, `0` otherwise.

**Day 2 (two agents):** two agents, two goals, five actions (up/down/left/right/wait). Agents cannot occupy the same cell. Reward is a single shared scalar per step: each agent's distance-based shaping reward summed across both agents, plus a team completion bonus awarded only when every goal is simultaneously occupied by a distinct agent. Both IQL and VDN train on this identical joint signal

## Day 1: Single-agent DQN

Standard DQN: a small MLP predicts Q-values for all four actions in one forward pass, trained via a replay buffer, a periodically-synced target network, and epsilon-greedy exploration decaying over training.

**Network:** 2 hidden layers, 64 units each, ReLU activations, no normalization or residual connections — a plain feedforward MLP was appropriate here, not a simplification; the state space has no sequence structure, variable length, or need for positional information that would justify attention or recurrence.

### Hyperparameter finding: gamma sensitivity

The standard `gamma = 0.99` default was actively harmful for this environment. With discounting this close to 1, the target-value gap between an optimal path and a path with one redundant step is roughly 1% — well within the network's normal approximation error, causing redundant steps and occasional Q-value-tie cycling. Lowering `gamma` to `0.9` widened that gap to roughly 9%, giving the network enough signal to reliably distinguish path lengths and resolving both issues.

### Results

100% success rate across 1000 fresh evaluation episodes (greedy policy, epsilon forced to 0), consistently matching the optimal (Manhattan-distance) step count.

### Network size vs. training data tradeoff

| Hidden units | Min. episodes for consistent convergence |
| ------------ | ---------------------------------------- |
| 64           | ~150                                     |
| 32           | ~200                                     |
| 16           | ~250                                     |

Smaller networks were also less *reliable* at a given episode count, not just slower to converge — consistent with less redundant capacity to absorb an unlucky initialization. Final model uses 64 hidden units.

## Day 2: IQL vs. VDN

### Implementation

Both algorithms share the same environment, network architecture, and hyperparameters, differing only in training mechanism:

- **IQL:** each agent has its own Q-network, its own replay buffer, and its own optimizer. Each trains independently to predict the same shared joint reward, with no mechanism to separate its own contribution from its teammate's.
- **VDN:** each agent still has its own Q-network, but transitions are stored in one shared replay buffer, and each agent's predicted Q-value for its taken action is summed into a single joint prediction (`Q1 + Q2`), trained against a single joint target (`reward + gamma * (max Q1' + max Q2')`) via one backward pass spanning both networks. This lets credit assignment to each agent emerge from the shared reward signal, without either agent seeing the other's individual contribution directly.

### Statistical methodology

Single training runs proved unreliable indicators of an algorithm's true performance: run-to-run variance in final success rate (different random seed, same hyperparameters) was substantially larger than test-set sampling noise alone. Each reported result below is the mean and standard deviation of **20 independently trained networks per condition**, evaluated on 1000 fresh greedy-policy test episodes each, compared via a paired t-test (paired on matched train/test iteration).

### Hyperparameter finding: team bonus scale and deferred reward

With the wait action added (needed for an agent to hold position while its teammate is still traveling), the team completion bonus arrives only after a variable, sometimes-long delay. Under discounting, a small bonus (`+1`) is heavily discounted by the time it reaches the value estimate driving the wait decision, weakening the incentive to learn to wait relative to the immediate, undiscounted shaping reward already collected during approach. This produced a "hit or miss" pattern where success depended heavily on whether a given training run's random exploration happened to reinforce waiting behavior early on.

Raising the team bonus resolved this cleanly:

| Team bonus | VDN mean (± sd)   | IQL mean (± sd)   | VDN − IQL | p-value |
| ---------- | ----------------- | ----------------- | --------- | ------- |
| 1          | 49.5% (± 11.9pp)  | 57.7% (± 13.3pp)  | −8.2pp    | 0.12 (not significant) |
| 3          | 83.6% (± 8.8pp)   | 74.6% (± 12.0pp)  | +9.1pp    | 0.0037  |
| 5          | 91.7% (± 5.3pp)   | 82.0% (± 9.6pp)   | +9.7pp    | 0.0020  |

(All at `gamma = 0.9`, 20 runs per condition per algorithm.)

Raising the bonus both improved mean performance and *tightened* run-to-run variance for both algorithms — consistent with the coordination behavior becoming something training reliably discovers rather than sometimes discovers. Notably, IQL's variance stayed consistently higher than VDN's across every condition tested, independent evidence supporting the underlying mechanism: independent agents training against a shared, non-decomposed reward converge less reliably than agents whose network structure performs credit assignment for them.

### Final result

At `gamma = 0.9`, `team_bonus = 5` (the final configuration): **VDN outperforms IQL by ~9.7 percentage points on average (91.7% vs. 82.0% success), a statistically significant and consistently reproduced gap (p = 0.002, n = 20 runs per algorithm), with VDN also converging more reliably run-to-run** (5.3pp vs. 9.6pp standard deviation).

## Repo structure

```
gridworld-vdn/
├── Environment.py     # two-agent gridworld env
├── DQN.py              # shared MLP architecture
├── IQL.py               # Independent Q-Learning training loop
├── vdn.py                # VDN training loop
├── testing.py             # greedy-policy evaluation + multi-run statistical comparison
├── plot_training.py        # training/eval metric plotting (loss percentiles, volatility, etc.)
├── test.ipynb                # exploratory notebook
├── Graphs/                    # saved training/eval figures
├── Results/                    # saved run statistics
└── README.md
```

## Stretch goals / future work

Ideas discussed during this project but scoped out to stay within the 1-2 day budget:

- **Full gamma × team-bonus grid sweep** (3 gammas × 5 bonus values × 20 runs × 2 algorithms ≈ 600 training runs) to map the full hyperparameter surface, rather than the single-dimension sweep done here. Estimated 10-30 hours of compute; worth doing with parallelized training runs if revisited.
- **QMIX comparison** — a more expressive factorization than VDN (nonlinear rather than a simple sum), as a third point of comparison alongside IQL and VDN.
- **Sparse reward (no shaping) comparison** — test whether the IQL/VDN gap widens further when credit assignment is harder (no dense per-step signal), which the current shaped-reward setup partially masks.
- **Fixed agent-to-goal assignment** — currently either agent may claim either goal (avoiding a combinatorial assignment problem); requiring agent 1 → goal 1 specifically would test whether IQL/VDN handle *implicit coordination on who goes where*, not just *when*, differently.
- **Larger grid (e.g. 8x8)** to test whether VDN's advantage widens with longer coordination horizons — confounded by simultaneously increasing state-space size and difficulty for both algorithms, so a cleaner version of this test would be:
- **Biased far-apart spawn positions on the existing 5x5 grid** — a cheaper way to stress-test long coordination/wait horizons without changing the state space size.
- **Randomized agent-processing order** — current collision resolution always gives agent 0 movement priority in a contested step; alternating or randomizing per-step would remove this minor asymmetry.