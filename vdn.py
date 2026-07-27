import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import plot_training


# Environment class - this is the gridworld environment that the agents will interact with
class Environment:
    def __init__(self, grid_size=5, num_agents=1):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_targets = num_agents
        self.agents = []
        self.targets = []
        self.reset()

    # method - reset() - resets the environment to a random state
    def reset(self):
        # resets grid and agents
        self.agents = []
        self.targets = []
        # place agents randomly
        for _ in range(self.num_agents):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            # don't let agents spawn on top of each other
            while (x, y) in self.agents:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
            self.agents.append((x, y))
        # place targets randomly
        for _ in range(self.num_targets):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            # make sure target isn't on top of an agent or another target
            while (x, y) in self.agents or (x, y) in self.targets:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
            self.targets.append((x, y))

    def _nearest_target_distance(self, agent_index):
        x, y = self.agents[agent_index]
        return min(abs(x - tx) + abs(y - ty) for tx, ty in self.targets)

    # method - step(action) - takes an action and returns the new_state, reward, done
    def step(self, actions):
        prev_distances = [
            self._nearest_target_distance(i) for i in range(self.num_agents)
        ]

        # actions can be 0: up, 1: down, 2: left, 3: right, 4: wait
        new_agents = []
        reward = 1
        done = True
        # move agents (watch for out of bounds)
        for i, action in enumerate(actions):
            x, y = self.agents[i]
            if action == 0 and y > 0:  # up
                y -= 1
            elif action == 1 and y < self.grid_size - 1:  # down
                y += 1
            elif action == 2 and x > 0:  # left
                x -= 1
            elif action == 3 and x < self.grid_size - 1:  # right
                x += 1
            # if agent is attempting to step on square of another agent
            if (x, y) in new_agents:
                new_agents.append(self.agents[i])
            else:
                new_agents.append((x, y))

        self.agents = new_agents

        new_distances = [
            self._nearest_target_distance(i) for i in range(self.num_agents)
        ]
        shaping_rewards = [
            prev_distances[i] - new_distances[i] for i in range(self.num_agents)
        ]

        team_bonus = (
            1 if all(target in self.agents for target in self.targets) else 0.0
        )  # change team bonus to +3 or +5
        done = team_bonus == 1

        rewards = [shaping_rewards[i] + team_bonus for i in range(self.num_agents)]
        return rewards, done

    # method - state() - returns the current state of the environment
    # format (xA1, yA1, xA2, yA2, ..., xT1, yT1, xT2, yT2, ...)
    def state(self, agent_index):
        state = []
        # add own position first
        state.extend(
            [
                self.agents[agent_index][0] / (self.grid_size - 1.0),
                self.agents[agent_index][1] / (self.grid_size - 1.0),
            ]
        )
        for i, agent in enumerate(self.agents):
            if i != agent_index:
                state.extend(
                    [
                        agent[0] / (self.grid_size - 1.0),
                        agent[1] / (self.grid_size - 1.0),
                    ]
                )  # normalize to [0, 1]
        for target in self.targets:
            state.extend(
                [target[0] / (self.grid_size - 1.0), target[1] / (self.grid_size - 1.0)]
            )  # normalize to [0, 1]
        return torch.tensor(state, dtype=torch.float32)

    # method - render() - prints the current state of the environment to the console
    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x, y in self.targets:
            grid[y][x] = "T"
        for i, (x, y) in enumerate(self.agents):
            grid[y][x] = f"{i}"
        for row in grid:
            print(" ".join(row))
        print()

    def set(self, agents, targets):
        self.agents = agents
        self.targets = targets


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------
# hyperparameters
grid_size = 5
agents = 2
layer_size = 64
learning_rate = 3e-4  # 4 try 3e-4
num_episodes = 1000
epsilon = 1.0
epsilon_end = 0.05
decay_rate = 0.99
buffer_size = 5000
warmup_period = 100
batch_size = 64  # 2 bring up to 64
gamma = 0.95
target_update_frequency = 100  # 3 experiment 50, 200
max_steps_per_episode = 75  # 5 try 75

# track
losses = [[0 for _ in range(0)] for _ in range(agents)]
episode_rewards = []
episode_lengths = []
epsilons = []

# initialize
env = Environment(grid_size, agents)
networks = []
for i in range(agents):
    q_network = DQN(input_size=4 * agents, hidden_size=layer_size, output_size=5)
    target_network = DQN(input_size=4 * agents, hidden_size=layer_size, output_size=5)
    target_network.load_state_dict(q_network.state_dict())
    networks.append([q_network, target_network])

# train
optimizers = []
for i in range(agents):
    optimizer = torch.optim.Adam(networks[i][0].parameters(), lr=learning_rate)
    optimizers.append(optimizer)

total_steps = 0
replay_buffers = [[0 for _ in range(0)] for _ in range(agents)]
for episode in range(num_episodes):
    env.reset()
    done = False
    episode_reward = 0
    steps = 0
    while not done and steps < max_steps_per_episode:
        steps += 1
        total_steps += 1
        # 1 observe state
        # for every agent
        actions = []
        states = []
        for i in range(agents):
            #   get state from environment
            state = env.state(i)
            states.append(state)
            #   VDN forward pass to get Q values
            Q_values = networks[i][0].forward(state)

            # 2 pick action
            #   select action based on Q values with epsilon-greedy policy
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                action = torch.argmax(Q_values).item()
            actions.append(action)

        # shared reward
        rewards, done = env.step(actions)

        for i in range(agents):
            # 3 step environment
            #   take action in environment and get next state, reward, done
            episode_reward += rewards[i]
            next_state = env.state(i)
            # 4 store in replay buffer

            replay_buffers[i].append(
                (states[i], actions[i], rewards[i], next_state, done)
            )
            #   fixed buffer length, if buffer is full, remove oldest sample
            if len(replay_buffers[i]) >= buffer_size:
                replay_buffers[i].pop(0)

            # 5 sample random batch from replay buffer
            #   wait until warmup period is over before sampling
            if len(replay_buffers[i]) > warmup_period:
                batch = random.sample(replay_buffers[i], batch_size)

                # 6 compute targets
                # sample is (state, action, reward, next_state, done)
                predicted_values = []
                target_values = []
                for sample in batch:
                    predicted_values.append(
                        networks[i][0].forward(sample[0])[sample[1]]
                    )
                    if sample[4]:
                        target_values.append(
                            torch.tensor(sample[2], dtype=torch.float32)
                        )
                    else:
                        with torch.no_grad():
                            target_values.append(
                                sample[2]
                                + gamma * max(networks[i][1].forward(sample[3]))
                            )

                # 7 compute loss, backpropagate, and update weights
                loss = F.mse_loss(
                    torch.stack(predicted_values), torch.stack(target_values)
                )

                optimizers[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(networks[i][0].parameters(), max_norm=10)
                optimizers[i].step()
                losses[i].append(loss.item())

                # 8 Every N steps, update target network
                if total_steps % target_update_frequency == 0:
                    networks[i][1].load_state_dict(networks[i][0].state_dict())

    episode_rewards.append(episode_reward)
    episode_lengths.append(steps)
    epsilon = max(epsilon_end, epsilon * decay_rate)
    epsilons.append(epsilon)

# graph
result = [(x + y) / 2.0 for x, y in zip(losses[0], losses[1])]
plot_training.plot_training_metrics(result, episode_rewards, episode_lengths, epsilons)
# plot_training.plot_training_metrics(
#     losses[1], episode_rewards, episode_lengths, epsilons
# )

# test
episode_rewards = []
episode_lengths = []
wasted_steps = []
failures = 0
for episode in range(1000):
    env.reset()
    # optimal_steps = abs(env.agents[0][0] - env.targets[0][0]) + abs(
    #     env.agents[0][1] - env.targets[0][1]
    # )
    done = False
    episode_reward = 0
    steps = 0

    while not done and steps < max_steps_per_episode:
        steps += 1
        state1 = env.state(0)
        state2 = env.state(1)
        with torch.no_grad():
            Q_values1 = networks[0][0].forward(state1)
            Q_values2 = networks[1][0].forward(state2)
        action1 = torch.argmax(Q_values1).item()
        action2 = torch.argmax(Q_values2).item()

        rewards, done = env.step([action1, action2])
        episode_reward += sum(rewards)

    episode_rewards.append(episode_reward)
    episode_lengths.append(steps)
    # wasted_steps.append(steps - optimal_steps)
    if steps == max_steps_per_episode:
        failures += 1
# for waste in wasted_steps:
#     if waste > 0:
#         print(waste)
# graph
print(failures)
plot_training.plot_training_metrics([], episode_rewards, episode_lengths, [])

# save
torch.save(networks[0][0].state_dict(), "day2_dqn1.pt")
torch.save(networks[1][0].state_dict(), "day2_dqn2.pt")
