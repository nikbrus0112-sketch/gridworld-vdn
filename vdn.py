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

    # method - reset() - resets the environment to a random state and returns the initial state
    def reset(self):
        # resets grid and agents
        self.agents = []
        self.targets = []
        # place agents randomly
        for _ in range(self.num_agents):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.agents.append((x, y))
        # place targets randomly
        for _ in range(self.num_targets):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            # make sure target isn't on top of an agent
            while (x, y) in self.agents:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
            self.targets.append((x, y))

    # method - step(action) - takes an action and returns the new_state, reward, done
    def step(self, actions):
        # move agents (watch for out of bounds)
        # actions can be 0: up, 1: down, 2: left, 3: right
        new_agents = []
        reward = 0
        done = False
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
            new_agents.append((x, y))
            # if an agent is on it's target then reward = 1 done = True
            if (x, y) in self.targets:
                reward += 1
                done = True
        self.agents = new_agents
        return self.state(), reward, done

    # method - state() - returns the current state of the environment
    # format (xA1, yA1, xA2, yA2, ..., xT1, yT1, xT2, yT2, ...)
    def state(self):
        state = []
        for agent in self.agents:
            state.extend(
                [agent[0] / (self.grid_size - 1.0), agent[1] / (self.grid_size - 1.0)]
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
        for x, y in self.agents:
            grid[y][x] = "A"
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
agents = 1
layer_size = 64
learning_rate = 1e-3
num_episodes = 150
epsilon = 1.0
epsilon_end = 0.05
decay_rate = 0.983
buffer_size = 10000
warmup_period = 100
batch_size = 32
gamma = 0.9
target_update_frequency = 100
max_steps_per_episode = 50

# track
losses = []
episode_rewards = []
episode_lengths = []
epsilons = []

# initialize
env = Environment(grid_size, agents)
q_network = DQN(input_size=4 * agents, hidden_size=layer_size, output_size=4)
target_network = DQN(input_size=4 * agents, hidden_size=layer_size, output_size=4)
target_network.load_state_dict(q_network.state_dict())

# train
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
total_steps = 0
replay_buffer = []
for episode in range(num_episodes):
    env.reset()
    done = False
    episode_reward = 0
    steps = 0
    while not done and steps < max_steps_per_episode:
        steps += 1
        total_steps += 1
        # 1 observe state
        #   get state from environment
        state = env.state()
        #   VDN forward pass to get Q values
        Q_values = q_network.forward(state)

        # 2 pick action
        #   select action based on Q values with epsilon-greedy policy
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = torch.argmax(Q_values).item()

        # 3 step environment
        #   take action in environment and get next state, reward, done
        next_state, reward, done = env.step([action])
        episode_reward += reward
        # 4 store in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        #   fixed buffer length, if buffer is full, remove oldest sample
        if len(replay_buffer) >= buffer_size:
            replay_buffer.pop(0)

        # 5 sample random batch from replay buffer
        #   wait until warmup period is over before sampling
        if len(replay_buffer) > warmup_period:
            batch = random.sample(replay_buffer, batch_size)

            # 6 compute targets
            # sample is (state, action, reward, next_state, done)
            predicted_values = []
            target_values = []
            for sample in batch:
                predicted_values.append(q_network.forward(sample[0])[sample[1]])
                if sample[4]:
                    target_values.append(torch.tensor(sample[2], dtype=torch.float32))
                else:
                    with torch.no_grad():
                        target_values.append(
                            sample[2] + gamma * max(target_network.forward(sample[3]))
                        )

            # 7 compute loss, backpropagate, and update weights
            loss = F.mse_loss(torch.stack(predicted_values), torch.stack(target_values))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_
            optimizer.step()
            losses.append(loss.item())

            # 8 Every N steps, update target network
            if total_steps % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    episode_rewards.append(episode_reward)
    episode_lengths.append(steps)
    epsilon = max(epsilon_end, epsilon * decay_rate)
    epsilons.append(epsilon)

# graph
plot_training.plot_training_metrics(losses, episode_rewards, episode_lengths, epsilons)

# test
episode_rewards = []
episode_lengths = []
wasted_steps = []

for episode in range(1000):
    env.reset()
    optimal_steps = abs(env.agents[0][0] - env.targets[0][0]) + abs(
        env.agents[0][1] - env.targets[0][1]
    )
    done = False
    episode_reward = 0
    steps = 0

    while not done and steps < max_steps_per_episode:
        steps += 1
        state = env.state()

        with torch.no_grad():
            Q_values = q_network.forward(state)
        action = torch.argmax(Q_values).item()

        next_state, reward, done = env.step([action])
        episode_reward += reward

    episode_rewards.append(episode_reward)
    episode_lengths.append(steps)
    wasted_steps.append(steps - optimal_steps)

for waste in wasted_steps:
    if waste > 0:
        print(waste)
# graph
plot_training.plot_training_metrics([], episode_rewards, episode_lengths, [])

# save
# torch.save(q_network.state_dict(), "day1_dqn.pt")
