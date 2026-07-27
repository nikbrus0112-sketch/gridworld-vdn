import torch
from torch.nn import functional as F
import random
import plot_training
from DQN import DQN


class IQL:
    def __init__(self, config):
        self.config = config

    def train(self, environ):

        # hyperparameters

        # track
        losses = [[0 for _ in range(0)] for _ in range(self.config.agents)]
        episode_rewards = []
        episode_lengths = []
        epsilons = []

        # initialize
        env = environ
        networks = []
        for i in range(self.config.agents):
            q_network = DQN(
                input_size=4 * self.config.agents,
                hidden_size=self.config.layer_size,
                output_size=5,
            )
            target_network = DQN(
                input_size=4 * self.config.agents,
                hidden_size=self.config.layer_size,
                output_size=5,
            )
            target_network.load_state_dict(q_network.state_dict())
            networks.append([q_network, target_network])

        # train
        optimizers = []
        for i in range(self.config.agents):
            optimizer = torch.optim.Adam(
                networks[i][0].parameters(), lr=self.config.learning_rate
            )
            optimizers.append(optimizer)

        total_steps = 0
        replay_buffers = [[0 for _ in range(0)] for _ in range(self.config.agents)]
        for episode in range(self.config.num_episodes):
            env.reset()
            done = False
            episode_reward = 0
            steps = 0
            while not done and steps < self.config.max_steps_per_episode:
                steps += 1
                total_steps += 1
                # 1 observe state
                # for every agent
                actions = []
                states = []
                for i in range(self.config.agents):
                    #   get state from environment
                    state = env.state(i)
                    states.append(state)
                    #   VDN forward pass to get Q values
                    Q_values = networks[i][0].forward(state)

                    # 2 pick action
                    #   select action based on Q values with epsilon-greedy policy
                    if random.random() < self.config.epsilon:
                        action = random.randint(0, 4)
                    else:
                        action = torch.argmax(Q_values).item()
                    actions.append(action)

                # shared reward
                reward, done = env.step(actions)

                for i in range(self.config.agents):
                    # 3 step environment
                    #   take action in environment and get next state, reward, done
                    episode_reward += reward
                    next_state = env.state(i)
                    # 4 store in replay buffer

                    replay_buffers[i].append(
                        (states[i], actions[i], reward, next_state, done)
                    )
                    #   fixed buffer length, if buffer is full, remove oldest sample
                    if len(replay_buffers[i]) >= self.config.buffer_size:
                        replay_buffers[i].pop(0)

                    # 5 sample random batch from replay buffer
                    #   wait until warmup period is over before sampling
                    if len(replay_buffers[i]) > self.config.warmup_period:
                        batch = random.sample(replay_buffers[i], self.config.batch_size)

                        # unpack and stack everything into batch tensors
                        Bstates = torch.stack([sample[0] for sample in batch])
                        Bactions = torch.tensor(
                            [sample[1] for sample in batch]
                        ).unsqueeze(1)
                        Brewards = torch.tensor(
                            [sample[2] for sample in batch], dtype=torch.float32
                        )
                        Bnext_states = torch.stack([sample[3] for sample in batch])
                        Bdones = torch.tensor(
                            [sample[4] for sample in batch], dtype=torch.float32
                        )

                        # predicted: each agent's Q-value for the action it actually took
                        Q_all = networks[i][0].forward(Bstates)
                        Q_predicted = Q_all.gather(1, Bactions).squeeze(1)

                        # target: joint reward + discounted sum of each agent's best next-state value
                        with torch.no_grad():
                            max_Q_next = (
                                networks[i][1].forward(Bnext_states).max(dim=1).values
                            )
                            targets = Brewards + self.config.gamma * (max_Q_next) * (
                                1 - Bdones
                            )

                        # single joint loss, backprop through both networks at once
                        loss = F.mse_loss(Q_predicted, targets)

                        optimizers[i].zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            networks[i][0].parameters(), max_norm=10
                        )
                        optimizers[i].step()
                        losses[i].append(loss.item())

                        # Every N steps, update target network
                        if total_steps % self.config.target_update_frequency == 0:
                            networks[i][1].load_state_dict(networks[i][0].state_dict())

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            epsilon = max(
                self.config.epsilon_end, self.config.epsilon * self.config.decay_rate
            )
            epsilons.append(epsilon)

        result = [(x + y) / 2.0 for x, y in zip(losses[0], losses[1])]
        stats = [result, episode_rewards, episode_lengths, epsilons]
        return networks, stats
