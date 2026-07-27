import torch
from torch.nn import functional as F
import random
from DQN import DQN


class VDN:
    def __init__(self, config):
        self.config = config

    def train(self, environ):
        # ---------------------------------------
        # hyperparameters

        # track
        losses = []
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
        combined_params = list(networks[0][0].parameters()) + list(
            networks[1][0].parameters()
        )
        optimizer = torch.optim.Adam(combined_params, lr=self.config.learning_rate)

        total_steps = 0
        replay_buffer = []
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

                # 3 step environment
                #   take action in environment and get next state, reward, done
                episode_reward += reward
                next_states = []
                next_states.append(env.state(0))
                next_states.append(env.state(1))
                # 4 store in replay buffer

                replay_buffer.append(
                    (
                        states[0],
                        states[1],
                        actions[0],
                        actions[1],
                        reward,
                        next_states[0],
                        next_states[1],
                        done,
                    )
                )
                #   fixed buffer length, if buffer is full, remove oldest sample
                if len(replay_buffer) >= self.config.buffer_size:
                    replay_buffer.pop(0)

                # 5 sample random batch from replay buffer
                #   wait until warmup period is over before sampling
                if len(replay_buffer) > self.config.warmup_period:
                    batch = random.sample(replay_buffer, self.config.batch_size)

                    # unpack and stack everything into batch tensors
                    states_1 = torch.stack([sample[0] for sample in batch])
                    states_2 = torch.stack([sample[1] for sample in batch])
                    actions_1 = torch.tensor([sample[2] for sample in batch]).unsqueeze(
                        1
                    )
                    actions_2 = torch.tensor([sample[3] for sample in batch]).unsqueeze(
                        1
                    )
                    rewards = torch.tensor(
                        [sample[4] for sample in batch], dtype=torch.float32
                    )
                    next_states_1 = torch.stack([sample[5] for sample in batch])
                    next_states_2 = torch.stack([sample[6] for sample in batch])
                    dones = torch.tensor(
                        [sample[7] for sample in batch], dtype=torch.float32
                    )

                    # predicted: each agent's Q-value for the action it actually took
                    Q1_all = networks[0][0].forward(states_1)
                    Q2_all = networks[1][0].forward(states_2)
                    Q1_predicted = Q1_all.gather(1, actions_1).squeeze(1)
                    Q2_predicted = Q2_all.gather(1, actions_2).squeeze(1)
                    joint_predicted = Q1_predicted + Q2_predicted

                    # target: joint reward + discounted sum of each agent's best next-state value
                    with torch.no_grad():
                        max_Q1_next = (
                            networks[0][1].forward(next_states_1).max(dim=1).values
                        )
                        max_Q2_next = (
                            networks[1][1].forward(next_states_2).max(dim=1).values
                        )
                        targets = rewards + self.config.gamma * (
                            max_Q1_next + max_Q2_next
                        ) * (1 - dones)

                    # single joint loss, backprop through both networks at once
                    loss = F.mse_loss(joint_predicted, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(combined_params, max_norm=10)
                    optimizer.step()
                    losses.append(loss.item())

                    # sync target networks
                    if total_steps % self.config.target_update_frequency == 0:
                        for i in range(self.config.agents):
                            networks[i][1].load_state_dict(networks[i][0].state_dict())

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            epsilon = max(
                self.config.epsilon_end, self.config.epsilon * self.config.decay_rate
            )
        epsilons.append(epsilon)
        stats = [losses, episode_rewards, episode_lengths, epsilons]
        return networks, stats
