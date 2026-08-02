import torch
import statistics
from VDN import VDN
from IQL import IQL
from Environment import Environment
from plot_training import plot_training_metrics


def test(env, max_steps_per_episode, networks):
    # test
    episode_rewards = []
    episode_lengths = []
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

            reward, done = env.step([action1, action2])
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        # wasted_steps.append(steps - optimal_steps)
        if steps > 8:
            failures += 1
    # for waste in wasted_steps:
    #     if waste > 0:
    #         print(waste)
    # graph
    success = 1000 - failures
    stats = [], episode_rewards, episode_lengths, []
    return stats, success


class Config:
    def __init__(self):
        self.grid_size = 5
        self.agents = 2
        self.layer_size = 64
        self.learning_rate = 7.54e-4
        self.num_episodes = 1500
        self.epsilon = 1.01
        self.epsilon_end = 0.05
        self.decay_rate = 0.99
        self.buffer_size = 2500
        self.warmup_period = 100
        self.batch_size = 64
        self.gamma = 0.781
        self.target_update_frequency = 100
        self.max_steps_per_episode = 75
        self.bonus = 50


config = Config()

env = Environment(config.grid_size, config.agents, config.bonus)

vdn_successes = []
iql_successes = []
env = Environment(config.grid_size, config.agents, config.bonus)

for i in range(20):
    vdn = VDN(config)
    networks, stats = vdn.train(env)
    # plot_training_metrics(stats)
    stats, vdn_success = test(env, config.max_steps_per_episode, networks)
    # plot_training_metrics(stats)
    vdn_successes.append(vdn_success)
    print("vdn test #", i, " | ", vdn_success)

    iql = IQL(config)
    networks, stats = iql.train(env)
    # plot_training.plot_training_metrics(stats)
    stats, iql_success = test(env, config.max_steps_per_episode, networks)
    # plot_training.plot_training_metrics(stats)
    iql_successes.append(iql_success)
    print("iql test #", i, " | ", iql_success)

print(
    "vdn std:   ",
    statistics.stdev(vdn_successes),
    " | iql std:  ",
    statistics.stdev(iql_successes),
)
print(
    "vdn mean: ", sum(vdn_successes) / 20.0, " | iql mean: ", sum(iql_successes) / 20.0
)
