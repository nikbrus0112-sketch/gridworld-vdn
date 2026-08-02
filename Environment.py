import torch
import random


# Environment class - this is the gridworld environment that the agents will interact with
class Environment:
    def __init__(self, grid_size=5, num_agents=1, bonus=1.0):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_targets = num_agents
        self.agents = []
        self.targets = []
        self.bonus = bonus
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

        targets_met = all(target in self.agents for target in self.targets)
        team_bonus = self.bonus if targets_met else 0.0
        done = targets_met

        reward = shaping_rewards[0] + shaping_rewards[1] + team_bonus
        return reward, done

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
