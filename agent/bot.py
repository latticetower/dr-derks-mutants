import random
from itertools import product

import numpy as np
import torch

from models.network_v1 import Network

NPZ_FILENAME = "weights/model_v1.npz"


class DerkPlayer(object):
    """
    Player which controls all agents
    Each arena has 3 agents which the player must control.
    """

    def __init__(self, n_agents, action_space):
        """
        Parameters:
         - n_agents: TOTAL number of agents being controlled (= #arenas * #agents per arena)
        """

        self.n_agents = n_agents
        self.action_space = action_space

        with np.load(NPZ_FILENAME) as data:
            weights = np.asarray(data['weights']).copy()
            biases = np.asarray(data['biases']).copy()

        self.networks = [
            Network(weights, biases) for _ in range(n_agents)
        ]

    def signal_env_reset(self, obs):
        """
        env.reset() was called
        """

    def take_action(self, env_step_ret):
        """
        Parameters:
         - env_step_ret: whatever env.step() returned (obs_n, rew_n, done_n, info_n)

        Returns: action for each agent for each arena

        Actions:
         - MoveX: A number between -1 and 1
         - Rotate: A number between -1 and 1
         - ChaseFocus: A number between 0 and 1
         - CastingSlot:
                        0 = don’t cast
                    1 - 3 = cast corresponding ability
         - ChangeFocus:
                        0 = keep current focus
                        1 = focus home statue
                    2 - 3 = focus teammates
                        4 = focus enemy statue
                    5 - 7 = focus enemy
        """
        observation_n, _rew_n, _done_n, _info_n = env_step_ret

        # null action
        # _null_action = np.zeros((self.n_agents, 5))

        # random action
        # _random_action = [self.action_space.sample() for i in range(self.n_agents)]

        action_n = [
            self.networks[i].forward(observation_n[i])
            for i in range(self.n_agents)
        ]

        # print(observation_n[0].shape)
        # print(len(action_n[0]))

        return action_n


class DerkAgent(object):
    """
    Player which controls all agents
    Each arena has 3 agents which the player must control.
    """

    def __init__(self, n_agents, estimator, epsilon=0.0, device: torch.device = 'cpu'):
        """
        Parameters:
         - n_agents: TOTAL number of agents being controlled (= #arenas * #agents per arena)
        """

        self.n_agents = n_agents
        self.estimator = estimator
        self.epsilon = epsilon
        self.device = device

        self.movements = np.linspace(-1.0, 1.0, 4)
        self.rotates = np.linspace(-1.0, 1.0, 4)
        self.chase_focuses = np.linspace(0.0, 1.0, 4)
        self.casting_slots = np.arange(4)
        self.change_focuses = np.arange(8)

        self.action_space = self.get_action_space()

    def get_action_space(self) -> torch.tensor:
        return torch.tensor([action for action in
                             product(self.movements, self.rotates, self.chase_focuses, self.casting_slots,
                                     self.change_focuses)],
                            dtype=torch.float32)

    def signal_env_reset(self, obs):
        """
        env.reset() was called
        """

    def take_action(self, env_step) -> np.array:
        """
        Parameters:
         - env_step_ret: whatever env.step() returned (obs_n, rew_n, done_n, info_n)

        Returns: action for each agent for each arena

        Actions:
         - MoveX: A number between -1 and 1
         - Rotate: A number between -1 and 1
         - ChaseFocus: A number between 0 and 1
         - CastingSlot:
                        0 = don’t cast
                    1 - 3 = cast corresponding ability
         - ChangeFocus:
                        0 = keep current focus
                        1 = focus home statue
                    2 - 3 = focus teammates
                        4 = focus enemy statue
                    5 - 7 = focus enemy
        """
        self.estimator.eval()

        bots_actions = list()

        if random.random() < self.epsilon:
            print('RANDOM')

            for _ in range(self.n_agents):
                move = np.random.choice(self.movements)
                rotate = np.random.choice(self.rotates)
                chase_focus = np.random.choice(self.chase_focuses)
                casting_slot = np.random.choice(self.casting_slots)
                change_focus = np.random.choice(self.change_focuses)

                bots_actions.append(np.array([move, rotate, chase_focus, casting_slot, change_focus]))

        else:
            print('ESTIMATED')

            agents_observations = env_step[0]

            for i in range(self.n_agents):
                agent_observation = torch.tensor([agents_observations[i] for _ in range(len(self.action_space))],
                                                 dtype=torch.float32)

                single_step = torch.cat([agent_observation, self.action_space], dim=1)

                values = self.estimator(single_step).detach().cpu().reshape(-1)

                max_value_index = torch.argmax(values)

                bots_actions.append(self.action_space[max_value_index].numpy())

        return np.array(bots_actions)

    def update_estimator(self, estimator):
        self.estimator = estimator

    def update_epsilon(self, epsilon: float):
        self.epsilon = epsilon
