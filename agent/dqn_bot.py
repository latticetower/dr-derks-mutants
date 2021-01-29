import numpy as np
from models.network_v1 import Network


NPZ_FILENAME = "weights/model_v2.npz"


class DerkPlayer:
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
            Network(weights, biases) for i in range(n_agents)
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
        _null_action = np.zeros((self.n_agents, 5))

        # random action
        #_random_action = [self.action_space.sample() for i in range(self.n_agents)]
        action_n = [
            self.networks[i].forward(observation_n[i])
            for i in range(self.n_agents)
        ]

        return action_n
