import numpy as np
import torch

from models.qnet import Qnet
from utils.common import random_actions

WEIGHTS_FILE = "weights/model_dqn_v1.pth"


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
        self.network = Qnet()
        self.use_gpu = torch.cuda.is_available()
        self.network.load_state_dict(torch.load(WEIGHTS_FILE))
        if self.use_gpu:
            self.network.cuda()
        self.network.eval()

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
        n_actions = 50
        observations, _rew_n, _done_n, _info_n = env_step_ret
        # sample actions
        observations = torch.from_numpy(observations).float()
        rand_actions = torch.from_numpy(
            random_actions(observations.shape[0], k=n_actions)).float()
        if self.use_gpu:
            observations = observations.cuda()
            rand_actions = rand_actions.cuda()
        rewards = self.network.forward(observations, rand_actions)
        #print("rewards", rewards.shape)
        best_ids = rewards.argmax(dim=1)
        best_rewards = rewards.index_select(1, best_ids)
        #print("in record_game:", rewards.shape, best_rewards, best_ids.shape, rand_actions.shape)
        # best_actions = rand_actions.gather(1, best_ids).detach().cpu()
        best_actions = torch.stack([
            row[i].squeeze()
            for row, i in zip(torch.unbind(rand_actions, 0), best_ids.unbind(0))
        ], axis=0).detach().cpu()
        # print(best_actions.shape)
        assert len(best_actions.shape)==2 and best_actions.shape[1] == 14
        move_rotate = best_actions.index_select(-1, torch.tensor([0, 1])).numpy()
        chase_focus = (best_actions.index_select(-1, torch.tensor([2]))).numpy()
        #print(move_rotate[: 1, :])
        casts = best_actions.index_select(
            -1, torch.tensor([3, 4, 5, 6])
        ).argmax(-1, keepdim=True).numpy()
        focuses = best_actions.index_select(
            -1, torch.tensor([7, 8, 9, 10, 11, 12, 13])
        ).argmax(-1, keepdim=True).numpy()
        # print(move_rotate_cf.shape, casts.shape, focuses.shape)
        actions = np.concatenate([
            move_rotate,
            chase_focus,
            casts,
            focuses
        ], axis=-1)
        assert len(actions.shape) == 2 and actions.shape[1] == 5

        return actions
