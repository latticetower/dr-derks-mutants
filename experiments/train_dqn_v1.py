"""Basic DQN training
"""
import argparse
import asyncio
import collections
import cv2

from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import os
import random
import shutil
import subprocess
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.qnet import Qnet
from utils.replay_buffer import ReplayBuffer
from utils.tg_writer import TelegramWriter
from utils.common import random_actions
from utils.common import save_mp4_files
from utils.common import seed_everything
from utils.common import make_screenshot


WEIGHTS_FILE = "weights/model_dqn_v1.pth"
SEED = 1337
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000

REWARD_FUNCTION = {
    "damageEnemyStatue": 4,
    "damageEnemyUnit": 2,
    "killEnemyStatue": 4,
    "killEnemyUnit": 2,
    "healFriendlyStatue": 1,
    "healTeammate1": 2,
    "healTeammate2": 2,
    "timeSpentHomeBase": 0,
    "timeSpentHomeTerritory": 0,
    "timeSpentAwayTerritory": 0,
    "timeSpentAwayBase": 1,
    "damageTaken": -1,
    "friendlyFire": -4,
    "healEnemy": -8,
    "fallDamageTaken": -100,
    "statueDamageTaken": -2,
    "manualBonus": 0,
    "victory": 1000,
    "loss": -100,
    "tie": 0,
    "teamSpirit": 0,
    "timeScaling": 0.8,
}


def train(q, q_target, memory, optimizer, batch_size=32, n_actions=50,
          use_gpu=False):
    q.train()
    losses = []
    for i in range(100):
        loss = 0.0
        optimizer.zero_grad()
        state, action, r, s_prime, done_mask = memory.sample(batch_size)
        assert action.shape[-1] == 5
        actions = np.concatenate([
            action[..., :3],
            np.eye(3)[action[..., 3].astype(np.int)],
            np.eye(7)[action[..., 4].astype(np.int)]
        ], axis=-1)
        actions = np.expand_dims(actions, axis=1)
        if len(actions.shape) != 3:
            print(action.shape)
            print(actions.shape)
            print(done_mask)
        actions = torch.from_numpy(actions).float()
        if use_gpu:
            state = state.cuda()
            actions = actions.cuda()
        q_a = q(state, actions).squeeze(1)

        rand_actions = torch.from_numpy(
            random_actions(s_prime.shape[0], k=n_actions)).float()
        if use_gpu:
            s_prime = s_prime.cuda()
            rand_actions = rand_actions.cuda()
        max_q_prime = q_target(s_prime, rand_actions).argmax(dim=1)
        r = torch.from_numpy(r).float().type_as(max_q_prime)
        done_mask = torch.from_numpy(done_mask).float().type_as(max_q_prime)
        target = r + gamma * max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target.detach())
        #print("Loss:", loss.item())
        losses.append(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()
    return losses


def record_game(env, q, savedir, n_episode, n_actions=50, use_gpu=False):
    q.eval()
    gamedir = os.path.join(savedir, f"{n_episode:010d}")
    if not os.path.exists(gamedir):
        os.makedirs(gamedir)
    q.eval()
    observations = env.reset()
    done = False
    score = 0
    i = 0
    image_frames = []
    while not done:
        observations = torch.from_numpy(observations).float()
        rand_actions = torch.from_numpy(
            random_actions(observations.shape[0], k=n_actions)).float()
        if use_cuda:
            observations = observations.cuda()
            rand_actions = rand_actions.cuda()
        rewards = q.forward(observations, rand_actions)
        print("rewards", rewards.shape)
        best_ids = rewards.argmax(dim=1)
        best_rewards = rewards.index_select(1, best_ids)
        print("in record_game:", rewards.shape, best_rewards, best_ids.shape, rand_actions.shape)
        # best_actions = rand_actions.gather(1, best_ids).detach().cpu()
        best_actions = torch.stack([
            row[i].squeeze()
            for row, i in zip(torch.unbind(rand_actions, 0), best_ids.unbind(0))
        ], axis=0).detach().cpu()
        # print(best_actions.shape)
        assert best_actions.shape == (6, 13)
        move_rotate = best_actions.index_select(-1, torch.tensor([0, 1])).numpy()
        chase_focus = (best_actions.index_select(-1, torch.tensor([2]))).numpy()
        print(move_rotate[: 1, :])
        casts = best_actions.index_select(
            -1, torch.tensor([3, 4, 5])
        ).argmax(-1, keepdim=True).numpy()
        focuses = best_actions.index_select(
            -1, torch.tensor([6, 7, 8, 9, 10, 11, 12])
        ).argmax(-1, keepdim=True).numpy()
        # print(move_rotate_cf.shape, casts.shape, focuses.shape)
        actions = np.concatenate([
            move_rotate,
            chase_focus,
            casts,
            focuses
        ], axis=-1)
        assert actions.shape == (6, 13)
        next_observations, r, done, info = env.step(actions)
        image_path = os.path.join(gamedir, f"frame_{i}.png")

        asyncio.get_event_loop().run_until_complete(
            make_screenshot(env, image_path))
        image_frames.append(image_path)
        done = np.all(done)
        observations = next_observations
        score += r
        i += 1
        if done:
            break
    # to do: save folder contents here + model
    #filename = save_frames_as_gif(
    #    image_frames, gamedir, filename=f'animation_{n_episode}.gif')
    model_path = os.path.join(gamedir, "torch_model.pth")
    torch.save(q.state_dict(), model_path)
    shutil.copy(model_path, WEIGHTS_FILE)
    return gamedir


def main(env, n_episodes=10000, start_training_at=2000, print_interval=20,
         batch_size=32, experiment_tags=[], tg=False, n_actions=50,
         use_gpu=False):
    # env = gym.make('CartPole-v1') 
    q = Qnet()
    q_target = Qnet()
    if use_cuda:
        q.cuda()
        q_target.cuda()
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    memory = ReplayBuffer()

    score = None
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(-start_training_at, n_episodes):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        # Linear annealing from 8% to 1%
        observations = env.reset()
        done = False

        while not done:
            observations = torch.from_numpy(observations).float()
            actions = random_actions(
                observations.shape[0], k=1, for_env=True).squeeze(1)
            # observations, epsilon)
            #print("shape", actions.shape)
            # print(actions)
            next_observations, r, done, info = env.step(actions)
            done = np.all(done)

            done_mask = 0.0 if done else 1.0
            memory.put((observations, actions, r, next_observations, done_mask))
            observations = next_observations
            # print(next_observations.shape)
            #print("R=", r)
            if score is None:
                score = np.asarray([x for x in r])
            else:
                score += np.asarray([x for x in r])
            if done:
                break
        if n_epi <= 0 or memory.size() <= batch_size:
            continue
        losses = train(q, q_target, memory, optimizer, n_actions=n_actions,
                       use_gpu=use_gpu)

        if n_epi % print_interval == 0 and n_epi > 0:
            q.eval()
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {}, {}, n_buffer : {}, eps : {}%".format(
                n_epi, score, print_interval, memory.size(), epsilon))
            savedir = "saves"

            image_dir = record_game(env, q, savedir, n_epi, use_gpu=use_gpu)
            save_mp4_files(
                image_dir, tg=tg, episode=n_epi, score=score,
                size=memory.size(), tags=experiment_tags)
            # f.add_text("#hashtag")
            score = None
    # env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--savedir", default="saves",
        help="directory where the run results are saved")
    parser.add_argument(
        "-m", default=None,
        help="additional text description for current run")
    parser.add_argument(
        "--notg", default=False, action="store_true",
        help="don't write to telegram")
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="random seed to make run deterministic")
    parser.add_argument(
        "--n_actions", type=int, default=50,
        help="number of actions to sample")
    parser.add_argument(
        "--gpu", action="store_true", default=True,
        help="Use GPU if available (default device)"
    )

    args = parser.parse_args()
    
    seed_everything(args.seed)

    token = os.environ.get('TELEGRAM_BOT_TOKEN', None)
    channel = os.environ.get('TELEGRAM_CHANNEL', None)
    if args.notg:
        token = None
        channel = None

    mode = "test"
    commit_hash = subprocess.check_output([
        'git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip('\n')
    experiment_tags = ["dqn_v1", f"commit_{str(commit_hash)}"]
    tgwriter = TelegramWriter(token, channel)
    with tgwriter.post() as f:
        f.add_text("-"*50)
        tags = " ".join(["#" + t for t in experiment_tags])
        f.add_text(f"Start new experiment {tags} ")
        f.add_param("Commit", commit_hash)
        f.add_param("Start command", repr("".join(sys.argv)))
        # f.add_media("protein.gif")
        # f.add_text("#hashtag")

    env = DerkEnv(
        mode="train",
        turbo_mode=True,
        home_team=[
            {'primaryColor': '#3AA8C1'},
            {'primaryColor': '#BD559C', 'slots': ['Talons', None, None]},
            {'primaryColor': '#832A0D', 'rewardFunction': {'healTeammate1': 1}}
        ],
        away_team=[
            {'primaryColor': '#2D5DA1'},
            {'primaryColor': '#D05340', 'slots': ['Talons', None, None]},
            {'primaryColor': '#FBE870', 'rewardFunction': {'healTeammate1': 1}}
        ],
        session_args = {
            "reward_function": REWARD_FUNCTION
        }
    )
    main(env,
         n_episodes=100,
         start_training_at=10,
         print_interval=2,
         batch_size=4,
         experiment_tags=experiment_tags,
         tg=(not args.notg),
         n_actions=args.n_actions,
         use_gpu=(args.gpu and torch.cuda.is_available())
    )
    env.close()