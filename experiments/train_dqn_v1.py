"""Basic DQN training
"""
import collections
import random
from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import subprocess
import sys
from models.qnet import Qnet
from utils.replay_buffer import ReplayBuffer
from utils.tg_writer import TelegramWriter
import cv2

NPZ_FILENAME = "weights/model_v2.npz"
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
    "statueDamageTaken": -1,
    "manualBonus": 0,
    "victory": 1000,
    "loss": -100,
    "tie": 0,
    "teamSpirit": 0.5,
    "timeScaling": 0.8,
}


def train(q, q_target, memory, optimizer, batch_size=32):
    q.train()
    losses = []
    for i in range(10):
        loss = []
        optimizer.zero_grad()
        for i in range(10):
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
            q_a = q(state, actions).squeeze(1)

            rand_actions = torch.from_numpy(
                Qnet.random_actions(s_prime.shape[0], k=40)).float()
            max_q_prime = q_target(s_prime, rand_actions).argmax(dim=1)
            r = torch.from_numpy(r).float()
            done_mask = torch.from_numpy(done_mask).float()
            target = r + gamma * max_q_prime * done_mask
            loss.append(F.smooth_l1_loss(q_a, target.detach()))
        loss = torch.stack(loss).mean()
        #print("Loss:", loss.item())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def save_frames_as_gif(image_paths, path='./', filename='gym_animation.gif'):
    frames = []
    for image_file in image_paths:
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=150)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(
        os.path.join(path, filename), writer='imagemagick', fps=60)


async def make_screenshot(env, path='example.png'):
    # loop = asyncio.get_running_loop()
    await env.app.page.screenshot({'path': path})


def record_game(env, q, savedir, n_episode):
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
            Qnet.random_actions(observations.shape[0], k=40)).float()
        rewards = q.forward(observations, rand_actions)
        best_ids = rewards.argmax(dim=1, keepdim=True)
        best_rewards = rewards.gather(1, best_ids)
        print("in record_game:", best_rewards, best_ids.shape, rand_actions.shape)
        best_actions = rand_actions.gather(1, best_ids).detach().cpu()
        # print(best_actions.shape)
        assert best_actions.shape == (6, 13)
        move_rotate_cf = best_actions.index_select(-1, torch.tensor([0, 1, 2])).numpy()
        casts = best_actions.index_select(
            -1, torch.tensor([3, 4, 5])
        ).argmax(-1, keepdim=True).numpy()
        focuses = best_actions.index_select(
            -1, torch.tensor([6, 7, 8, 9, 10, 11, 12])
        ).argmax(-1, keepdim=True).numpy()
        # print(move_rotate_cf.shape, casts.shape, focuses.shape)
        actions = np.concatenate([
            move_rotate_cf,
            casts,
            focuses
        ], axis=-1)
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
    filename = save_frames_as_gif(
        image_frames, gamedir, filename=f'animation_{n_episode}.gif')
    torch.save(q.state_dict(),
        os.path.join(gamedir, "torch_model.pth"))
    return filename


def main(env, n_episodes=10000, start_training_at=2000, print_interval=20,
         batch_size=32, tgwriter=TelegramWriter()):
    # env = gym.make('CartPole-v1') 
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    memory = ReplayBuffer()

    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(-start_training_at, n_episodes):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        # Linear annealing from 8% to 1%
        observations = env.reset()
        done = False
        q.eval()
        while not done:
            observations = torch.from_numpy(observations).float()
            actions = q.random_actions(
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
            score += r
            if done:
                break
        if n_epi <= 0 or memory.size() <= batch_size:
            continue
        losses = train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi > 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {}, {}, n_buffer : {}, eps : {}%".format(
                n_epi, score, print_interval, memory.size(), epsilon))
            savedir = "saves"

            gif_path = record_game(env, q, savedir, n_epi)
            with tgwriter.post() as f:
                f.add_param("Episode", n_epi)
                f.add_param("Score", score)
                f.add_param("Memory used", memory.size())
                f.add_param("Start command", "".join(sys.argv))
                f.add_text(f"{experiment_tags}")
                f.add_media(gif_path)
                # f.add_text("#hashtag")
            score = 0.0
    # env.close()


if __name__ == '__main__':
    token = os.environ.get('TELEGRAM_BOT_TOKEN', None)
    channel = os.environ.get('TELEGRAM_CHANNEL', None)
    
    mode = "test"
    commit_hash = subprocess.check_output([
        'git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip('\n')
    experiment_tags = f"#dqn_v1 #commit_{str(commit_hash)}"
    tgwriter = TelegramWriter(token, channel, tags=experiment_tags)
    with tgwriter.post() as f:
        f.add_text("-"*50)
        f.add_text(f"Start new experiment {experiment_tags} ")
        f.add_param("Commit", commit_hash)
        f.add_param("Start command", "".join(sys.argv))
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
         tgwriter=tgwriter
    )
    env.close()