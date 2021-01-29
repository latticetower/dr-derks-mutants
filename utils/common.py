import numpy as np
import os
import shlex
import subprocess
import torch


def random_actions(batch_size=6, k=1, for_env=False):
    move = np.random.uniform(-1, 1, (batch_size * k, 1))
    rotate = np.random.uniform(-1, 1, (batch_size * k, 1))
    chase_focus = np.random.random(((batch_size * k, 1)))
    cast = np.random.randint(0, 3, (batch_size*k, ))
    focus = np.random.randint(0, 7, (batch_size*k, ))
    if for_env:
        cast = np.expand_dims(cast, -1)
        focus = np.expand_dims(focus, -1)
    else:
        chase_focus = chase_focus/2.0 - 0.5
        cast = torch.eye(3)[cast]
        focus = torch.eye(7)[focus]
    result = np.concatenate([move, rotate, chase_focus, cast, focus], axis=-1)
    return result.reshape(batch_size, k, -1)


def save_mp4_files(datadir, tg=False, episode=0, score=[], size=0, tags=[]):
    command = [
        "./runner.sh",
        "python utils/gif.py",
        "--datadir",
        datadir,
        "--episode",
        str(episode),
        "--size",
        str(size),
    ]
    if isinstance(score, float):
        score = [score]
    score = np.asarray(score).flatten()

    for s in score:
        command.extend(["--score", str(s)])
    if tg:
        command.append("--tg")
    for tag in tags:
        command.extend([
            "--tag",
            str(tag)
        ])
    #runner = command[0]
    command = " ".join(command)
    command_path = os.path.join(datadir, "command.sh")
    with open(command_path, 'w') as f:
        f.write(command)
    # command = "bash " + command_path
    # command = shlex.split(command)
    # print(command)
    subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT)
