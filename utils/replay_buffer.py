import collections
import random
import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_limit=50000):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            if len(s.shape) == 2:
                size = s.shape[0]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
            else:
                size = 1
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append([r])
                s_prime_lst.append(s_prime)

            done_mask_lst.append([done_mask]*size)
        s_lst = torch.stack(s_lst, axis=0).float()

        a_lst = np.stack(a_lst, axis=0)

        r_lst = np.concatenate(r_lst, axis=0)
        s_prime_lst = torch.from_numpy(
            np.stack(s_prime_lst, axis=0)).float()
        done_mask_lst = np.concatenate(done_mask_lst, axis=0)

        return s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst

    def size(self):
        return len(self.buffer)
