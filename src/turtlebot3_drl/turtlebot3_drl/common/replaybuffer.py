import numpy as np
import random
from collections import deque
import itertools


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.max_size = size

    def sample(self, batchsize):
        batch = []
        batchsize = min(batchsize, self.get_length())
        batch = random.sample(self.buffer, batchsize)
        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])

        return s_array, a_array, r_array, new_s_array, done_array

    def sample_sequence(self, batchsize, sequence_length):
        batchsize = min(batchsize, self.get_length())
        if self.get_length() < sequence_length:
            return None

        transitions = list(self.buffer)
        sequences = []
        max_start = self.get_length() - sequence_length
        attempts = 0

        while len(sequences) < batchsize and attempts < batchsize * 20:
            attempts += 1
            start = random.randint(0, max_start)
            sequence = transitions[start:start + sequence_length]

            # Keep sequences inside one episode. A terminal flag is allowed only
            # at the final transition, where it naturally ends the sequence.
            if any(float(step[4][0]) > 0.5 for step in sequence[:-1]):
                continue
            sequences.append(sequence)

        if not sequences:
            return None

        obs = np.float32([[step[0] for step in sequence] for sequence in sequences])
        actions = np.float32([[step[1] for step in sequence] for sequence in sequences])
        rewards = np.float32([[step[2][0] for step in sequence] for sequence in sequences])
        next_obs = np.float32([[step[3] for step in sequence] for sequence in sequences])
        dones = np.float32([[step[4][0] for step in sequence] for sequence in sequences])

        return obs, actions, rewards, next_obs, dones

    def get_length(self):
        return len(self.buffer)

    def add_sample(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.buffer.append(transition)
