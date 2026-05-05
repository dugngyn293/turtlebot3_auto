import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """Unified replay buffer for all DRL algorithms.

    Supports two sampling modes:
      - sample()          : IID transitions  (DQN / DDPG / TD3)
      - sample_sequence() : Contiguous episode-aligned sequences (DreamerV3)

    DreamerV3 requires sequences that stay within one episode so the RSSM
    can learn temporal dynamics without being confused by cross-episode
    boundary artefacts.  The sampler enforces this by rejecting any candidate
    window that contains a mid-sequence terminal flag.

    Buffer layout (each element is a tuple):
        (obs, action, [reward], next_obs, [done])
    where obs / next_obs are flat float32 arrays (BEV image or LiDAR state).
    """

    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)
        self.max_size = size

    # ---------------------------------------------------------------------- #
    #                         IID transition sampling                         #
    # ---------------------------------------------------------------------- #

    def sample(self, batchsize: int):
        """Sample a random mini-batch of individual transitions.

        Used by DQN, DDPG, TD3.
        """
        batchsize = min(batchsize, self.get_length())
        batch = random.sample(self.buffer, batchsize)
        s      = np.float32([t[0] for t in batch])
        a      = np.float32([t[1] for t in batch])
        r      = np.float32([t[2] for t in batch])
        ns     = np.float32([t[3] for t in batch])
        done   = np.float32([t[4] for t in batch])
        return s, a, r, ns, done

    # ---------------------------------------------------------------------- #
    #                       Sequence sampling (DreamerV3)                     #
    # ---------------------------------------------------------------------- #

    def sample_sequence(self, batchsize: int, sequence_length: int):
        """Sample *batchsize* contiguous, episode-aligned sequences.

        Each returned sequence has exactly *sequence_length* steps drawn
        from the same episode.  A terminal flag is only allowed at the
        very last position in the window (natural episode end).

        Returns
        -------
        obs       : (B, T, obs_dim)      — observations at each step
        actions   : (B, T, action_dim)   — actions taken
        rewards   : (B, T)               — scalar rewards
        next_obs  : (B, T, obs_dim)      — next observations
        dones     : (B, T)               — episode-done flags [0 or 1]

        Returns None when the buffer does not yet hold enough data.
        """
        n = self.get_length()
        if n < sequence_length:
            return None

        # Work on a snapshot so the deque can be appended during training
        transitions = list(self.buffer)
        max_start = n - sequence_length

        sequences = []
        # Allow up to 40× attempts to fill the batch so that, even in short
        # episodes, we almost always succeed without an infinite loop.
        attempts = 0
        max_attempts = batchsize * 40

        while len(sequences) < batchsize and attempts < max_attempts:
            attempts += 1
            start = random.randint(0, max_start)
            seq = transitions[start : start + sequence_length]

            # Reject if any transition *before* the last one is terminal.
            # done flag is stored as a list [float] → seq[i][4][0]
            if any(float(seq[i][4][0]) > 0.5 for i in range(len(seq) - 1)):
                continue

            sequences.append(seq)

        if not sequences:
            # Fallback: return whatever valid sequences we managed to collect
            # (can happen early in training with very short episodes).
            return None

        # Stack into arrays
        obs      = np.float32([[step[0] for step in seq] for seq in sequences])
        actions  = np.float32([[step[1] for step in seq] for seq in sequences])
        rewards  = np.float32([[step[2][0] for step in seq] for seq in sequences])
        next_obs = np.float32([[step[3] for step in seq] for seq in sequences])
        dones    = np.float32([[step[4][0] for step in seq] for seq in sequences])

        return obs, actions, rewards, next_obs, dones

    # ---------------------------------------------------------------------- #
    #                              Utilities                                  #
    # ---------------------------------------------------------------------- #

    def add_sample(self, s, a, r, new_s, done):
        """Push one transition onto the buffer (FIFO when full)."""
        self.buffer.append((s, a, r, new_s, done))

    def get_length(self) -> int:
        return len(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)