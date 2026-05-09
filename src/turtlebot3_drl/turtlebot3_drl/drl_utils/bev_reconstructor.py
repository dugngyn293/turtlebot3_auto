"""Reconstruct flattened BEV observation batches from raw grid snapshots."""

import numpy as np

from ..common.lidar_accumulator import build_obs_from_grids


class BEVReconstructor:
    """Convert raw replay-buffer grid snapshots into DreamerV3 BEV tensors."""

    def __init__(self, grid_size=48, resolution=0.05):
        self.grid_size = int(grid_size)
        self.resolution = float(resolution)
        self.state_size = 4 * self.grid_size * self.grid_size

    def reconstruct_batch(self, raw_sequences: list) -> np.ndarray:
        batch_size = len(raw_sequences)
        if batch_size == 0:
            return np.zeros((0, 0, self.state_size), dtype=np.float32)
        sequence_length = len(raw_sequences[0])
        result = np.zeros(
            (batch_size, sequence_length, self.state_size), dtype=np.float32
        )

        for b, sequence in enumerate(raw_sequences):
            for t, step in enumerate(sequence):
                tensor = build_obs_from_grids(
                    step['wall_grid'],
                    step['dynamic_grid'],
                    step['robot_pose'],
                    step['goal_pose'],
                    self.grid_size,
                    self.resolution,
                )
                result[b, t] = tensor.reshape(-1)
        return result
