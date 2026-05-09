"""Robot-centric accumulated LiDAR BEV observations."""

import math

import numpy as np


def build_obs_from_grids(wall_grid, dynamic_grid, robot_pose, goal_pose,
                         grid_size=48, resolution=0.05) -> np.ndarray:
    """Build a (4, H, W) float32 BEV tensor from stored occupancy grids."""
    wall = np.asarray(wall_grid, dtype=np.float32)
    dynamic = np.asarray(dynamic_grid, dtype=np.float32)
    if wall.shape != (grid_size, grid_size):
        wall = wall.reshape(grid_size, grid_size)
    if dynamic.shape != (grid_size, grid_size):
        dynamic = dynamic.reshape(grid_size, grid_size)

    robot = np.zeros((grid_size, grid_size), dtype=np.float32)
    goal = np.zeros((grid_size, grid_size), dtype=np.float32)
    center = grid_size // 2
    robot[center, center] = 1.0

    theta = float(np.asarray(robot_pose, dtype=np.float32)[2])
    heading_col = int(round(center + 3.0 * math.cos(theta)))
    heading_row = int(round(center - 3.0 * math.sin(theta)))
    if 0 <= heading_row < grid_size and 0 <= heading_col < grid_size:
        robot[heading_row, heading_col] = 0.7

    goal_pose = np.asarray(goal_pose, dtype=np.float32)
    local_goal = BEVAccumulator.to_robot_frame_static(
        goal_pose[:2].reshape(1, 2), np.asarray(robot_pose, dtype=np.float32)
    )
    rows, cols, inside = BEVAccumulator.to_pixel_static(
        local_goal, grid_size, resolution, return_inside=True
    )
    goal[rows[0], cols[0]] = 1.0 if inside[0] else 0.5

    return np.stack(
        [np.clip(wall, 0.0, 1.0), np.clip(dynamic, 0.0, 1.0), robot, goal],
        axis=0,
    ).astype(np.float32, copy=False)


class BEVAccumulator:
    """Accumulate LiDAR hits into a 4-channel robot-centric BEV tensor."""

    def __init__(self, grid_size=48, resolution=0.05, decay=0.85):
        self.grid_size = int(grid_size)
        self.resolution = float(resolution)
        self.decay = float(decay)
        self.fov_radius = self.grid_size * self.resolution / 2.0
        self.reset()

    def reset(self):
        self.wall_grid = np.zeros((self.grid_size, self.grid_size), np.float32)
        self.dynamic_grid = np.zeros((self.grid_size, self.grid_size), np.float32)

    def update(self, lidar_points_map: np.ndarray, robot_pose: np.ndarray,
               static_map: np.ndarray = None):
        points = np.asarray(lidar_points_map, dtype=np.float32)
        if points.size == 0:
            self.dynamic_grid *= self.decay
            return
        points = points.reshape(-1, 2)
        points_local = self._to_robot_frame(points, robot_pose)

        self.dynamic_grid *= self.decay
        rows, cols, inside = self._to_pixel(points_local, return_inside=True)
        if rows.size == 0:
            return

        if static_map is None:
            dyn_rows, dyn_cols = rows[inside], cols[inside]
            self.dynamic_grid[dyn_rows, dyn_cols] = 1.0
            return

        static_map = np.asarray(static_map)
        wall_mask = self._static_mask(rows, cols, inside, static_map)
        self.wall_grid[rows[wall_mask], cols[wall_mask]] = 1.0
        dyn_mask = inside & ~wall_mask
        self.dynamic_grid[rows[dyn_mask], cols[dyn_mask]] = 1.0

    def get_tensor(self, robot_pose: np.ndarray, goal_pose: np.ndarray) -> np.ndarray:
        return build_obs_from_grids(
            self.wall_grid,
            self.dynamic_grid,
            robot_pose,
            goal_pose,
            self.grid_size,
            self.resolution,
        )

    def get_raw_state(self) -> dict:
        return {
            'wall_grid': self.wall_grid.copy(),
            'dynamic_grid': self.dynamic_grid.copy(),
        }

    def _to_robot_frame(self, points_map: np.ndarray,
                        robot_pose: np.ndarray) -> np.ndarray:
        return self.to_robot_frame_static(points_map, robot_pose)

    def _to_pixel(self, points_local: np.ndarray, return_inside=False):
        return self.to_pixel_static(
            points_local, self.grid_size, self.resolution, return_inside
        )

    @staticmethod
    def to_robot_frame_static(points_map: np.ndarray,
                              robot_pose: np.ndarray) -> np.ndarray:
        points_map = np.asarray(points_map, dtype=np.float32).reshape(-1, 2)
        robot_pose = np.asarray(robot_pose, dtype=np.float32)
        dxdy = points_map - robot_pose[:2]
        theta = float(robot_pose[2])
        c = math.cos(theta)
        s = math.sin(theta)
        rot = np.array([[c, s], [-s, c]], dtype=np.float32)
        return dxdy @ rot.T

    @staticmethod
    def to_pixel_static(points_local: np.ndarray, grid_size: int,
                        resolution: float, return_inside=False):
        points_local = np.asarray(points_local, dtype=np.float32).reshape(-1, 2)
        center = grid_size // 2
        raw_cols = center + points_local[:, 0] / resolution
        raw_rows = center - points_local[:, 1] / resolution
        inside = (
            (raw_rows >= 0) & (raw_rows <= grid_size - 1)
            & (raw_cols >= 0) & (raw_cols <= grid_size - 1)
        )
        rows = np.clip(np.rint(raw_rows), 0, grid_size - 1).astype(np.int32)
        cols = np.clip(np.rint(raw_cols), 0, grid_size - 1).astype(np.int32)
        if return_inside:
            return rows, cols, inside
        return rows, cols

    def _static_mask(self, rows, cols, inside, static_map):
        mask = np.zeros(rows.shape, dtype=bool)
        valid = inside & (rows >= 0) & (rows < static_map.shape[0]) \
            & (cols >= 0) & (cols < static_map.shape[1])
        mask[valid] = static_map[rows[valid], cols[valid]] > 0
        return mask
