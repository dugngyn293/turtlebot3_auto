"""Bird's-Eye-View renderer for TurtleBot3 DRL navigation.

Produces a fixed-size top-down image of the robot's local environment that
DreamerV3 uses as its observation.  The image encodes:

  Channel / colour      Meaning
  ──────────────────    ────────────────────────────────────────
  Black  (0,0,0)        Free space / background
  Red    (170,0,35)     Arena walls (border)
  Blue   (55,55,255)    Robot body + heading arrow
  Green  (120,255,0)    Goal position (cross marker)
  Orange (255,25,45)    Dynamic obstacles

The rendered image is normalised to [0,1] float32 and flattened before being
handed to the agent.  This is the only place where raw world-coordinates are
converted to pixels, so all arena/image size configuration lives here.

Changes vs original
───────────────────
* Added ``render_from_scan`` — an alternative renderer that builds the BEV
  from raw LiDAR ranges rather than Gazebo obstacle odometry.  This is
  needed for the real-robot environment (``drl_environment_real.py``) where
  obstacle position data is not available, but also works as a drop-in for
  the simulation environment when ``ENABLE_BEV_STATE=True``.
* PNG writer (``save_png``) unchanged — kept as a dependency-free writer.
* ``BEVRenderer.flatten`` normalises uint8 → float32 in [0, 1].
"""

import math
import os
import struct
import zlib

import numpy as np


# ── Colour palette ──────────────────────────────────────────────────────── #
BACKGROUND = np.array([0,   0,   0],   dtype=np.uint8)
WALL       = np.array([170, 0,   35],  dtype=np.uint8)
ROBOT      = np.array([55,  55,  255], dtype=np.uint8)
GOAL       = np.array([120, 255, 0],   dtype=np.uint8)
OBSTACLE   = np.array([255, 25,  45],  dtype=np.uint8)
LIDAR_HIT  = np.array([220, 120, 0],   dtype=np.uint8)   # scan-based obstacle


class BEVRenderer:
    """Renders a top-down BEV image from Gazebo world coordinates.

    Parameters
    ----------
    image_size   : int   — pixel width (= height) of the square output image
    arena_length : float — arena extent in the X direction (metres)
    arena_width  : float — arena extent in the Y direction (metres)
    margin       : int   — pixel border reserved for the wall outline
    """

    def __init__(self, image_size: int, arena_length: float,
                 arena_width: float, margin: int = 7):
        self.image_size   = image_size
        self.arena_length = arena_length
        self.arena_width  = arena_width
        self.margin       = margin

    # -------------------------------------------------------------------- #
    #                     Public rendering methods                          #
    # -------------------------------------------------------------------- #

    def render(self, robot_x: float, robot_y: float, robot_heading: float,
               goal_x: float, goal_y: float,
               obstacle_positions: list) -> np.ndarray:
        """Render BEV from Gazebo obstacle odometry (simulation only).

        Parameters
        ----------
        obstacle_positions : list of (x, y) tuples or None entries
        """
        image = np.full((self.image_size, self.image_size, 3), BACKGROUND,
                        dtype=np.uint8)
        self._draw_border(image)

        for obs in obstacle_positions:
            if obs is not None:
                self._draw_dot(image, obs[0], obs[1], OBSTACLE, radius=2)

        self._draw_cross(image, goal_x, goal_y, GOAL, arm=2, thickness=1)
        self._draw_robot(image, robot_x, robot_y, robot_heading)
        return image

    def render_from_scan(self, robot_x: float, robot_y: float,
                         robot_heading: float,
                         goal_x: float, goal_y: float,
                         scan_ranges: list,
                         angle_min: float = -math.pi,
                         angle_increment: float = None) -> np.ndarray:
        """Render BEV from raw LiDAR scan ranges.

        This variant does **not** require obstacle odometry, making it
        usable on real robots or when ``ENABLE_BEV_STATE=True`` but obstacle
        positions are unavailable.

        Parameters
        ----------
        scan_ranges       : normalised LiDAR distances in [0, 1]
        angle_min         : start angle of the scan (radians), default -π
        angle_increment   : angular step (radians); if None, computed as
                            2π / len(scan_ranges)
        """
        n = len(scan_ranges)
        if angle_increment is None:
            angle_increment = (2.0 * math.pi) / n

        image = np.full((self.image_size, self.image_size, 3), BACKGROUND,
                        dtype=np.uint8)
        self._draw_border(image)

        # Paint LiDAR hit points in world coordinates
        for i, r_norm in enumerate(scan_ranges):
            if r_norm >= 1.0:           # no obstacle detected (at cap distance)
                continue
            # Convert normalised range back to a world-frame hit point
            from ..common.settings import LIDAR_DISTANCE_CAP
            r_m = r_norm * LIDAR_DISTANCE_CAP
            angle = robot_heading + angle_min + i * angle_increment
            hit_x = robot_x + r_m * math.cos(angle)
            hit_y = robot_y + r_m * math.sin(angle)
            self._draw_dot(image, hit_x, hit_y, LIDAR_HIT, radius=1)

        self._draw_cross(image, goal_x, goal_y, GOAL, arm=2, thickness=1)
        self._draw_robot(image, robot_x, robot_y, robot_heading)
        return image

    def flatten(self, image: np.ndarray) -> list:
        """Convert uint8 HxWxC image → normalised float32 flat list."""
        return (image.astype(np.float32) / 255.0).reshape(-1).tolist()

    def save_png(self, path: str, image_or_state) -> None:
        save_png(path, self.as_image(image_or_state))

    def as_image(self, image_or_state: np.ndarray) -> np.ndarray:
        """Accept either a flat state vector or an (H, W, 3) uint8 array."""
        image = np.asarray(image_or_state)
        if image.ndim == 1:
            image = image.reshape(self.image_size, self.image_size, 3)
        if image.dtype != np.uint8:
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        return image

    # -------------------------------------------------------------------- #
    #                       Coordinate transforms                           #
    # -------------------------------------------------------------------- #

    def _world_to_pixel(self, x: float, y: float):
        usable = self.image_size - 2 * self.margin - 1
        px = self.margin + (x + self.arena_length / 2.0) / self.arena_length * usable
        py = self.margin + (self.arena_width / 2.0 - y) / self.arena_width * usable
        px = int(np.clip(round(px), 0, self.image_size - 1))
        py = int(np.clip(round(py), 0, self.image_size - 1))
        return px, py

    # -------------------------------------------------------------------- #
    #                         Drawing primitives                            #
    # -------------------------------------------------------------------- #

    def _draw_border(self, image: np.ndarray) -> None:
        m = self.margin
        e = self.image_size - m - 1
        image[m   : m+1, m:e+1]  = WALL
        image[e   : e+1, m:e+1]  = WALL
        image[m:e+1, m   : m+1]  = WALL
        image[m:e+1, e   : e+1]  = WALL

    def _draw_dot(self, image: np.ndarray, x: float, y: float,
                  color: np.ndarray, radius: int) -> None:
        px, py = self._world_to_pixel(x, y)
        for yy in range(py - radius, py + radius + 1):
            for xx in range(px - radius, px + radius + 1):
                if 0 <= xx < self.image_size and 0 <= yy < self.image_size:
                    if (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2:
                        image[yy, xx] = color

    def _draw_cross(self, image: np.ndarray, x: float, y: float,
                    color: np.ndarray, arm: int, thickness: int) -> None:
        px, py = self._world_to_pixel(x, y)
        for offset in range(-arm, arm + 1):
            for w in range(-thickness, thickness + 1):
                self._set_pixel(image, px + offset, py + w, color)
                self._set_pixel(image, px + w, py + offset, color)

    def _draw_robot(self, image: np.ndarray, x: float, y: float,
                    heading: float) -> None:
        self._draw_cross(image, x, y, ROBOT, arm=2, thickness=1)
        px, py = self._world_to_pixel(x, y)
        end_x = int(round(px + math.cos(heading) * 5))
        end_y = int(round(py - math.sin(heading) * 5))
        self._draw_line(image, px, py, end_x, end_y, ROBOT)

    def _draw_line(self, image: np.ndarray,
                   x0: int, y0: int, x1: int, y1: int,
                   color: np.ndarray) -> None:
        """Bresenham line from (x0,y0) to (x1,y1)."""
        dx =  abs(x1 - x0);  dy = -abs(y1 - y0)
        sx =  1 if x0 < x1 else -1
        sy =  1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self._set_pixel(image, x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy;  x0 += sx
            if e2 <= dx:
                err += dx;  y0 += sy

    def _set_pixel(self, image: np.ndarray, x: int, y: int,
                   color: np.ndarray) -> None:
        if 0 <= x < self.image_size and 0 <= y < self.image_size:
            image[y, x] = color


# ========================================================================== #
#                      Dependency-free PNG writer                             #
# ========================================================================== #

def save_png(path: str, image: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 numpy array to *path* as a PNG.

    Uses only the Python standard library (struct, zlib) — no Pillow or cv2
    required.  Safe to call from the ROS2 environment.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    image = np.asarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError('BEV PNG writer expects RGB (3-channel) images')

    # PNG filter byte 0x00 (None) before every row
    raw = b''.join(b'\x00' + image[y].tobytes() for y in range(height))

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        payload = chunk_type + data
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + payload + struct.pack('>I', crc)

    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, level=1))
    png += chunk(b'IEND', b'')

    with open(path, 'wb') as f:
        f.write(png)