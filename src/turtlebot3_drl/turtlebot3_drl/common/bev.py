import math
import os
import struct
import zlib

import numpy as np


BACKGROUND = np.array([0, 0, 0], dtype=np.uint8)
WALL = np.array([170, 0, 35], dtype=np.uint8)
ROBOT = np.array([55, 55, 255], dtype=np.uint8)
GOAL = np.array([120, 255, 0], dtype=np.uint8)
OBSTACLE = np.array([255, 25, 45], dtype=np.uint8)


class BEVRenderer:
    def __init__(self, image_size, arena_length, arena_width, margin=7):
        self.image_size = image_size
        self.arena_length = arena_length
        self.arena_width = arena_width
        self.margin = margin

    def render(self, robot_x, robot_y, robot_heading, goal_x, goal_y, obstacle_positions):
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image[:, :] = BACKGROUND
        self._draw_border(image)

        for obstacle in obstacle_positions:
            if obstacle is not None:
                self._draw_dot(image, obstacle[0], obstacle[1], OBSTACLE, radius=2)

        self._draw_cross(image, goal_x, goal_y, GOAL, arm=2, thickness=1)
        self._draw_robot(image, robot_x, robot_y, robot_heading)
        return image

    def flatten(self, image):
        return (image.astype(np.float32) / 255.0).reshape(-1).tolist()

    def save_png(self, path, image_or_state):
        save_png(path, self.as_image(image_or_state))

    def as_image(self, image_or_state):
        image = np.asarray(image_or_state)
        if image.ndim == 1:
            image = image.reshape(self.image_size, self.image_size, 3)
        if image.dtype != np.uint8:
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        return image

    def _world_to_pixel(self, x, y):
        usable = self.image_size - 2 * self.margin - 1
        px = self.margin + (x + self.arena_length / 2.0) / self.arena_length * usable
        py = self.margin + (self.arena_width / 2.0 - y) / self.arena_width * usable
        px = int(np.clip(round(px), 0, self.image_size - 1))
        py = int(np.clip(round(py), 0, self.image_size - 1))
        return px, py

    def _draw_border(self, image):
        left = self.margin
        right = self.image_size - self.margin - 1
        top = self.margin
        bottom = self.image_size - self.margin - 1
        image[top:top + 1, left:right + 1] = WALL
        image[bottom:bottom + 1, left:right + 1] = WALL
        image[top:bottom + 1, left:left + 1] = WALL
        image[top:bottom + 1, right:right + 1] = WALL

    def _draw_dot(self, image, x, y, color, radius):
        px, py = self._world_to_pixel(x, y)
        for yy in range(py - radius, py + radius + 1):
            for xx in range(px - radius, px + radius + 1):
                if 0 <= xx < self.image_size and 0 <= yy < self.image_size:
                    if (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2:
                        image[yy, xx] = color

    def _draw_cross(self, image, x, y, color, arm, thickness):
        px, py = self._world_to_pixel(x, y)
        for offset in range(-arm, arm + 1):
            for width in range(-thickness, thickness + 1):
                self._set_pixel(image, px + offset, py + width, color)
                self._set_pixel(image, px + width, py + offset, color)

    def _draw_robot(self, image, x, y, heading):
        self._draw_cross(image, x, y, ROBOT, arm=2, thickness=1)
        px, py = self._world_to_pixel(x, y)
        end_x = int(round(px + math.cos(heading) * 5))
        end_y = int(round(py - math.sin(heading) * 5))
        self._draw_line(image, px, py, end_x, end_y, ROBOT)

    def _draw_line(self, image, x0, y0, x1, y1, color):
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self._set_pixel(image, x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _set_pixel(self, image, x, y, color):
        if 0 <= x < self.image_size and 0 <= y < self.image_size:
            image[y, x] = color


def save_png(path, image):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    image = np.asarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError('BEV PNG writer expects RGB images')

    raw = b''.join(b'\x00' + image[y].tobytes() for y in range(height))

    def chunk(chunk_type, data):
        payload = chunk_type + data
        crc = zlib.crc32(payload) & 0xffffffff
        return struct.pack('>I', len(data)) + payload + struct.pack('>I', crc)

    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, level=1))
    png += chunk(b'IEND', b'')

    with open(path, 'wb') as f:
        f.write(png)
