"""Gym-style raw LiDAR BEV environment wrapper for DreamerV3 experiments.

The main training entry point in this repository still uses the ROS service
environment in ``drl_environment.py``.  This wrapper is a lightweight adapter
for scripts that want a Gym-compatible ``reset()`` / ``step()`` interface and
raw state dictionaries consumed by ``DreamerV3.get_action``.
"""

import math
import time

import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ..common import utilities as util
from ..common.settings import (
    LIDAR_DISTANCE_CAP,
    SPEED_ANGULAR_MAX,
    SPEED_LINEAR_MAX,
    THRESHOLD_COLLISION,
    THREHSOLD_GOAL,
    TOPIC_ODOM,
    TOPIC_SCAN,
    TOPIC_VELO,
)


class BEVEnvironment(Node):
    """ROS 2 / Gazebo wrapper returning raw LiDAR map points and poses."""

    def __init__(self, max_steps=500):
        super().__init__('bev_environment')
        self.max_steps = int(max_steps)
        self.goal_pose = np.zeros(3, dtype=np.float32)
        self.robot_pose = np.zeros(3, dtype=np.float32)
        self.scan_msg = None
        self.steps = 0
        self.previous_goal_distance = None

        self.cmd_vel_pub = self.create_publisher(Twist, TOPIC_VELO, 10)
        self.create_subscription(
            LaserScan, TOPIC_SCAN, self._scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(Odometry, TOPIC_ODOM, self._odom_callback, 10)

    def reset(self) -> dict:
        self.steps = 0
        self.previous_goal_distance = None
        self.cmd_vel_pub.publish(Twist())
        self._spin_until_ready()
        state = self._get_state()
        self.previous_goal_distance = self._goal_distance(state['robot_pose'])
        return state

    def step(self, action: list) -> tuple[dict, float, bool, dict]:
        twist = Twist()
        twist.linear.x = float(np.clip(action[0], -1.0, 1.0)) * SPEED_LINEAR_MAX
        twist.angular.z = float(np.clip(action[1], -1.0, 1.0)) * SPEED_ANGULAR_MAX
        self.cmd_vel_pub.publish(twist)

        time.sleep(0.05)
        rclpy.spin_once(self, timeout_sec=0.01)
        self.steps += 1

        state = self._get_state()
        reward, done, reason = self._compute_reward_done(state)
        if done:
            self.cmd_vel_pub.publish(Twist())
        return state, reward, done, {'reason': reason}

    def set_goal(self, x: float, y: float, theta: float = 0.0):
        self.goal_pose = np.asarray([x, y, theta], dtype=np.float32)

    def _scan_callback(self, msg: LaserScan):
        self.scan_msg = msg

    def _odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        _, _, yaw = util.euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_pose = np.asarray([pos.x, pos.y, yaw], dtype=np.float32)

    def _spin_until_ready(self):
        while rclpy.ok() and self.scan_msg is None:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _get_state(self) -> dict:
        return {
            'lidar_points_map': self._get_lidar_points_map_frame(),
            'robot_pose': self.robot_pose.copy(),
            'goal_pose': self.goal_pose.copy(),
        }

    def _get_lidar_points_map_frame(self) -> np.ndarray:
        if self.scan_msg is None:
            return np.zeros((0, 2), dtype=np.float32)

        points = []
        for i, scan_range in enumerate(self.scan_msg.ranges):
            r = float(scan_range)
            if not math.isfinite(r) or r <= 0.0 or r > LIDAR_DISTANCE_CAP:
                continue
            angle = self.scan_msg.angle_min + i * self.scan_msg.angle_increment
            angle += float(self.robot_pose[2])
            points.append((
                float(self.robot_pose[0]) + r * math.cos(angle),
                float(self.robot_pose[1]) + r * math.sin(angle),
            ))

        if len(points) > 360:
            indices = np.linspace(0, len(points) - 1, 360).astype(np.int32)
            points = [points[i] for i in indices]
        return np.asarray(points, dtype=np.float32).reshape(-1, 2)

    def _compute_reward_done(self, state):
        goal_distance = self._goal_distance(state['robot_pose'])
        if self.previous_goal_distance is None:
            self.previous_goal_distance = goal_distance

        progress = self.previous_goal_distance - goal_distance
        self.previous_goal_distance = goal_distance

        reward = -0.01 + 0.1 * progress
        done = False
        reason = ''

        min_range = self._min_scan_range()
        if goal_distance < THREHSOLD_GOAL:
            reward += 1.0
            done = True
            reason = 'goal'
        elif min_range < THRESHOLD_COLLISION:
            reward -= 1.0
            done = True
            reason = 'collision'
        elif self.steps >= self.max_steps:
            done = True
            reason = 'timeout'
        elif min_range < 0.5:
            reward -= 0.1

        return float(reward), done, reason

    def _goal_distance(self, robot_pose):
        diff = self.goal_pose[:2] - np.asarray(robot_pose, dtype=np.float32)[:2]
        return float(np.linalg.norm(diff))

    def _min_scan_range(self):
        if self.scan_msg is None:
            return LIDAR_DISTANCE_CAP
        ranges = [
            float(r) for r in self.scan_msg.ranges
            if math.isfinite(float(r)) and float(r) > 0.0
        ]
        return min(ranges) if ranges else LIDAR_DISTANCE_CAP
