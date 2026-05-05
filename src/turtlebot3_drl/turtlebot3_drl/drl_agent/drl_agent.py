#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert, Tomas

import copy
import os
import sys
import time
import numpy as np

from ..common.settings import (
    ENABLE_VISUAL, ENABLE_STACKING,
    OBSERVE_STEPS, MODEL_STORE_INTERVAL, GRAPH_DRAW_INTERVAL,
)

from ..common.storagemanager import StorageManager
from ..common.graph import Graph
from ..common.logger import Logger
if ENABLE_VISUAL:
    from ..common.visual import DrlVisual
from ..common import utilities as util

from .dqn import DQN
from .ddpg import DDPG
from .td3 import TD3
from .dreamerv3 import DreamerV3

from turtlebot3_msgs.srv import DrlStep, Goal
from std_srvs.srv import Empty

import rclpy
from rclpy.node import Node
from ..common.replaybuffer import ReplayBuffer


class DrlAgent(Node):
    def __init__(self, training, algorithm, load_session="", load_episode=0, real_robot=0):
        super().__init__(algorithm + '_agent')
        self.algorithm = algorithm
        self.training = int(training)
        self.load_session = load_session
        self.episode = int(load_episode)
        self.real_robot = real_robot

        if not self.training and not self.load_session:
            quit("\033[1m\033[93mInvalid command: Testing but no model to load specified, "
                 "see readme for correct format\033[0m}")

        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(util.stage) if not self.real_robot else 1
        print(f"{'training' if self.training else 'testing'} on stage: {util.stage}")
        self.total_steps = 0

        # ── Instantiate the requested algorithm ─────────────────────────── #
        if self.algorithm == 'dqn':
            self.model = DQN(self.device, self.sim_speed)
        elif self.algorithm == 'ddpg':
            self.model = DDPG(self.device, self.sim_speed)
        elif self.algorithm == 'td3':
            self.model = TD3(self.device, self.sim_speed)
        elif self.algorithm == 'dreamerv3':
            self.model = DreamerV3(self.device, self.sim_speed)
        else:
            quit("\033[1m\033[93m"
                 f"invalid algorithm specified ({self.algorithm}), "
                 "choose one of: dqn, ddpg, td3, dreamerv3\033[0m}")

        # DreamerV3 defines its own (lower) observe_steps; other algorithms
        # fall back to the global OBSERVE_STEPS value from settings.py.
        self.observe_steps = getattr(self.model, 'observe_steps', OBSERVE_STEPS)

        self.replay_buffer = ReplayBuffer(self.model.buffer_size)
        self.graph = Graph()

        # ── Model loading ─────────────────────────────────────────────────── #
        self.sm = StorageManager(
            self.algorithm, self.load_session, self.episode, self.device, util.stage
        )

        if self.load_session:
            del self.model
            self.model = self.sm.load_model()
            self.model.device = self.device
            self.sm.load_weights(self.model.networks)
            if self.training:
                self.replay_buffer.buffer = self.sm.load_replay_buffer(
                    self.model.buffer_size,
                    os.path.join(self.load_session,
                                 'stage' + str(self.sm.stage) + '_latest_buffer.pkl'),
                )
            self.total_steps = self.graph.set_graphdata(
                self.sm.load_graphdata(), self.episode
            )
            # Restore per-model observe_steps after reload too
            self.observe_steps = getattr(self.model, 'observe_steps', OBSERVE_STEPS)
            print(f"global steps: {self.total_steps}")
            print(f"loaded model {self.load_session} "
                  f"(eps {self.episode}): {self.model.get_model_parameters()}")
        else:
            self.sm.new_session_dir(util.stage)
            self.sm.store_model(self.model)

        self.graph.session_dir = self.sm.session_dir
        self.logger = Logger(
            self.training, self.sm.machine_dir, self.sm.session_dir,
            self.sm.session, self.model.get_model_parameters(),
            self.model.get_model_configuration(), str(util.stage),
            self.algorithm, self.episode,
        )
        if ENABLE_VISUAL:
            self.visual = DrlVisual(self.model.state_size, self.model.hidden_size)
            self.model.attach_visual(self.visual)

        # ── ROS clients / servers ─────────────────────────────────────────── #
        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.goal_comm_client = self.create_client(Goal, 'goal_comm')
        if not self.real_robot:
            self.gazebo_pause = self.create_client(Empty, '/pause_physics')
            self.gazebo_unpause = self.create_client(Empty, '/unpause_physics')
        self.process()

    # ======================================================================== #
    #                              Main loop                                    #
    # ======================================================================== #

    def process(self):
        util.pause_simulation(self, self.real_robot)

        while True:
            util.wait_new_goal(self)

            episode_done = False
            step = 0
            reward_sum = 0.0
            loss_critic = 0.0
            loss_actor = 0.0
            action_past = [0.0, 0.0]
            state = util.init_episode(self)

            # DreamerV3 keeps a recurrent state across steps — reset it at
            # the start of every episode regardless of training/testing mode.
            if self.algorithm == 'dreamerv3':
                self.model.reset_state()

            if ENABLE_STACKING:
                frame_buffer = [0.0] * (
                    self.model.state_size * self.model.stack_depth * self.model.frame_skip
                )
                state = (
                    [0.0] * (self.model.state_size * (self.model.stack_depth - 1))
                    + list(state)
                )
                next_state = [0.0] * (self.model.state_size * self.model.stack_depth)

            util.unpause_simulation(self, self.real_robot)
            time.sleep(0.5)
            episode_start = time.perf_counter()

            while not episode_done:
                # ── Action selection ────────────────────────────────────────── #
                if self.training and self.total_steps < self.observe_steps:
                    action = self.model.get_action_random()
                else:
                    action = self.model.get_action(
                        state, self.training, step, ENABLE_VISUAL
                    )

                action_current = action
                if self.algorithm == 'dqn':
                    action_current = self.model.possible_actions[action]

                # ── Environment step ────────────────────────────────────────── #
                next_state, reward, episode_done, outcome, distance_traveled = \
                    util.step(self, action_current, action_past)
                action_past = copy.deepcopy(action_current)
                reward_sum += reward

                if ENABLE_STACKING:
                    frame_buffer = (
                        frame_buffer[self.model.state_size:] + list(next_state)
                    )
                    next_state = []
                    for depth in range(self.model.stack_depth):
                        start = (
                            self.model.state_size * (self.model.frame_skip - 1)
                            + self.model.state_size * self.model.frame_skip * depth
                        )
                        next_state += frame_buffer[start : start + self.model.state_size]

                # ── Training ─────────────────────────────────────────────────── #
                if self.training:
                    self.replay_buffer.add_sample(
                        state, action, [reward], next_state, [episode_done]
                    )
                    # DreamerV3 needs at least sequence_length transitions in the
                    # buffer before it can sample; others just need batch_size.
                    min_buffer = getattr(
                        self.model, 'sequence_length', self.model.batch_size
                    )
                    if self.replay_buffer.get_length() >= max(
                        self.model.batch_size, min_buffer
                    ):
                        loss_c, loss_a = self.model._train(self.replay_buffer)
                        loss_critic += loss_c
                        loss_actor += loss_a

                if ENABLE_VISUAL:
                    self.visual.update_reward(reward_sum)

                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)

            # ── End of episode ────────────────────────────────────────────── #
            util.pause_simulation(self, self.real_robot)
            self.total_steps += step
            duration = time.perf_counter() - episode_start
            self.finish_episode(
                step, duration, outcome, distance_traveled,
                reward_sum, loss_critic, loss_actor,
            )

    # ======================================================================== #
    #                         Episode bookkeeping                              #
    # ======================================================================== #

    def finish_episode(self, step, eps_duration, outcome, dist_traveled,
                       reward_sum, loss_critic, loss_actor):
        if self.total_steps < self.observe_steps:
            print(f"Observe phase: {self.total_steps}/{self.observe_steps} steps")
            return

        self.episode += 1
        print(
            f"Epi: {self.episode:<5}R: {reward_sum:<8.0f}"
            f"outcome: {util.translate_outcome(outcome):<13}",
            end='',
        )
        print(
            f"steps: {step:<6}steps_total: {self.total_steps:<7}"
            f"time: {eps_duration:<6.2f}"
        )

        if not self.training:
            self.logger.update_test_results(step, outcome, dist_traveled, eps_duration, 0)
            return

        self.graph.update_data(step, self.total_steps, outcome, reward_sum,
                               loss_critic, loss_actor)
        self.logger.file_log.write(
            f"{self.episode}, {reward_sum}, {outcome}, {eps_duration}, "
            f"{step}, {self.total_steps}, {self.replay_buffer.get_length()}, "
            f"{loss_critic / step}, {loss_actor / step}\n"
        )

        if (self.episode % MODEL_STORE_INTERVAL == 0) or self.episode == 1:
            self.sm.save_session(
                self.episode, self.model.networks,
                self.graph.graphdata, self.replay_buffer.buffer,
            )
            self.logger.update_comparison_file(
                self.episode,
                self.graph.get_success_count(),
                self.graph.get_reward_average(),
            )
        if (self.episode % GRAPH_DRAW_INTERVAL == 0) or self.episode == 1:
            self.graph.draw_plots(self.episode)


# ========================================================================== #
#                               Entry points                                  #
# ========================================================================== #

def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    drl_agent = DrlAgent(*args)
    rclpy.spin(drl_agent)
    drl_agent.destroy()
    rclpy.shutdown()

def main_train(args=sys.argv[1:]):
    main(['1'] + list(args))

def main_test(args=sys.argv[1:]):
    main(['0'] + list(args))

def main_real(args=sys.argv[1:]):
    main(['0'] + list(args) + ['0'])

if __name__ == '__main__':
    main()