# DreamerV3 With BEV Observation For TurtleBot3 Navigation

## Slide 1: Project Goal

**Title:** Integrating DreamerV3 Into TurtleBot3 DRL Navigation

**Bullets:**
- Added a model-based RL agent based on DreamerV3.
- Replaced LiDAR-vector observation with BEV image observation.
- Target: learn obstacle-aware navigation in dynamic Gazebo environments.
- Key challenge: moving obstacles require temporal visual context.

**Speaker Notes:**
The goal of this work was to move beyond standard off-policy methods like TD3 and integrate DreamerV3, which learns a world model from sequences of observations. Since the environment contains moving obstacles, a single LiDAR scan or a single image is not enough. The agent needs temporal BEV observations to understand motion.

---

## Slide 2: Baseline System

**Title:** Existing TD3/DDPG Navigation Pipeline

**Bullets:**
- Original agents used compact LiDAR-based state vectors.
- Environment returned: laser scan, goal distance, goal angle, previous action.
- TD3 sampled random transitions from replay buffer.
- Goal and reset logic was handled by ROS2/Gazebo services.

**Speaker Notes:**
The original repo was built around TD3, DDPG, and DQN. TD3 works smoothly because it uses small vector observations and simple one-step transitions. DreamerV3 required deeper integration because it learns from sequences and reconstructs observations through a world model.

---

## Slide 3: DreamerV3 Integration

**Title:** New DreamerV3 Agent

**Bullets:**
- Added `dreamerv3.py` as a new algorithm option.
- Added world model components:
  - CNN encoder and decoder
  - RSSM recurrent state model
  - reward head
  - continue head
  - actor and critic
- Registered `dreamerv3` in the common `DrlAgent` training loop.

**Speaker Notes:**
DreamerV3 was added as a new agent while keeping the existing ROS2 training loop. The implementation includes a recurrent state-space model, categorical stochastic latent state, symlog/two-hot value targets, and imagined rollouts for actor-critic learning.

---

## Slide 4: BEV Observation

**Title:** Replacing LiDAR Vector With BEV Image Input

**Bullets:**
- Added BEV renderer in `common/bev.py`.
- Environment now returns a flattened RGB image.
- Default BEV size: `64 x 64 x 3`.
- Visual encoding:
  - black background
  - red boundary
  - blue robot with heading
  - green goal
  - red obstacle dots

**Speaker Notes:**
Instead of using raw LiDAR as the main state, the environment now renders a bird's-eye-view image. This image gives DreamerV3 a spatial layout of the robot, goal, walls, and dynamic obstacles. The CNN encoder processes this image directly.

---

## Slide 5: Temporal BEV Sequences

**Title:** Training With Consecutive Seconds Of BEV Images

**Bullets:**
- Replay buffer now supports sequence sampling.
- DreamerV3 samples image sequences instead of single transitions.
- Current design samples frames by real time:
  - `t`
  - `t + 1s`
  - `t + 2s`
  - ...
- This helps the world model infer obstacle motion direction.

**Speaker Notes:**
Because obstacles move randomly, one image cannot explain their motion. The replay buffer was extended so DreamerV3 can receive BEV images from consecutive seconds. This lets the world model learn velocity and direction patterns for dynamic obstacles.

---

## Slide 6: CNN Encoder And Decoder

**Title:** Image-Based World Model

**Bullets:**
- CNN encoder changed from 1D LiDAR processing to 2D image processing.
- CNN decoder reconstructs BEV images from Dreamer latent states.
- Reconstruction loss trains the latent model to preserve spatial information.
- Actor and critic operate on Dreamer latent states.

**Speaker Notes:**
The encoder compresses the BEV image into an embedding. The RSSM updates the latent state over time, and the decoder reconstructs the BEV image. This encourages the latent state to contain enough information about robot pose, goal location, walls, and obstacle positions.

---

## Slide 7: Debug Visualization

**Title:** BEV Debug Outputs

**Bullets:**
- Added debug images in the repo `media/` folder.
- Latest step image:
  - `media/turtlebot3_drl_bev_latest_step.png`
- Latest batch image:
  - `media/turtlebot3_drl_bev_latest_batch.png`
- Batch preview shows the latest temporal BEV sequence.

**Speaker Notes:**
To verify what DreamerV3 actually receives, two PNG outputs were added. The step image shows the most recent observation, while the batch image shows the sequence of frames used for training. This is important for validating temporal consistency.

---

## Slide 8: Reset And Episode Handling

**Title:** Cleaner Reset Logic For Model-Based Learning

**Bullets:**
- Immediate reset on:
  - wall collision
  - obstacle collision
  - tumble
  - goal success
- Added reset logs in environment.
- Reset prevents bad post-collision observations from entering training.
- Dreamer latent state resets when an episode ends.

**Speaker Notes:**
For DreamerV3, post-collision frames can corrupt the world model. The reset logic was tightened so invalid states do not continue into the replay buffer. When an episode ends, the agent resets its recurrent latent state.

---

## Slide 9: Gazebo Synchronization

**Title:** Keeping Robot And Obstacles Synchronized

**Bullets:**
- Physics is paused while waiting for a new goal.
- Physics is paused after reset.
- Robot and moving obstacles start together when the agent begins an episode.
- This avoids obstacles moving for several seconds before the robot acts.

**Speaker Notes:**
One practical issue was that Gazebo obstacles could continue moving while the agent was waiting for a new goal or preparing an episode. The system now pauses simulation during waiting/reset phases and unpauses only when the agent is ready to act.

---

## Slide 10: Training Workflow

**Title:** Running DreamerV3 Training

**Bullets:**
- Build package:
  - `colcon build --symlink-install --packages-select turtlebot3_drl`
- Run four ROS2 nodes:
  - Gazebo stage
  - `gazebo_goals`
  - `environment`
  - `train_agent dreamerv3`
- Dreamer warmup uses random driving to collect replay data.

**Speaker Notes:**
The training workflow stays similar to TD3, but the agent name is now `dreamerv3`. During the observe phase, the robot collects random driving data before model learning becomes effective.

---

## Slide 11: Current Limitations

**Title:** What Still Needs Validation

**Bullets:**
- DreamerV3 is heavier than TD3.
- BEV image input increases computation.
- Saving debug PNG every step can slow training.
- Sequence length must match episode duration.
- Needs runtime validation inside Docker with PyTorch and Gazebo.

**Speaker Notes:**
The main tradeoff is cost. DreamerV3 uses images, sequence batches, a world model, and imagined rollouts, so one training step is much heavier than TD3. The next phase should focus on profiling and tuning image size, sequence length, and debug frequency.

---

## Slide 12: Next Steps

**Title:** Next Improvements

**Bullets:**
- Reduce debug image saving frequency.
- Tune BEV resolution: `64x64` vs `32x32`.
- Tune temporal sequence:
  - sequence length
  - interval between frames
- Add quantitative comparison against TD3.
- Evaluate success rate in dynamic obstacle stages.

**Speaker Notes:**
The next work should compare DreamerV3 against TD3 under the same dynamic obstacle setup. Important metrics include success rate, collision rate, average reward, and training speed.

