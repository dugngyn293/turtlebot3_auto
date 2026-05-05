#!/usr/bin/env python3
# DreamerV3 integration for turtlebot3_auto
# Implements the full DreamerV3 algorithm (Hafner et al., Nature 2025)
# integrated with the turtlebot3 DRL navigation framework.
#
# Architecture:
#   World Model  : RSSM (GRU deterministic + categorical stochastic latent)
#                  + Encoder/Decoder + RewardHead + ContinueHead
#   Behavior     : Actor-Critic trained entirely in imagination
#   Observation  : BEV image (64×64×3) flattened as float32 vector
#                  (ENABLE_BEV_STATE = True in settings.py)
#
# Key DreamerV3 innovations included:
#   - Symlog transform  : handles reward scale variability
#   - Twohot encoding   : richer gradient signal vs scalar regression
#   - Free bits         : prevents KL collapse
#   - Percentile norm   : stable actor-critic returns
#   - Unimix            : prevents categorical probability collapse
#   - Straight-through  : gradients through discrete sampling
#   - LayerNorm in RSSM : stabilises GRU hidden state magnitude
#   - Weighted continue : upweights rare terminal transitions

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.bev import save_png
from ..common.settings import (
    BEV_BATCH_IMAGE_PATH,
    BEV_IMAGE_SIZE,
    # DreamerV3-specific (added to settings.py)
    DREAMER_SEQUENCE_LENGTH,
    DREAMER_HORIZON,
    DREAMER_EMBED_SIZE,
    DREAMER_DETER_SIZE,
    DREAMER_STOCH_SIZE,
    DREAMER_STOCH_CLASSES,
    DREAMER_NUM_BINS,
    DREAMER_FREE_BITS,
    DREAMER_KL_COEF,
    DREAMER_LAMBDA,
    DREAMER_ENTROPY_COEF,
    DREAMER_WORLD_LR,
    DREAMER_ACTOR_LR,
    DREAMER_CRITIC_LR,
    DREAMER_CRITIC_EMA,
    DREAMER_GRAD_CLIP,
    DREAMER_HIDDEN_SIZE,
    DREAMER_OBSERVE_STEPS,
)
from .off_policy_agent import OffPolicyAgent, Network


# ========================================================================== #
#                           Utility functions                                 #
# ========================================================================== #

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x)*ln(|x|+1).  Compresses large reward scales."""
    return torch.where(x >= 0, torch.log1p(x), -torch.log1p(-x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse symlog: sign(x)*(exp(|x|)-1)."""
    return torch.where(x >= 0, torch.exp(x) - 1.0, 1.0 - torch.exp(-x))


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Soft one-hot over *bins* for scalar values *x* (in symlog space).

    Values between two adjacent bin centres activate both with proportional
    weights — richer gradient signal than a single-bin hard assignment.
    """
    x = x.clamp(bins[0].item(), bins[-1].item())
    below = (bins.unsqueeze(0) <= x.unsqueeze(-1)).sum(-1) - 1
    below = below.clamp(0, bins.numel() - 2)
    above = below + 1
    w_above = (x - bins[below]) / (bins[above] - bins[below] + 1e-8)
    w_above = w_above.clamp(0.0, 1.0)
    target = torch.zeros(*x.shape, bins.numel(), device=x.device)
    target.scatter_(-1, below.unsqueeze(-1), (1.0 - w_above).unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
    return target


def twohot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value in original space from twohot logits."""
    probs = F.softmax(logits, dim=-1)
    return symexp((probs * bins).sum(-1))


class ReturnNormalizer:
    """EMA percentile normaliser (Section 3 of DreamerV3 paper).

    scale = max(1, P95 – P5) so small-scale returns are never amplified.
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.low: float | None = None
        self.high: float | None = None

    def update(self, returns: torch.Tensor) -> None:
        low = torch.quantile(returns.detach().float(), 0.05).item()
        high = torch.quantile(returns.detach().float(), 0.95).item()
        if self.low is None:
            self.low, self.high = low, high
        else:
            self.low = self.decay * self.low + (1.0 - self.decay) * low
            self.high = self.decay * self.high + (1.0 - self.decay) * high

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        if self.low is None:
            return returns
        scale = max(1.0, self.high - self.low)
        return returns / scale


# ========================================================================== #
#                              Network Modules                                #
# ========================================================================== #

class MLP(nn.Module):
    """Feedforward network with LayerNorm + SiLU (DreamerV3 default)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [out_dim]
        blocks: list[nn.Module] = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks.append(nn.LayerNorm(dims[i + 1]))
                blocks.append(nn.SiLU())
        self.net = nn.Sequential(*blocks)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CnnEncoder(Network):
    """Convolutional BEV encoder.  BEV image → embedding vector.

    Input: flattened (H*W*C,) float32 image normalised to [0, 1].
    """

    def __init__(self, name: str, state_size: int, action_size: int,
                 hidden_size: int, embed_size: int = 256,
                 image_size: int = 64, image_channels: int = 3):
        super().__init__(name)
        self.image_size = image_size
        self.image_channels = image_channels

        # Four conv layers → AdaptiveAvgPool → 128 × 4 × 4 = 2048 features
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        cnn_out = 128 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(cnn_out, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.SiLU(),
        )
        self.apply(super().init_weights)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states shape: (..., H*W*C)
        original_shape = states.shape[:-1]
        imgs = states.reshape(-1, self.image_size, self.image_size, self.image_channels)
        imgs = imgs.permute(0, 3, 1, 2)               # NHWC → NCHW
        embed = self.fc(self.conv(imgs))
        return embed.reshape(*original_shape, -1)


class CnnDecoder(Network):
    """Convolutional BEV decoder.  Latent → reconstructed BEV image."""

    def __init__(self, name: str, latent_size: int, state_size: int,
                 hidden_size: int, image_size: int = 64, image_channels: int = 3):
        super().__init__(name)
        self.image_size = image_size
        self.image_channels = image_channels

        self.fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 128 * 4 * 4),
            nn.LayerNorm(128 * 4 * 4),
            nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
        )
        self.apply(super().init_weights)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        original_shape = latents.shape[:-1]
        x = self.fc(latents.reshape(-1, latents.shape[-1]))
        imgs = self.deconv(x.reshape(-1, 128, 4, 4))
        imgs = F.interpolate(imgs, size=(self.image_size, self.image_size),
                             mode='bilinear', align_corners=False)
        imgs = torch.sigmoid(imgs).permute(0, 2, 3, 1)         # NCHW → NHWC
        return imgs.reshape(*original_shape, -1)

    def loss(self, latents: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        pred = self.forward(latents)
        # Symlog MSE — handles image value scale consistently
        return F.mse_loss(symlog(pred), symlog(states))


class RSSM(Network):
    """Recurrent State-Space Model (the core of DreamerV3's world model).

    Deterministic state h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    Stochastic state  z_t ~ Categorical  (posterior uses real obs embed,
                                           prior uses h only — for dreaming)

    Key improvements vs naive RSSM:
      - Pre-GRU linear projection (as in DreamerV3 paper appendix)
      - LayerNorm after GRU hidden state (prevents magnitude explosion)
      - Unimix in categorical sampling (prevents probability collapse)
      - Straight-through gradient through one-hot sample
    """

    def __init__(self, name: str, embed_size: int, action_size: int,
                 hidden_size: int, deter_size: int = 256,
                 stoch_size: int = 16, classes: int = 16,
                 unimix: float = 0.01):
        super().__init__(name)
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.classes = classes
        self.stoch_flat = stoch_size * classes
        self.latent_size = deter_size + self.stoch_flat
        self.unimix = unimix

        # Pre-GRU projection: flatten z + action → hidden (stabilises GRU input)
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_flat + action_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_size, deter_size)
        self.gru_norm = nn.LayerNorm(deter_size)    # critical for stability

        # Prior: p(z | h)  — used during imagination
        self.prior = MLP(deter_size, self.stoch_flat, hidden=hidden_size, layers=1)

        # Posterior: q(z | h, embed(obs))  — used during observation
        self.posterior = MLP(deter_size + embed_size, self.stoch_flat,
                             hidden=hidden_size, layers=1)
        self.apply(super().init_weights)

    def initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.deter_size, device=device)
        z = torch.zeros(batch_size, self.stoch_size, self.classes, device=device)
        z[..., 0] = 1.0  # one-hot on class 0
        return h, z

    def get_latent(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z.reshape(z.shape[0], -1)], dim=-1)

    def _sample(self, logits: torch.Tensor):
        """Categorical sample with unimix + straight-through gradient."""
        shape = logits.shape[:-1] + (self.stoch_size, self.classes)
        logits = logits.reshape(shape)

        # Unimix: blend softmax with uniform to prevent dead classes
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / self.classes
        probs = (1.0 - self.unimix) * probs + self.unimix * uniform

        # Sample + straight-through one-hot
        idx = torch.multinomial(probs.reshape(-1, self.classes), 1)
        one_hot = F.one_hot(idx.squeeze(-1), self.classes).float()
        one_hot = one_hot.reshape(shape)
        z = one_hot + probs - probs.detach()   # straight-through
        return z, probs

    def observe_step(self, h, z, action, embed):
        """Advance one step using real observation embed."""
        x = self.pre_gru(torch.cat([z.reshape(z.shape[0], -1), action], dim=-1))
        h = self.gru_norm(self.gru(x, h))
        prior_z, prior_probs = self._sample(self.prior(h))
        post_z, post_probs = self._sample(
            self.posterior(torch.cat([h, embed], dim=-1))
        )
        return h, post_z, prior_probs, post_probs

    def imagine_step(self, h, z, action):
        """Advance one step without observation (pure dreaming)."""
        x = self.pre_gru(torch.cat([z.reshape(z.shape[0], -1), action], dim=-1))
        h = self.gru_norm(self.gru(x, h))
        z, _ = self._sample(self.prior(h))
        return h, z

    def observe_sequence(self, embeds, actions):
        """Process a full (B, T, ...) sequence for world model training."""
        B, T = embeds.shape[:2]
        h, z = self.initial_state(B, embeds.device)
        h_list, z_list, prior_list, post_list = [], [], [], []
        for t in range(T):
            h, z, prior_p, post_p = self.observe_step(h, z, actions[:, t], embeds[:, t])
            h_list.append(h)
            z_list.append(z)
            prior_list.append(prior_p)
            post_list.append(post_p)
        return (
            torch.stack(h_list, dim=1),
            torch.stack(z_list, dim=1),
            torch.stack(prior_list, dim=1),
            torch.stack(post_list, dim=1),
        )

    def kl_loss(self, prior_probs, post_probs, free_bits: float = 1.0):
        """KL divergence with free-bits regularisation.

        Two terms (dynamics + representation) following DreamerV3 appendix:
          dynamics     : KL(post || sg(prior))   trains the posterior
          representation: KL(sg(post) || prior)   trains the prior
        Free bits prevent KL collapse: clamp per-variable KL from below.
        """
        # Clamp near-zero to avoid log(0)
        eps = 1e-8
        p_prior = prior_probs.clamp(eps, 1.0)
        p_post = post_probs.clamp(eps, 1.0)

        # KL per categorical variable (sum over classes, shape B,T,stoch_size)
        kl_dyn = (p_post.detach() * (p_post.detach().log() - p_prior.log())).sum(-1)
        kl_rep = (p_post * (p_post.log() - p_prior.detach().log())).sum(-1)

        # Free bits: clamp each variable's KL from below
        free = free_bits / self.stoch_size
        kl_dyn = kl_dyn.clamp(min=free).sum(-1).mean()
        kl_rep = kl_rep.clamp(min=free).sum(-1).mean()
        return 0.5 * kl_dyn + 0.1 * kl_rep


class RewardHead(Network):
    """Twohot reward predictor: latent → distribution over reward bins."""

    def __init__(self, name: str, latent_size: int, hidden_size: int, num_bins: int):
        super().__init__(name)
        self.net = MLP(latent_size, num_bins, hidden=hidden_size, layers=2)
        self.apply(super().init_weights)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)

    def loss(self, latents: torch.Tensor, rewards: torch.Tensor,
             bins: torch.Tensor) -> torch.Tensor:
        target = twohot_encode(symlog(rewards), bins)
        log_probs = F.log_softmax(self(latents), dim=-1)
        return -(target * log_probs).sum(-1).mean()


class ContinueHead(Network):
    """Episode continuation predictor: latent → P(not done).

    Terminal transitions are rare, so we weight them 10× to prevent the
    model from always predicting "continue".
    """

    def __init__(self, name: str, latent_size: int, hidden_size: int):
        super().__init__(name)
        self.net = MLP(latent_size, 1, hidden=hidden_size, layers=2)
        self.apply(super().init_weights)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents).squeeze(-1)

    def loss(self, latents: torch.Tensor, continues: torch.Tensor) -> torch.Tensor:
        logits = self(latents)
        # Upweight terminal (continue=0) transitions 10×
        weight = torch.where(continues < 0.5,
                             torch.full_like(continues, 10.0),
                             torch.ones_like(continues))
        return F.binary_cross_entropy_with_logits(logits, continues, weight=weight)


class Actor(Network):
    """Continuous action actor with squashed Normal distribution.

    Outputs tanh-squashed actions in [-1, 1].
    During training we use the reparameterised sample for gradient flow
    through the imagined trajectories.
    """

    def __init__(self, name: str, latent_size: int, action_size: int,
                 hidden_size: int):
        super().__init__(name)
        self.action_size = action_size
        self.backbone = MLP(latent_size, hidden_size, hidden=hidden_size, layers=2)
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.std_head = nn.Linear(hidden_size, action_size)
        self.apply(super().init_weights)

    def forward(self, latents: torch.Tensor, visualize: bool = False) -> torch.Tensor:
        """Deterministic (mean) action — used during evaluation."""
        return torch.tanh(self.mean_head(self.backbone(latents)))

    def sample(self, latents: torch.Tensor):
        """Stochastic action sample (reparameterised) — used during training."""
        feat = self.backbone(latents)
        mean = self.mean_head(feat)
        std = F.softplus(self.std_head(feat)) + 0.1
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        entropy = dist.entropy().sum(-1)
        return action, log_prob.sum(-1), entropy


class Critic(Network):
    """Twohot value predictor: latent → distribution over return bins."""

    def __init__(self, name: str, latent_size: int, hidden_size: int, num_bins: int):
        super().__init__(name)
        self.net = MLP(latent_size, num_bins, hidden=hidden_size, layers=2)
        self.apply(super().init_weights)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)

    def loss(self, latents: torch.Tensor, returns: torch.Tensor,
             bins: torch.Tensor) -> torch.Tensor:
        target = twohot_encode(symlog(returns), bins)
        log_probs = F.log_softmax(self(latents), dim=-1)
        return -(target * log_probs).sum(-1).mean()


# ========================================================================== #
#                       Main DreamerV3 Agent                                  #
# ========================================================================== #

class DreamerV3(OffPolicyAgent):
    """DreamerV3 agent integrated with the turtlebot3_auto framework.

    Observation space: BEV image (BEV_IMAGE_SIZE × BEV_IMAGE_SIZE × 3),
                       flattened to a float32 vector.  Requires
                       ENABLE_BEV_STATE = True in settings.py.

    Action space     : 2 continuous actions [linear_vel, angular_vel],
                       output in [-1, 1] and scaled by the environment.

    Training loop    :
      Every step  → add (s, a, r, s', done) to replay buffer.
      Every step  → if buffer ≥ batch_size AND past observe phase:
                      1) sample_sequence of length DREAMER_SEQUENCE_LENGTH
                      2) train world model on real sequences
                      3) train actor-critic in imagination
      Observe phase → DREAMER_OBSERVE_STEPS random steps before any training.

    Run with:
      ros2 run turtlebot3_drl train_agent dreamerv3
      ros2 run turtlebot3_drl test_agent  dreamerv3 <model> <episode>
    """

    def __init__(self, device: torch.device, sim_speed: float):
        super().__init__(device, sim_speed)

        # ── Observation / action dimensions ─────────────────────────────── #
        self.image_size = BEV_IMAGE_SIZE
        self.image_channels = 3
        self.state_size = self.image_size * self.image_size * self.image_channels
        self.input_size = self.state_size   # overrides OffPolicyAgent default

        # ── Hyper-parameters (read from settings.py) ─────────────────────── #
        self.sequence_length = DREAMER_SEQUENCE_LENGTH
        self.horizon = DREAMER_HORIZON
        self.embed_size = DREAMER_EMBED_SIZE
        self.deter_size = DREAMER_DETER_SIZE
        self.stoch_size = DREAMER_STOCH_SIZE
        self.stoch_classes = DREAMER_STOCH_CLASSES
        self.num_bins = DREAMER_NUM_BINS
        self.free_bits = DREAMER_FREE_BITS
        self.kl_coef = DREAMER_KL_COEF
        self.lam = DREAMER_LAMBDA
        self.entropy_coef = DREAMER_ENTROPY_COEF
        self.world_lr = DREAMER_WORLD_LR
        self.actor_lr = DREAMER_ACTOR_LR
        self.critic_lr = DREAMER_CRITIC_LR
        self.critic_ema = DREAMER_CRITIC_EMA
        self.grad_clip = DREAMER_GRAD_CLIP
        self.dreamer_hidden = DREAMER_HIDDEN_SIZE
        self.observe_steps = DREAMER_OBSERVE_STEPS  # overrides OffPolicyAgent

        # Twohot reward/value bins — in symlog space
        # We put 255 bins between symlog(-20) and symlog(20) ≈ [-3.04, 3.04]
        _lo = float(symlog(torch.tensor(-20.0)))
        _hi = float(symlog(torch.tensor(20.0)))
        self.bins = torch.linspace(_lo, _hi, self.num_bins, device=self.device)

        # ── Networks ─────────────────────────────────────────────────────── #
        self.encoder = CnnEncoder(
            'encoder', self.input_size, self.action_size,
            self.dreamer_hidden, self.embed_size,
            self.image_size, self.image_channels,
        ).to(self.device)

        self.rssm = RSSM(
            'rssm', self.embed_size, self.action_size,
            self.dreamer_hidden, self.deter_size,
            self.stoch_size, self.stoch_classes,
        ).to(self.device)

        self.decoder = CnnDecoder(
            'decoder', self.rssm.latent_size, self.input_size,
            self.dreamer_hidden, self.image_size, self.image_channels,
        ).to(self.device)

        self.reward_head = RewardHead(
            'reward_head', self.rssm.latent_size,
            self.dreamer_hidden, self.num_bins,
        ).to(self.device)

        self.continue_head = ContinueHead(
            'continue_head', self.rssm.latent_size, self.dreamer_hidden,
        ).to(self.device)

        self.actor = Actor(
            'actor', self.rssm.latent_size,
            self.action_size, self.dreamer_hidden,
        ).to(self.device)

        self.critic = Critic(
            'critic', self.rssm.latent_size,
            self.dreamer_hidden, self.num_bins,
        ).to(self.device)

        # EMA target critic for stable value bootstrapping
        self.critic_target = Critic(
            'target_critic', self.rssm.latent_size,
            self.dreamer_hidden, self.num_bins,
        ).to(self.device)
        self.hard_update(self.critic_target, self.critic)

        # networks list is used by StorageManager for save/load
        self.networks = [
            self.encoder, self.rssm, self.decoder,
            self.reward_head, self.continue_head,
            self.actor, self.critic, self.critic_target,
        ]

        # ── Optimizers ───────────────────────────────────────────────────── #
        _world_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_head.parameters())
            + list(self.continue_head.parameters())
        )
        self.world_optimizer = torch.optim.Adam(
            _world_params, lr=self.world_lr, eps=1e-5
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=1e-5
        )

        # ── Auxiliary training state ──────────────────────────────────────── #
        self.return_normalizer = ReturnNormalizer()
        self._h: torch.Tensor | None = None
        self._z: torch.Tensor | None = None
        self._last_action: torch.Tensor | None = None

    # ====================================================================== #
    #                      Environment interaction API                        #
    # ====================================================================== #

    def reset_state(self) -> None:
        """Reset recurrent RSSM state at episode boundaries."""
        self._h = None
        self._z = None
        self._last_action = None

    def get_action_random(self) -> list:
        """Return a random action (used during observe phase)."""
        self.reset_state()
        return [float(np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0))
                for _ in range(self.action_size)]

    @torch.no_grad()
    def get_action(self, state, is_training: bool,
                   step: int, visualize: bool = False) -> list:
        """Select an action given the current BEV observation.

        Uses the posterior RSSM step so the internal belief stays
        grounded in real observations at every environment step.
        """
        if step == 0:
            self.reset_state()

        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)) \
                       .to(self.device).unsqueeze(0)

        if self._h is None:
            self._h, self._z = self.rssm.initial_state(1, self.device)
            self._last_action = torch.zeros(1, self.action_size, device=self.device)

        embed = self.encoder(state_t)
        self._h, self._z, _, _ = self.rssm.observe_step(
            self._h, self._z, self._last_action, embed
        )
        latent = self.rssm.get_latent(self._h, self._z)

        if is_training:
            action, _, _ = self.actor.sample(latent)
        else:
            action = self.actor(latent, visualize)

        self._last_action = action.detach()
        return action.squeeze(0).clamp(-1.0, 1.0).cpu().numpy().tolist()

    def train(self, state, action, reward, state_next, done) -> list:
        """Step-level train hook (DreamerV3 uses batch training via _train)."""
        return [0.0, 0.0]

    # ====================================================================== #
    #                         Batch training                                 #
    # ====================================================================== #

    def _train(self, replaybuffer) -> list:
        """Main training call — invoked by drl_agent.py after every step.

        Returns [world_loss + critic_loss, actor_loss] to match the
        (loss_critic, loss_actor) signature expected by DrlAgent.
        """
        batch = replaybuffer.sample_sequence(self.batch_size, self.sequence_length)
        if batch is None:
            return [0.0, 0.0]

        obs, actions, rewards, next_obs, dones = batch

        # Save a preview image for debugging (latest BEV in the batch)
        self._save_latest_batch_image(next_obs)

        # Move to device
        obs_t = torch.from_numpy(next_obs).to(self.device)       # (B, T, state_size)
        act_t = torch.from_numpy(actions).to(self.device)        # (B, T, action_size)
        rew_t = torch.from_numpy(rewards).to(self.device)        # (B, T)
        don_t = torch.from_numpy(dones).to(self.device)          # (B, T)

        world_loss = self._train_world_model(obs_t, act_t, rew_t, don_t)
        actor_loss, critic_loss = self._train_actor_critic(obs_t, act_t)

        self.soft_update(self.critic_target, self.critic, self.critic_ema)
        self.iteration += 1
        return [world_loss + critic_loss, actor_loss]

    # ====================================================================== #
    #                       World model training                              #
    # ====================================================================== #

    def _train_world_model(self, obs, actions, rewards, dones) -> float:
        """Phase 1: fit RSSM + decoder + reward head + continue head.

        Loss = decoder (symlog MSE)
             + reward  (twohot CE)
             + continue (weighted BCE)
             + kl_coef * KL(posterior || prior)  with free-bits
        """
        embeds = self.encoder(obs)                                 # (B, T, E)
        h_states, z_posts, prior_probs, post_probs = \
            self.rssm.observe_sequence(embeds, actions)
        latents = self._sequence_latents(h_states, z_posts)       # (B, T, L)
        flat_lat = latents.reshape(-1, latents.shape[-1])
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_rew = rewards.reshape(-1)
        flat_con = (1.0 - dones).reshape(-1)

        decoder_loss = self.decoder.loss(flat_lat, flat_obs)
        reward_loss = self.reward_head.loss(flat_lat, flat_rew, self.bins)
        continue_loss = self.continue_head.loss(flat_lat, flat_con)
        kl_loss = self.rssm.kl_loss(prior_probs, post_probs, self.free_bits)
        total = decoder_loss + reward_loss + continue_loss + self.kl_coef * kl_loss

        self.world_optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_head.parameters())
            + list(self.continue_head.parameters()),
            self.grad_clip,
        )
        self.world_optimizer.step()
        return float(total.detach().cpu())

    # ====================================================================== #
    #                     Actor-critic in imagination                         #
    # ====================================================================== #

    def _train_actor_critic(self, obs, actions):
        """Phase 2: imagine H steps forward and train actor + critic.

        Gradients for the actor flow through:
          actor → sampled_action → rssm.imagine_step → reward_head → returns
        This is much more informative than REINFORCE.
        """
        # Encode real observations to get posterior starting states (no grad)
        with torch.no_grad():
            embeds = self.encoder(obs)
            h_states, z_posts, _, _ = self.rssm.observe_sequence(embeds, actions)
            B, T = h_states.shape[:2]
            h = h_states.reshape(B * T, -1).detach()
            z = z_posts.reshape(B * T, *z_posts.shape[2:]).detach()

        # Imagine H steps forward — gradient flows through actor decisions
        imagined_latents: list[torch.Tensor] = []
        imagined_entropies: list[torch.Tensor] = []

        for _ in range(self.horizon):
            latent = self.rssm.get_latent(h, z)
            imagined_latents.append(latent)
            action, _, entropy = self.actor.sample(latent)
            imagined_entropies.append(entropy)
            h, z = self.rssm.imagine_step(h, z, action)

        # Add terminal latent for value bootstrapping
        imagined_latents.append(self.rssm.get_latent(h, z))
        latents = torch.stack(imagined_latents, dim=1)             # (N, H+1, L)
        flat_lat = latents[:, :-1].reshape(-1, latents.shape[-1])  # (N*H, L)

        # Predict rewards + continues from imagined states
        rewards = twohot_decode(self.reward_head(flat_lat), self.bins) \
                       .reshape(-1, self.horizon)
        continues = torch.sigmoid(self.continue_head(flat_lat)) \
                         .reshape(-1, self.horizon)

        # Bootstrap values from EMA target critic (no grad)
        with torch.no_grad():
            all_lat = latents.reshape(-1, latents.shape[-1])
            values = twohot_decode(self.critic_target(all_lat), self.bins) \
                          .reshape(-1, self.horizon + 1)

        returns = self._lambda_returns(rewards, values, continues)  # (N, H)
        self.return_normalizer.update(returns.detach())

        # ── Critic update ────────────────────────────────────────────────── #
        # Detach latents and targets — critic is trained on its own pass
        crit_lat = flat_lat.detach()
        crit_tgt = returns.reshape(-1).detach()
        critic_loss = self.critic.loss(crit_lat, crit_tgt, self.bins)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # ── Actor update ─────────────────────────────────────────────────── #
        norm_returns = self.return_normalizer.normalize(returns)
        entropy = torch.stack(imagined_entropies, dim=1).mean()
        actor_loss = -norm_returns.mean() - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        return float(actor_loss.detach().cpu()), float(critic_loss.detach().cpu())

    # ====================================================================== #
    #                           Helpers                                       #
    # ====================================================================== #

    def _lambda_returns(self, rewards, values, continues):
        """GAE-style λ-returns.

        G_t = r_t + γ·c_t·[(1-λ)·V_{t+1} + λ·G_{t+1}]
        """
        H = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        last = values[:, -1]
        for t in reversed(range(H)):
            bootstrap = (1.0 - self.lam) * values[:, t + 1] + self.lam * last
            last = rewards[:, t] + self.discount_factor * continues[:, t] * bootstrap
            returns[:, t] = last
        return returns

    def _sequence_latents(self, h_states, z_states):
        """Reshape sequence (B, T, ...) to (B, T, latent_size) tensor."""
        B, T = h_states.shape[:2]
        h = h_states.reshape(B * T, -1)
        z = z_states.reshape(B * T, self.stoch_size, self.stoch_classes)
        return self.rssm.get_latent(h, z).reshape(B, T, -1)

    def _save_latest_batch_image(self, batch_states: np.ndarray) -> None:
        """Write the most-recent BEV frame from the batch to disk for inspection."""
        img = batch_states[0, -1].reshape(
            self.image_size, self.image_size, self.image_channels
        )
        save_png(BEV_BATCH_IMAGE_PATH,
                 np.clip(img * 255.0, 0, 255).astype(np.uint8))

    def get_model_parameters(self) -> str:
        base = super().get_model_parameters()
        dreamer_params = [
            self.image_size, self.image_channels,
            self.sequence_length, self.horizon,
            self.embed_size, self.deter_size,
            self.stoch_size, self.stoch_classes,
            self.num_bins, self.free_bits,
            self.kl_coef, self.lam,
            self.world_lr, self.actor_lr, self.critic_lr,
        ]
        return base + ', dreamerv3: ' + ', '.join(map(str, dreamer_params))