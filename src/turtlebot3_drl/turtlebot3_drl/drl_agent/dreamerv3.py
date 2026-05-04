import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.bev import save_png
from ..common.settings import BEV_BATCH_IMAGE_PATH, BEV_IMAGE_SIZE
from .off_policy_agent import OffPolicyAgent, Network


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def unimix(logits, mix=0.01):
    probs = torch.softmax(logits, dim=-1)
    return (1.0 - mix) * probs + mix / probs.shape[-1]


def straight_through_sample(probs):
    index = torch.distributions.Categorical(probs=probs).sample()
    sample = F.one_hot(index, probs.shape[-1]).float()
    return sample + probs - probs.detach()


def twohot_encode(values, bins):
    values = symlog(values).clamp(bins[0].item(), bins[-1].item())
    index = torch.bucketize(values, bins).clamp(1, bins.numel() - 1)
    low = bins[index - 1]
    high = bins[index]
    weight_high = (values - low) / (high - low).clamp_min(1e-6)
    weight_low = 1.0 - weight_high
    target = torch.zeros(values.shape + (bins.numel(),), device=values.device)
    target.scatter_add_(-1, (index - 1).unsqueeze(-1), weight_low.unsqueeze(-1))
    target.scatter_add_(-1, index.unsqueeze(-1), weight_high.unsqueeze(-1))
    return target


def twohot_decode(logits, bins):
    probs = torch.softmax(logits, dim=-1)
    return symexp(torch.sum(probs * bins, dim=-1))


class ReturnNormalizer:
    def __init__(self, decay=0.99, eps=1e-6):
        self.decay = decay
        self.eps = eps
        self.low = None
        self.high = None

    def update(self, returns):
        low = torch.quantile(returns.detach().flatten(), 0.05)
        high = torch.quantile(returns.detach().flatten(), 0.95)
        if self.low is None:
            self.low = low
            self.high = high
        else:
            self.low = self.decay * self.low + (1.0 - self.decay) * low
            self.high = self.decay * self.high + (1.0 - self.decay) * high

    def normalize(self, returns):
        if self.low is None:
            return returns
        scale = torch.clamp(self.high - self.low, min=1.0)
        return returns / (scale + self.eps)


class CnnEncoder(Network):
    def __init__(self, name, state_size, action_size, hidden_size, embed_size=256, image_size=64, image_channels=3):
        super().__init__(name)
        self.image_size = image_size
        self.image_channels = image_channels
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, embed_size),
            nn.ELU(),
        )
        self.apply(super().init_weights)

    def forward(self, states):
        original_shape = states.shape[:-1]
        images = states.reshape(-1, self.image_size, self.image_size, self.image_channels)
        images = images.permute(0, 3, 1, 2)
        embed = self.fc(self.conv(images))
        return embed.reshape(*original_shape, -1)


class CnnDecoder(Network):
    def __init__(self, name, latent_size, state_size, hidden_size, image_size=64, image_channels=3):
        super().__init__(name)
        self.image_size = image_size
        self.image_channels = image_channels
        self.fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 128 * 4 * 4),
            nn.ELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
        )
        self.apply(super().init_weights)

    def forward(self, latents):
        original_shape = latents.shape[:-1]
        latents = latents.reshape(-1, latents.shape[-1])
        image_seed = self.fc(latents).reshape(-1, 128, 4, 4)
        images = self.deconv(image_seed)
        images = F.interpolate(images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        images = torch.sigmoid(images).permute(0, 2, 3, 1)
        return images.reshape(*original_shape, -1)

    def loss(self, latents, states):
        return F.mse_loss(self(latents), states)


class RSSM(Network):
    def __init__(self, name, embed_size, action_size, hidden_size, deter_size=256, stoch_size=16, classes=16):
        super().__init__(name)
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.classes = classes
        self.stoch_flat_size = stoch_size * classes
        self.latent_size = deter_size + self.stoch_flat_size

        self.gru = nn.GRUCell(self.stoch_flat_size + action_size, deter_size)
        self.prior = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.stoch_flat_size),
        )
        self.posterior = nn.Sequential(
            nn.Linear(deter_size + embed_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.stoch_flat_size),
        )
        self.apply(super().init_weights)

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.deter_size, device=device)
        z = torch.zeros(batch_size, self.stoch_size, self.classes, device=device)
        z[..., 0] = 1.0
        return h, z

    def get_latent(self, h, z):
        return torch.cat((h, z.reshape(z.shape[0], -1)), dim=-1)

    def _dist(self, logits):
        return unimix(logits.reshape(-1, self.stoch_size, self.classes))

    def forward(self, h, z, action):
        return self.imagine_step(h, z, action)

    def observe_step(self, h, z, action, embed):
        h = self.gru(torch.cat((z.reshape(z.shape[0], -1), action), dim=-1), h)
        prior_probs = self._dist(self.prior(h))
        post_probs = self._dist(self.posterior(torch.cat((h, embed), dim=-1)))
        z = straight_through_sample(post_probs)
        return h, z, prior_probs, post_probs

    def imagine_step(self, h, z, action):
        h = self.gru(torch.cat((z.reshape(z.shape[0], -1), action), dim=-1), h)
        prior_probs = self._dist(self.prior(h))
        z = straight_through_sample(prior_probs)
        return h, z

    def observe_sequence(self, embeds, actions):
        batch_size, sequence_length = embeds.shape[:2]
        h, z = self.initial_state(batch_size, embeds.device)
        h_states, z_states, priors, posts = [], [], [], []

        for t in range(sequence_length):
            h, z, prior, post = self.observe_step(h, z, actions[:, t], embeds[:, t])
            h_states.append(h)
            z_states.append(z)
            priors.append(prior)
            posts.append(post)

        return (
            torch.stack(h_states, dim=1),
            torch.stack(z_states, dim=1),
            torch.stack(priors, dim=1),
            torch.stack(posts, dim=1),
        )

    def kl_loss(self, prior_probs, post_probs, free_bits):
        prior = torch.distributions.Categorical(probs=prior_probs.detach())
        post = torch.distributions.Categorical(probs=post_probs)
        dynamics_kl = torch.distributions.kl_divergence(post, prior)

        prior = torch.distributions.Categorical(probs=prior_probs)
        post = torch.distributions.Categorical(probs=post_probs.detach())
        representation_kl = torch.distributions.kl_divergence(post, prior)

        dynamics_kl = torch.clamp(dynamics_kl, min=free_bits).mean()
        representation_kl = torch.clamp(representation_kl, min=free_bits).mean()
        return 0.5 * dynamics_kl + 0.1 * representation_kl


class RewardHead(Network):
    def __init__(self, name, latent_size, hidden_size, num_bins):
        super().__init__(name)
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, num_bins),
        )
        self.apply(super().init_weights)

    def forward(self, latents):
        return self.net(latents)

    def loss(self, latents, rewards, bins):
        target = twohot_encode(rewards, bins)
        return -(target * F.log_softmax(self(latents), dim=-1)).sum(-1).mean()


class ContinueHead(Network):
    def __init__(self, name, latent_size, hidden_size):
        super().__init__(name)
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )
        self.apply(super().init_weights)

    def forward(self, latents):
        return self.net(latents).squeeze(-1)

    def loss(self, latents, continues):
        return F.binary_cross_entropy_with_logits(self(latents), continues)


class Actor(Network):
    def __init__(self, name, latent_size, action_size, hidden_size):
        super().__init__(name)
        self.backbone = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        self.mean = nn.Linear(hidden_size, action_size)
        self.std = nn.Linear(hidden_size, action_size)
        self.apply(super().init_weights)

    def forward(self, latents, visualize=False):
        features = self.backbone(latents)
        return torch.tanh(self.mean(features))

    def sample(self, latents):
        features = self.backbone(latents)
        mean = self.mean(features)
        std = F.softplus(self.std(features)) + 0.1
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        entropy = dist.entropy().sum(-1)
        return action, log_prob.sum(-1), entropy


class Critic(Network):
    def __init__(self, name, latent_size, hidden_size, num_bins):
        super().__init__(name)
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, num_bins),
        )
        self.apply(super().init_weights)

    def forward(self, latents):
        return self.net(latents)

    def loss(self, latents, returns, bins):
        target = twohot_encode(returns, bins)
        return -(target * F.log_softmax(self(latents), dim=-1)).sum(-1).mean()


class DreamerV3(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        self.image_size = BEV_IMAGE_SIZE
        self.image_channels = 3
        self.state_size = self.image_size * self.image_size * self.image_channels
        self.input_size = self.state_size
        self.sequence_length = 16
        self.horizon = 15
        self.embed_size = 256
        self.deter_size = 256
        self.stoch_size = 16
        self.stoch_classes = 16
        self.num_bins = 255
        self.free_bits = 1.0
        self.kl_coef = 0.5
        self.lam = 0.95
        self.entropy_coef = 3e-4
        self.actor_lr = 3e-5
        self.critic_lr = 3e-5
        self.world_lr = 1e-4

        self.bins = torch.linspace(-20.0, 20.0, self.num_bins, device=self.device)
        self.encoder = CnnEncoder(
            'encoder',
            self.input_size,
            self.action_size,
            self.hidden_size,
            self.embed_size,
            self.image_size,
            self.image_channels,
        ).to(self.device)
        self.rssm = RSSM('rssm', self.embed_size, self.action_size, self.hidden_size, self.deter_size, self.stoch_size, self.stoch_classes).to(self.device)
        self.decoder = CnnDecoder(
            'decoder',
            self.rssm.latent_size,
            self.input_size,
            self.hidden_size,
            self.image_size,
            self.image_channels,
        ).to(self.device)
        self.reward_head = RewardHead('reward_head', self.rssm.latent_size, self.hidden_size, self.num_bins).to(self.device)
        self.continue_head = ContinueHead('continue_head', self.rssm.latent_size, self.hidden_size).to(self.device)
        self.actor = Actor('actor', self.rssm.latent_size, self.action_size, self.hidden_size).to(self.device)
        self.critic = Critic('critic', self.rssm.latent_size, self.hidden_size, self.num_bins).to(self.device)
        self.critic_target = Critic('target_critic', self.rssm.latent_size, self.hidden_size, self.num_bins).to(self.device)
        self.hard_update(self.critic_target, self.critic)

        self.networks = [
            self.encoder,
            self.rssm,
            self.decoder,
            self.reward_head,
            self.continue_head,
            self.actor,
            self.critic,
            self.critic_target,
        ]

        world_params = (
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_head.parameters()) +
            list(self.continue_head.parameters())
        )
        self.world_optimizer = torch.optim.Adam(world_params, lr=self.world_lr, eps=1e-5)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)
        self.return_normalizer = ReturnNormalizer()
        self._h = None
        self._z = None
        self._last_action = None

    def get_action_random(self):
        self.reset_state()
        return [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)] * self.action_size

    def reset_state(self):
        self._h = None
        self._z = None
        self._last_action = None

    def get_action(self, state, is_training, step, visualize=False):
        if step == 0:
            self.reset_state()

        with torch.no_grad():
            state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device).unsqueeze(0)
            if self._h is None:
                self._h, self._z = self.rssm.initial_state(1, self.device)
                self._last_action = torch.zeros(1, self.action_size, device=self.device)

            embed = self.encoder(state)
            self._h, self._z, _, _ = self.rssm.observe_step(self._h, self._z, self._last_action, embed)
            latent = self.rssm.get_latent(self._h, self._z)

            if is_training:
                action, _, _ = self.actor.sample(latent)
            else:
                action = self.actor(latent, visualize)

            self._last_action = action
            return action.squeeze(0).clamp(-1.0, 1.0).cpu().numpy().tolist()

    def _train(self, replaybuffer):
        batch = replaybuffer.sample_sequence(self.batch_size, self.sequence_length)
        if batch is None:
            return [0.0, 0.0]

        obs, actions, rewards, next_obs, dones = batch
        self._save_latest_batch_image(next_obs)
        obs = torch.from_numpy(next_obs).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        world_loss = self._train_world_model(obs, actions, rewards, dones)
        actor_loss, critic_loss = self._train_actor_critic(obs, actions)
        self.soft_update(self.critic_target, self.critic, 0.02)
        self.iteration += 1
        return [world_loss + critic_loss, actor_loss]

    def train(self, state, action, reward, state_next, done):
        return [0.0, 0.0]

    def _train_world_model(self, obs, actions, rewards, dones):
        embeds = self.encoder(obs)
        h_states, z_posts, prior_probs, post_probs = self.rssm.observe_sequence(embeds, actions)
        latents = self._latents_from_sequence(h_states, z_posts)
        flat_latents = latents.reshape(-1, latents.shape[-1])

        decoder_loss = self.decoder.loss(flat_latents, obs.reshape(-1, obs.shape[-1]))
        reward_loss = self.reward_head.loss(flat_latents, rewards.reshape(-1), self.bins)
        continue_loss = self.continue_head.loss(flat_latents, (1.0 - dones).reshape(-1))
        kl_loss = self.rssm.kl_loss(prior_probs, post_probs, self.free_bits)
        total_loss = decoder_loss + reward_loss + continue_loss + self.kl_coef * kl_loss

        self.world_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_head.parameters()) +
            list(self.continue_head.parameters()),
            10.0,
        )
        self.world_optimizer.step()
        return float(total_loss.detach().cpu())

    def _train_actor_critic(self, obs, actions):
        with torch.no_grad():
            embeds = self.encoder(obs)
            h_states, z_posts, _, _ = self.rssm.observe_sequence(embeds, actions)
            h = h_states.reshape(-1, h_states.shape[-1]).detach()
            z = z_posts.reshape(-1, *z_posts.shape[2:]).detach()

        imagined_latents = []
        imagined_entropies = []
        for _ in range(self.horizon):
            latent = self.rssm.get_latent(h, z)
            imagined_latents.append(latent)
            action, _, entropy = self.actor.sample(latent)
            imagined_entropies.append(entropy)
            h, z = self.rssm.imagine_step(h, z, action)

        final_latent = self.rssm.get_latent(h, z)
        imagined_latents.append(final_latent)
        latents = torch.stack(imagined_latents, dim=1)
        flat_latents = latents[:, :-1].reshape(-1, latents.shape[-1])

        rewards = twohot_decode(self.reward_head(flat_latents), self.bins).reshape(-1, self.horizon)
        continues = torch.sigmoid(self.continue_head(flat_latents)).reshape(-1, self.horizon)

        with torch.no_grad():
            values = twohot_decode(self.critic_target(latents.reshape(-1, latents.shape[-1])), self.bins)
            values = values.reshape(-1, self.horizon + 1)

        returns = self._lambda_returns(rewards, values, continues)
        self.return_normalizer.update(returns.detach())

        critic_loss = self.critic.loss(flat_latents.detach(), returns.reshape(-1).detach(), self.bins)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        normalized_returns = self.return_normalizer.normalize(returns)
        entropy = torch.stack(imagined_entropies, dim=1).mean()
        actor_loss = -normalized_returns.mean() - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        return float(actor_loss.detach().cpu()), float(critic_loss.detach().cpu())

    def _lambda_returns(self, rewards, values, continues):
        returns = torch.zeros_like(rewards)
        last = values[:, -1]
        for t in reversed(range(rewards.shape[1])):
            bootstrap = (1.0 - self.lam) * values[:, t + 1] + self.lam * last
            last = rewards[:, t] + self.discount_factor * continues[:, t] * bootstrap
            returns[:, t] = last
        return returns

    def _latents_from_sequence(self, h_states, z_states):
        batch_size, sequence_length = h_states.shape[:2]
        h = h_states.reshape(batch_size * sequence_length, -1)
        z = z_states.reshape(batch_size * sequence_length, self.stoch_size, self.stoch_classes)
        return self.rssm.get_latent(h, z).reshape(batch_size, sequence_length, -1)

    def _save_latest_batch_image(self, batch_states):
        image = batch_states[0, -1].reshape(self.image_size, self.image_size, self.image_channels)
        save_png(BEV_BATCH_IMAGE_PATH, np.clip(image * 255.0, 0, 255).astype(np.uint8))

    def get_model_parameters(self):
        base = super().get_model_parameters()
        dreamer = [
            self.image_size,
            self.image_channels,
            self.sequence_length,
            self.horizon,
            self.embed_size,
            self.deter_size,
            self.stoch_size,
            self.stoch_classes,
            self.num_bins,
            self.free_bits,
            self.kl_coef,
            self.lam,
        ]
        return base + ', dreamerv3: ' + ', '.join(map(str, dreamer))
