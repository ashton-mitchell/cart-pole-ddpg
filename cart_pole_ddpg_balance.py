import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class DMCWrapper:

    def __init__(self, domain="cartpole", task="balance", seed=0):
        self._domain = domain
        self._task = task
        self._seed = seed
        self._build_env()

    def _build_env(self):
        self.env = suite.load(
            domain_name=self._domain,
            task_name=self._task,
            task_kwargs={"random": self._seed}
        )
        ts = self.env.reset()
        self._obs_dim = self._flatten_obs(ts).shape[0]
        spec = self.env.action_spec()
        self._act_dim = spec.shape[0]
        self._act_low = spec.minimum
        self._act_high = spec.maximum
        self._act_limit = float(spec.maximum[0])

    def reset(self):
        ts = self.env.reset()
        return self._flatten_obs(ts)

    def step(self, action):
        action = np.clip(action, self._act_low, self._act_high)
        ts = self.env.step(action)
        obs = self._flatten_obs(ts)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        done = ts.last()
        return obs, reward, done, {}

    def _flatten_obs(self, ts):
        obs_parts = []
        for v in ts.observation.values():
            obs_parts.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(obs_parts, axis=0).astype(np.float32)

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def act_dim(self):
        return self._act_dim

    @property
    def act_limit(self):
        return self._act_limit


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}


def mlp(in_dim, out_dim, hidden_sizes=(256, 256), activation=nn.ReLU, output_activation=None):
    layers = []
    last_dim = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last_dim, h))
        layers.append(activation())
        last_dim = h
    layers.append(nn.Linear(last_dim, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden_sizes=(256, 256))
        self.act_limit = act_limit

    def forward(self, obs):
        x = self.net(obs)
        return torch.tanh(x) * self.act_limit


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, 1, hidden_sizes=(256, 256))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q_value = self.q(x)
        return torch.squeeze(q_value, -1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update_params(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.mul_(1.0 - tau)
        tgt_param.data.add_(tau * src_param.data)


def train_ddpg(
    seed,
    num_episodes=200,
    max_ep_len=500,
    start_steps=1000,
    batch_size=128,
    replay_size=100000,
    gamma=0.99,
    tau=0.005,
    actor_lr=1e-3,
    critic_lr=1e-3,
    expl_noise=0.1,
):
    set_seed(seed)
    env = DMCWrapper(domain="cartpole", task="balance", seed=seed)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    act_limit = env.act_limit

    actor = Actor(obs_dim, act_dim, act_limit).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    actor_target = Actor(obs_dim, act_dim, act_limit).to(device)
    critic_target = Critic(obs_dim, act_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)

    total_steps = 0
    episode_returns = []

    for ep in range(num_episodes):
        obs = env.reset()
        ep_return = 0.0

        for t in range(max_ep_len):
            total_steps += 1

            if total_steps < start_steps:
                act = np.random.uniform(-act_limit, act_limit, size=act_dim)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    act_t = actor(obs_t).cpu().numpy()[0]
                noise = expl_noise * act_limit * np.random.randn(act_dim)
                act = np.clip(act_t + noise, -act_limit, act_limit)

            next_obs, reward, done, _ = env.step(act)
            replay_buffer.store(obs, act, reward, next_obs, float(done))
            ep_return += reward
            obs = next_obs

            if replay_buffer.size >= batch_size:
                batch = replay_buffer.sample_batch(batch_size)
                with torch.no_grad():
                    next_actions = actor_target(batch["next_obs"])
                    target_q = critic_target(batch["next_obs"], next_actions)
                    backup = batch["rews"] + gamma * (1.0 - batch["done"]) * target_q

                current_q = critic(batch["obs"], batch["acts"])
                critic_loss = ((current_q - backup) ** 2).mean()
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                actor_loss = -critic(batch["obs"], actor(batch["obs"])).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                soft_update_params(actor, actor_target, tau)
                soft_update_params(critic, critic_target, tau)

            if done:
                break

        episode_returns.append(ep_return)
        print(f"[Seed {seed}] Episode {ep+1}/{num_episodes}, Return: {ep_return:.1f}")

    return actor, episode_returns


def evaluate_policy(actor, seed=10, num_episodes=10, max_ep_len=500):
    env = DMCWrapper(domain="cartpole", task="balance", seed=seed)
    all_returns = []
    for ep in range(num_episodes):
        obs = env.reset()
        ep_return = 0.0
        for t in range(max_ep_len):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act = actor(obs_t).cpu().numpy()[0]
            obs, reward, done, _ = env.step(act)
            ep_return += reward
            if done:
                break
        all_returns.append(ep_return)
        print(f"[Eval seed {seed}] Episode {ep+1}/{num_episodes}, Return: {ep_return:.1f}")
    return np.array(all_returns)


train_seeds = [0, 1, 2]
num_episodes = 200

all_train_returns = []
final_actors = []

for s in train_seeds:
    actor, ep_returns = train_ddpg(seed=s, num_episodes=num_episodes)
    final_actors.append(actor)
    all_train_returns.append(ep_returns)

all_train_returns = np.array(all_train_returns)

final_returns = all_train_returns[:, -1]
best_idx = int(np.argmax(final_returns))
best_actor = final_actors[best_idx]
eval_returns = evaluate_policy(best_actor, seed=10, num_episodes=10)

print("\nTraining returns (per seed, last episode):", final_returns)
print("Evaluation mean return (seed 10):", eval_returns.mean(), "+/-", eval_returns.std())

mean_train = all_train_returns.mean(axis=0)
std_train = all_train_returns.std(axis=0)

episodes = np.arange(1, num_episodes + 1)

plt.figure(figsize=(8, 5))
plt.plot(episodes, mean_train, label="Train (mean reward)")
plt.fill_between(
    episodes,
    mean_train - std_train,
    mean_train + std_train,
    alpha=0.3,
    label="Train (±1 std)",
)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DDPG on DMC Cartpole/Balance (3 training seeds)")
plt.legend()
plt.grid(True)
plt.show()