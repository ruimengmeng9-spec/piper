"""Episode-constant action latency in policy steps; buffers clear on reset."""
import torch


class ActionDelay:
    def __init__(self, num_envs, action_dim, maximum, device):
        self.buffer = torch.zeros(maximum + 1, num_envs, action_dim, device=device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ids = torch.arange(num_envs, device=device)

    def reset(self, env_ids, minimum, maximum):
        self.buffer[:, env_ids] = 0.
        self.steps[env_ids] = torch.randint(minimum, maximum + 1,
                                           (len(env_ids),), device=self.buffer.device)

    def push(self, actions):
        self.buffer[1:] = self.buffer[:-1].clone()
        self.buffer[0] = actions
        return self.buffer[self.steps, self.ids].clone()
