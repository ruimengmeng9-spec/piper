"""Frozen C81 policy plus a small learned XY residual; external actor33/action6."""
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.utils import unpad_trajectories

class LateXYActorCriticRecurrent(ActorCriticRecurrent):
    def __init__(self, obs, obs_groups, num_actions, xy_adapter_hidden_dims=(64,32), xy_adapter_bound=.25, **kwargs):
        super().__init__(obs,obs_groups,num_actions,**kwargs)
        assert num_actions==6 and self.get_actor_obs(obs).shape[-1]==33 and obs_groups['policy']==['policy']
        assert self.memory_a.rnn.hidden_size==33 and not self.state_dependent_std
        for parameter in self.parameters():parameter.requires_grad_(False)
        self.xy_adapter_bound=float(xy_adapter_bound)
        self.xy_adapter=nn.Sequential(nn.Linear(72,xy_adapter_hidden_dims[0]),nn.ELU(),nn.Linear(xy_adapter_hidden_dims[0],xy_adapter_hidden_dims[1]),nn.ELU(),nn.Linear(xy_adapter_hidden_dims[1],2))
        nn.init.zeros_(self.xy_adapter[-1].weight);nn.init.zeros_(self.xy_adapter[-1].bias)
        self.register_buffer('xy_feature_mean',torch.zeros(1,72));self.register_buffer('xy_feature_std',torch.ones(1,72))

    def adapter_mean(self, normalized_actor33, recurrent33, base_actions6):
        # Every feature derives from the original33 measured inputs/history.
        features=torch.cat((normalized_actor33,recurrent33,base_actions6),dim=-1)
        residual=self.xy_adapter_bound*torch.tanh(self.xy_adapter((features-self.xy_feature_mean)/self.xy_feature_std))
        return torch.cat((base_actions6[...,:2]+residual,base_actions6[...,2:]),dim=-1)

    def act_inference(self,obs):
        x=self.actor_obs_normalizer(self.get_actor_obs(obs))
        z=self.memory_a(x).squeeze(0);base=self.actor(z)
        return self.adapter_mean(x,z,base)

    def act(self,obs,masks=None,hidden_state=None):
        x=self.actor_obs_normalizer(self.get_actor_obs(obs))
        z=self.memory_a(x,masks,hidden_state).squeeze(0)
        features=unpad_trajectories(x,masks).squeeze(0) if masks is not None else x
        mean=self.adapter_mean(features,z,self.actor(z))
        std=self.std.expand_as(mean) if self.noise_std_type=='scalar' else torch.exp(self.log_std).expand_as(mean)
        self.distribution=Normal(mean,std)
        return self.distribution.sample()

    def update_normalization(self,obs):
        # Both the original encoder and the calibrated adapter normalization
        # remain fixed. The dedicated BC/PPO optimizer trains only the adapter.
        pass
