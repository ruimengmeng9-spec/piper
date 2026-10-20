"""Learned late position head with original recurrent rotation policy frozen."""
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.utils import unpad_trajectories
class LateXYZActorCriticRecurrent(ActorCriticRecurrent):
 def __init__(self,obs,obs_groups,num_actions,**kwargs):
  super().__init__(obs,obs_groups,num_actions,**kwargs)
  assert num_actions==6 and self.get_actor_obs(obs).shape[-1]==33 and obs_groups['policy']==['policy'] and not self.state_dependent_std
  for p in self.parameters():p.requires_grad_(False)
  self.position_head=nn.Sequential(nn.Linear(9,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,3));self.register_buffer('position_mean',torch.zeros(1,9));self.register_buffer('position_std',torch.ones(1,9))
 def position(self,xraw,base):
  f=torch.cat((xraw[...,:6],base[...,3:].clamp(-1,1)),dim=-1)
  return .25*torch.tanh(self.position_head((f-self.position_mean)/self.position_std))
 def act_inference(self,obs):
  xraw=self.get_actor_obs(obs);z=self.memory_a(self.actor_obs_normalizer(xraw)).squeeze(0);base=self.actor(z)
  return torch.cat((self.position(xraw,base),base[...,3:]),dim=-1)
 def act(self,obs,masks=None,hidden_state=None):
  raw=self.get_actor_obs(obs);z=self.memory_a(self.actor_obs_normalizer(raw),masks,hidden_state).squeeze(0);x=unpad_trajectories(raw,masks).squeeze(0) if masks is not None else raw;base=self.actor(z);mean=torch.cat((self.position(x,base),base[...,3:]),dim=-1);std=self.std.expand_as(mean) if self.noise_std_type=='scalar' else torch.exp(self.log_std).expand_as(mean);self.distribution=Normal(mean,std);return self.distribution.sample()
 def update_normalization(self,obs):pass
