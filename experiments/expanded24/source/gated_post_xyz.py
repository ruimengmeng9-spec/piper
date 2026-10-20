"""Original recurrent post actor plus an inherited learned late XYZ head."""
import torch
from torch.distributions import Normal
from rsl_rl.utils import unpad_trajectories
from late_xyz2 import LateXYZBandActorCriticRecurrent
class GatedPostXYZActorCriticRecurrent(LateXYZBandActorCriticRecurrent):
 def selected(self,x):
  d=-x[...,2]/20;f=x[...,24:27]
  return (d>=.020)&(d<=.050)&(x[...,3:6].norm(dim=-1)<.04)&(f.abs().amax(-1)<=2)&(f.norm(dim=-1)/.02<8)
 def mean_for(self,x,base):
  proposal=torch.cat((self.position(x,base),base[...,3:]),dim=-1)
  return torch.where(self.selected(x)[...,None],proposal,base)
 def act_inference(self,obs):
  x=self.get_actor_obs(obs);z=self.memory_a(self.actor_obs_normalizer(x)).squeeze(0);return self.mean_for(x,self.actor(z))
 def act(self,obs,masks=None,hidden_state=None):
  raw=self.get_actor_obs(obs);z=self.memory_a(self.actor_obs_normalizer(raw),masks,hidden_state).squeeze(0);x=unpad_trajectories(raw,masks).squeeze(0) if masks is not None else raw;mean=self.mean_for(x,self.actor(z));std=self.std.expand_as(mean) if self.noise_std_type=='scalar' else torch.exp(self.log_std).expand_as(mean);self.distribution=Normal(mean,std);return self.distribution.sample()
