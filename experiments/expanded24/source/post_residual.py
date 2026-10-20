"""Bounded post XY neural residual; original Z/rotations preserved exactly."""
import torch
from torch import nn
from gated_post_xyz import GatedPostXYZActorCriticRecurrent
class PostResidualActorCriticRecurrent(GatedPostXYZActorCriticRecurrent):
 def __init__(self,obs,obs_groups,num_actions,**kwargs):
  super().__init__(obs,obs_groups,num_actions,**kwargs)
  self.position_head=nn.Sequential(nn.Linear(13,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,2));nn.init.zeros_(self.position_head[-1].weight);nn.init.zeros_(self.position_head[-1].bias);self.position_mean=torch.zeros(1,13);self.position_std=torch.ones(1,13)
 def residual_features(self,x,b):
  step=b[...,3:].clamp(-1,1)*.012;near=x[...,2:3].abs()/20<=.03;cap=step*torch.minimum(torch.ones_like(step[...,:1]),.001/step.norm(dim=-1,keepdim=True).clamp_min(1e-12));bounded=torch.where(near,cap,step)
  return torch.cat((x[...,:6],bounded/.012,near.to(x.dtype),b[...,:3].clamp(-1,1)),dim=-1)
 def residual_gate(self,x,b):
  d=-x[...,2]/20;f=x[...,24:27]
  return (d>=.020)&(d<=.050)&(x[...,3:6].norm(dim=-1)<.10)&(f.abs().amax(-1)<=2)&(f.norm(dim=-1)/.02<8)&(b[...,2]>-.53)&(b[...,2]<-.10)

 def residual_from_features(self,f):
  r=.10*torch.tanh(self.position_head(f));return r*torch.minimum(torch.ones_like(r[...,:1]),.10/r.norm(dim=-1,keepdim=True).clamp_min(1e-12))
 def correction(self,x,b):return self.residual_from_features((self.residual_features(x,b)-self.position_mean)/self.position_std)
 def mean_for(self,x,b):
  a=torch.cat((b[...,:2]+self.correction(x,b),b[...,2:]),dim=-1)
  return torch.where(self.residual_gate(x,b)[...,None],a,b)
