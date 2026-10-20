"""Smooth bounded post XY neural correction, preserving original Z/rotations."""
import torch
from torch import nn
from post_residual import PostResidualActorCriticRecurrent
class PostSmoothResidualActorCriticRecurrent(PostResidualActorCriticRecurrent):
 def __init__(self,obs,obs_groups,num_actions,**kwargs):
  super().__init__(obs,obs_groups,num_actions,**kwargs)
  self.position_head=nn.Sequential(nn.Linear(13,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,2));nn.init.zeros_(self.position_head[-1].weight);nn.init.zeros_(self.position_head[-1].bias)
 def residual_from_features(self,f):
  u=self.position_head(f);return .10*u/torch.sqrt(1+u.square().sum(dim=-1,keepdim=True))
