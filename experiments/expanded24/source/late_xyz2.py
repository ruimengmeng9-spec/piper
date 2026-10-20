"""Learned position head with an observed native30mm band feature."""
import torch
from torch import nn
from late_xyz import LateXYZActorCriticRecurrent
class LateXYZBandActorCriticRecurrent(LateXYZActorCriticRecurrent):
 def __init__(self,obs,obs_groups,num_actions,**kwargs):
  super().__init__(obs,obs_groups,num_actions,**kwargs)
  self.position_head=nn.Sequential(nn.Linear(10,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,3));self.position_mean=torch.zeros(1,10);self.position_std=torch.ones(1,10)
 def position(self,xraw,base):
  band=(xraw[...,2:3].abs()/20<=.03).to(xraw.dtype);f=torch.cat((xraw[...,:6],base[...,3:].clamp(-1,1),band),dim=-1)
  return .25*torch.tanh(self.position_head((f-self.position_mean)/self.position_std))
