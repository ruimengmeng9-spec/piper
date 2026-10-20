"""Keep the controller context73 unchanged; extend training-value context only."""
import torch
from domain_value_upgrade import append_domain_context

class ConditionedShadow:
 def __init__(self,raw,boundary):
  self.model=boundary.model;self.feature=torch.zeros(raw.num_envs,75,device=raw.device);self.value=torch.zeros(raw.num_envs,device=raw.device)
  friction=raw.friction_scales.to(raw.device).detach().clone();stiffness=raw.stiffness_scales.to(raw.device).detach().clone();original=boundary.compose
  def compose(r,base,first):
   action=original(r,base,first);self.feature=append_domain_context(boundary.context.detach(),friction,stiffness).detach().clone()
   with torch.no_grad():self.value=(self.model.critic(self.feature)[:,0]*100.).detach().clone()
   return action
  boundary.compose=compose;raw.critic_shadow=self
