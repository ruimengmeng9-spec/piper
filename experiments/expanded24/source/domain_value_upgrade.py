"""Critic-only conditioning on existing normalized friction and stiffness."""
import torch
from torch import nn

def domain_quadratic_inputs(domain):
 assert domain.shape[-1]==2 and torch.isfinite(domain).all()
 f,s=domain.unbind(-1)
 return torch.stack((f.square(),f*s,s.square()),-1)

class DomainLinear(nn.Module):
 def __init__(self,public):
  super().__init__();assert isinstance(public,nn.Linear) and public.in_features==73
  self.public=public;devices=[public.weight.device.index] if public.weight.is_cuda else []
  with torch.random.fork_rng(devices=devices):
   self.domain=nn.Linear(2,public.out_features,bias=False,device=public.weight.device,dtype=public.weight.dtype)
   self.quadratic=nn.Linear(3,public.out_features,bias=False,device=public.weight.device,dtype=public.weight.dtype)
   with torch.no_grad():self.domain.weight.zero_();self.quadratic.weight.zero_()
  self.train(public.training)
 def forward(self,x):
  assert x.shape[-1]==75
  original=self.public(x[...,:73].contiguous())+self.domain(x[...,73:])
  return original+self.quadratic(domain_quadratic_inputs(x[...,73:]))

def upgrade_critic(model):
 assert isinstance(model.critic[0],nn.Linear) and model.critic[0].in_features==73
 model.critic[0]=DomainLinear(model.critic[0])

def append_domain_context(context,friction,stiffness):
 assert context.shape[-1]==73
 f=friction.to(device=context.device,dtype=context.dtype).flatten();s=stiffness.to(device=context.device,dtype=context.dtype).flatten()
 domain=torch.stack(((f-1.)/.4,(s-1.)/.2),-1)
 if context.ndim==3:domain=domain[None].expand(context.shape[0],-1,-1)
 assert domain.shape[:-1]==context.shape[:-1]
 return torch.cat((context,domain),-1)
