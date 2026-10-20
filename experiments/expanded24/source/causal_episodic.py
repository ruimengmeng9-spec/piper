"""Public33 recurrent actor plus a neural bias from pre-intervention context."""
import torch
from torch.distributions import Normal
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.utils import unpad_trajectories

class EpisodicBiasHead(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(18,64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,6));self.register_buffer('mean',torch.zeros(1,18));self.register_buffer('scale',torch.ones(1,18));torch.nn.init.zeros_(self.net[4].weight);torch.nn.init.zeros_(self.net[4].bias)
 def forward(self,x):
  u=self.net((x-self.mean)/self.scale);p=u[...,:3];r=u[...,3:];return torch.cat((.04*p/torch.sqrt(1+p.square().sum(-1,keepdim=True)),.025*r/torch.sqrt(1+r.square().sum(-1,keepdim=True))),dim=-1)

class CausalEpisodicActorCriticRecurrent(ActorCriticRecurrent):
 def __init__(self,obs,obs_groups,num_actions,**kwargs):
  super().__init__(obs,obs_groups,num_actions,**kwargs);assert num_actions==6 and self.get_actor_obs(obs).shape[-1]==33 and obs_groups['policy']==['policy'] and not self.state_dependent_std
  assert isinstance(self.memory_a.rnn,torch.nn.GRU) and self.memory_a.rnn.hidden_size==33 and self.memory_a.rnn.num_layers==1
  for p in self.parameters():p.requires_grad_(False)
  self.causal_head=EpisodicBiasHead();self.register_buffer('public_context_indices',torch.tensor(list(range(12))+list(range(27,33)),dtype=torch.long));self.context=None;self.context_valid=None
 def public_domain(self,x,b):
  f=x[...,24:27];d=-x[...,2]/20
  return (d>=-.015)&(d<=.05)&(x[...,3:6].norm(dim=-1)<.10)&(f.abs().amax(-1)<=2)&(f.norm(dim=-1)/.02<8)&(b[...,2]>-.53)&(b[...,2]<-.10)
 def _mean_inference(self,obs):
  x=self.get_actor_obs(obs);cold=self.memory_a.hidden_state is None;xn=self.actor_obs_normalizer(x);z=self.memory_a(xn).squeeze(0);b=self.actor(z);g=self.public_domain(x,b)
  if cold or self.context is None or len(self.context)!=len(x):self.context=torch.zeros(len(x),18,device=x.device,dtype=x.dtype);self.context_valid=torch.zeros(len(x),dtype=torch.bool,device=x.device)
  first=g&~self.context_valid;self.context=torch.where(first[:,None],x[...,self.public_context_indices],self.context);self.context_valid|=first
  return torch.where(g[:,None],b+self.causal_head(self.context),b)
 def act_inference(self,obs):return self._mean_inference(obs)
 def act(self,obs,masks=None,hidden_state=None):
  if masks is None:mean=self._mean_inference(obs)
  else:
   x=self.get_actor_obs(obs);assert x.ndim==3
   if isinstance(hidden_state,(tuple,list)):assert len(hidden_state)==1;hidden_state=hidden_state[0]
   if hidden_state is None:hidden_state=torch.zeros(1,x.shape[1],52,device=x.device,dtype=x.dtype)
   assert hidden_state.shape==(1,x.shape[1],52)
   z,_=self.memory_a.rnn(self.actor_obs_normalizer(x),hidden_state[...,:33]);b=self.actor(z);g=self.public_domain(x,b);context=hidden_state[0,:,33:51];valid=hidden_state[0,:,51]>.5;hist=[]
   for t in range(len(x)):
    first=g[t]&~valid&masks[t];context=torch.where(first[:,None],x[t,...,self.public_context_indices],context);valid=valid|first;hist.append(context)
   context=torch.stack(hist);mean=torch.where(g[...,None],b+self.causal_head(context),b);mean=unpad_trajectories(mean,masks).squeeze(0)
  std=self.std.expand_as(mean) if self.noise_std_type=='scalar' else torch.exp(self.log_std).expand_as(mean);self.distribution=Normal(mean,std);return self.distribution.sample()
 def get_hidden_states(self):
  h=self.memory_a.hidden_state
  if h is None:return None,self.memory_c.hidden_state
  assert self.context is not None and self.context_valid is not None
  return torch.cat((h,self.context[None],self.context_valid.to(h.dtype)[None,:,None]),dim=-1),self.memory_c.hidden_state
 def reset(self,dones=None):
  super().reset(dones)
  if dones is None:self.context=None;self.context_valid=None
  elif self.context is not None:
   q=dones.reshape(-1).bool()
   if len(q)!=len(self.context):self.context=None;self.context_valid=None
   else:self.context[q]=0.;self.context_valid[q]=False
 def update_normalization(self,obs):pass
