"""Preserve the qualified residual's feature map while its post skill learns."""
import copy,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from boundary_rl import model_hash

def precision():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')

def sequence_mean(model,x,h0):
 """Complete native observation histories, with no cached GRU features as input."""
 y,_=model.memory_a.rnn(x,h0)
 return model.actor(y.reshape(-1,y.shape[-1])).reshape(*y.shape[:2],6)

class FrozenBoundaryHistory:
 def __init__(self,raw,boundary):
  assert not boundary.stochastic and not hasattr(boundary,'frozen_boundary_history')
  self.raw=raw;self.boundary=boundary;self.model=boundary.base
  assert isinstance(self.model.memory_a.rnn,nn.GRU) and self.model.memory_a.rnn.num_layers==1 and self.model.memory_a.rnn.dropout==0
  self.rnn=copy.deepcopy(self.model.memory_a.rnn).eval();self.rnn.flatten_parameters();self.actor=copy.deepcopy(self.model.actor).eval()
  self.source=Path('/workspace/piper-insertion/logs/specialist_stage_20261015_postrnn1_fit/fitted_recurrent_post_actor.pt');self.source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest()
  assert self.source_sha256=='d092d8c3dbb5a440d1f90f0e0c005bf1b754a5d3d7107234f7a50a13812abf1f'
  state=torch.load(self.source,map_location=raw.device,weights_only=False)['model_state_dict']
  self.rnn.load_state_dict({k[len('memory_a.rnn.'):]:v for k,v in state.items() if k.startswith('memory_a.rnn.')});self.actor.load_state_dict({k[len('actor.'):]:v for k,v in state.items() if k.startswith('actor.')})
  for k,v in self.model.state_dict().items():
   if k.startswith('actor_obs_normalizer.'):assert torch.equal(v,state[k]),k
  for p in [*self.rnn.parameters(),*self.actor.parameters()]:p.requires_grad_(False)
  self.h=torch.zeros(1,raw.num_envs,33,device=raw.device);self.hashes=(model_hash(self.rnn),model_hash(self.actor));act=boundary.act_inference;reset=boundary.reset
  def inference(obs):
   assert not self.model.actor_obs_normalizer.training
   self.x=self.model.actor_obs_normalizer(self.model.get_actor_obs(obs)).detach().clone()
   h=self.model.memory_a.hidden_state;self.actor_h0=torch.zeros_like(self.h) if h is None else h.detach().clone()
   with torch.no_grad():
    y,self.h=self.rnn(self.x[None],self.h);self.anchor=self.actor(y[0]).detach().clone()
   out=act(obs)
   boundary.features=torch.cat((self.x.clamp(-8,8),self.h[0]),-1).detach()
   boundary.mu=boundary.model.mean(boundary.features);boundary.latent=boundary.mu
   boundary.logp=Normal(boundary.mu,boundary.model.std).log_prob(boundary.latent)
   boundary.delta=boundary.model.scale*torch.tanh(boundary.latent)
   return out
  def sync_reset(dones=None):
   reset(dones)
   if dones is None:self.h.zero_()
   else:self.h[:,dones.bool().flatten()]=0
  boundary.act_inference=inference;boundary.reset=sync_reset;boundary.frozen_boundary_history=self
 def verify(self):assert (model_hash(self.rnn),model_hash(self.actor))==self.hashes

class RecurrentHook:
 def __init__(self,raw,boundary,bridge):
  self.active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool);self.sigma=torch.tensor([.01,.01,.01,.0075,.0075,.0075],device=raw.device);original=boundary.compose
  def compose(r,base,first):
   mask=r.training_student_control&self.active&~first;mean=boundary.last_base.detach().clone()
   assert not (mask&r.preparing).any() and not (mask&(r.supervisor.mode!=0)).any()
   if mask.any():assert torch.equal(base[mask],mean[mask].clamp(-1,1))
   latent=mean.clone()
   if mask.any():latent[mask]=Normal(mean[mask],self.sigma).sample()
   lp=Normal(mean,self.sigma).log_prob(latent).sum(-1);mixed=base.clone();mixed[mask]=latent[mask].clamp(-1,1)
   out=original(r,mixed,first);assert torch.equal(out[~mask],(base+torch.where(boundary.mask,boundary.delta,0.))[~mask])
   self.cache=(bridge.x.clone(),latent.detach(),lp.detach(),mask.clone(),mean,bridge.anchor.clone())
   return out
  boundary.compose=compose
