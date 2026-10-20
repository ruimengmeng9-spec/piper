"""Conservative learned mean bias; original actor33/6 and GRU retained."""
import torch
from causal_episodic import EpisodicBiasHead,CausalEpisodicActorCriticRecurrent
class RiskGuidedBiasHead(EpisodicBiasHead):
 def __init__(self):
  super().__init__();self.register_buffer('trust_scale',torch.tensor([[.0075,.0075,.010,.005,.005,.005]]))
 def forward(self,x):return self.trust_scale*torch.tanh(self.net((x-self.mean)/self.scale))
class RiskGuidedActorCriticRecurrent(CausalEpisodicActorCriticRecurrent):
 def __init__(self,obs,obs_groups,num_actions,**kwargs):
  super().__init__(obs,obs_groups,num_actions,**kwargs);self.causal_head=RiskGuidedBiasHead()
