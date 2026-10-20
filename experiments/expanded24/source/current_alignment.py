"""Current-observation neural specialist for the near-alignment domain."""
import torch
from current_recovery import CurrentObsRecoveryActorCriticRecurrent
class CurrentObsAlignmentActorCriticRecurrent(CurrentObsRecoveryActorCriticRecurrent):
 def __init__(self,obs,obs_groups,num_actions,**kwargs):
  super().__init__(obs,obs_groups,num_actions,**kwargs)
  with torch.no_grad():self.current_actor_mask.zero_();self.current_actor_mask[:,[0,1,3,4,5]]=1
