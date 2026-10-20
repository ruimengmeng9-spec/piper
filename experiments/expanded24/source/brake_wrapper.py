"""Fixed validated measured-history brake after the unchanged latent policy transform."""
import torch
from retry_brake import FinalRetryAxialBrake
class BrakeHook:
 columns=['mode','attempts','exits','blocked_streak','preparing','just_exited','active','selected','latched','lead_m']+['before_'+str(i) for i in range(6)]+['after_'+str(i) for i in range(6)]
 def __init__(self,raw,boundary,policy_hook=None,release_on=False):
  self.brake=FinalRetryAxialBrake(raw.num_envs,raw.device,release_on=release_on);original=boundary.compose;reset=boundary.reset
  def compose(r,base,first):
   before=original(r,base,first);active=policy_hook.active if policy_hook is not None else torch.ones(r.num_envs,device=r.device,dtype=torch.bool);s=r.supervisor;x=boundary.public
   after,selected=self.brake.apply(x,before,s.mode,s.attempts,s.exits,s.streak,r.preparing|~active,first)
   fields=[s.mode,s.attempts,s.exits,s.streak,r.preparing,first,active,selected,self.brake.latched,-x[:,29]/20]
   self.cache=torch.cat([torch.stack(fields,-1).float(),before,after],-1).detach().clone()
   return after
  def sync_reset(done=None):reset(done);self.brake.reset(done)
  boundary.compose=compose;boundary.reset=sync_reset
