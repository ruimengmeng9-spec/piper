"""Unadopted diagnostic candidate: abandon a blocked final retry using measured history only."""
import torch
class FinalRetryAxialBrake:
 def __init__(self,n,device,release_on=False):
  self.release_on=bool(release_on)
  self.latched=torch.zeros(n,device=device,dtype=torch.bool)
  self.intervention_steps=torch.zeros(n,device=device,dtype=torch.long)
 def reset(self,done=None):
  if done is None:self.latched.zero_();self.intervention_steps.zero_()
  else:
   q=done.bool().flatten();self.latched[q]=False;self.intervention_steps[q]=0
 def apply(self,public33,actions,mode,attempts,exits,blocked_streak,preparing,just_exited):
  assert public33.shape[-1]==33 and actions.shape[-1]==6
  eligible=(mode==0)&(attempts>=3)&(exits>0)&~preparing&~just_exited
  # blocked_streak is the unchanged supervisor's measured-force/estimated-depth
  # counter. No true geometry, true force, material or outcome labels are read.
  self.latched|=eligible&(blocked_streak>=2)
  selected=eligible&self.latched
  # Positive hole-frame Z unloads the existing accumulated command. This latch
  # stays set until native episode reset: no fourth insertion retry is created.
  downward_lead=-public33[:,29]/20
  floor=torch.where(downward_lead>.001,torch.full_like(downward_lead,.35),torch.zeros_like(downward_lead))
  if self.release_on:
   measured=public33[:,24:27];valid=measured.abs().amax(-1)<=2.
   force=torch.linalg.vector_norm(measured/.02,dim=-1)
   floor=torch.where((downward_lead>.001)|(valid&(force>=5.)),torch.full_like(downward_lead,.35),torch.zeros_like(downward_lead))
  out=actions.clone();out[:,2]=torch.where(selected,torch.maximum(out[:,2],floor),out[:,2])
  self.intervention_steps+=selected.long()
  if self.release_on:
   for k in [0,1,3,4,5]:out[:,k]=torch.where(selected,torch.zeros_like(out[:,k]),out[:,k])
   assert not out[selected][:,[0,1,3,4,5]].any()
  else:assert torch.equal(out[:,[0,1,3,4,5]],actions[:,[0,1,3,4,5]])
  assert torch.equal(out[~selected],actions[~selected])
  assert (out[selected,2]>=0).all()
  return out,selected
