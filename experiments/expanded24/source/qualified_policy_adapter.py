"""Expose a bound policy method so the standard evaluator resets every history."""
import json
from pathlib import Path
import torch
class NativePolicy:
 def __init__(self,boundary,brake=None):
  self.boundary=boundary;self.brake=brake;self.calls=0;self.resets=0;self.done_count=0;self.brake_steps=0
 def act_inference(self,obs):
  self.calls+=1
  return self.boundary.act_inference(obs)
 def reset(self,dones=None):
  self.boundary.reset(dones);self.resets+=1
  n=self.boundary.raw.num_envs;q=torch.ones(n,dtype=torch.bool,device=self.boundary.raw.device) if dones is None else dones.bool().flatten()
  self.done_count+=int(q.sum())
  h=self.boundary.base.memory_a.hidden_state
  if h is not None:assert not h[:,q].any()
  bridge=getattr(self.boundary,'frozen_boundary_history',None)
  if bridge is not None:assert not bridge.h[:,q].any()
  if self.brake is not None:
   assert not self.brake.brake.latched[q].any()
   if hasattr(self.brake,'cache'):self.brake_steps+=int(self.brake.cache[:,7].sum())
 def finalize(self,logdir):
  assert self.calls>0 and self.resets==self.calls+1
  bridge=getattr(self.boundary,'frozen_boundary_history',None)
  if bridge is not None:bridge.verify()
  v=dict(status='passed',policy_calls=self.calls,reset_calls=self.resets,reset_environment_count=self.done_count,bound_method=True,main_GRU_reset_checked=True,frozen_GRU_reset_checked=bridge is not None,fixed_brake_reset_checked=self.brake is not None,brake_selected_steps=self.brake_steps)
  (Path(logdir)/'evaluation_reset_audit.json').write_text(json.dumps(v,indent=2))
