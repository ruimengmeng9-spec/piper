"""Passive initial-stage history: no RNG, policy replay or observation refresh."""
import json,types
from pathlib import Path
import numpy as np
import torch

class Trace:
 def __init__(self,raw,logdir):
  self.raw=raw;self.D=Path(logdir);self.rows=[];self.columns={};self.tick=0
  self.ids=torch.tensor([25,26,29,30,72,73],device=raw.device)
  self.generation=torch.full((raw.num_envs,),-1,device=raw.device,dtype=torch.long)
  self.initial_action=torch.zeros(raw.num_envs,6,device=raw.device);self.late_action=self.initial_action.clone()
  for name,dest in [('initial',self.initial_action),('late_initial',self.late_action)]:
   policy=getattr(raw,name+'_policy')
   def observe(obs,_policy=policy,_dest=dest):
    out=_policy(obs);_dest.copy_(out);return out
   setattr(raw,name+'_policy',observe)
  pre=raw._pre_physics_step;reset=raw._reset_idx
  def before(this,actions):
   command=this.command_pos.clone();pre(actions)
   s=this.supervisor;x=this.r2_last_actor;f=x[:,24:27]
   late=(s.mode==0)&(s.exits==0)&~this.step_preparing&(-x[:,2]/20>=.020)&(-x[:,2]/20<=.050)&(x[:,3:6].norm(dim=-1)<.04)&(f.abs().amax(-1)<=2)&(f.norm(dim=-1)/.02<8)
   fields=dict(tick=torch.full_like(s.mode,self.tick),env=torch.arange(this.num_envs,device=this.device),generation=self.generation,
    step=this.episode_controller_steps,mode=s.mode,attempts=s.attempts,exits=s.exits,late_selected=late,
    actor33=x,initial_action=self.initial_action,late_action=self.late_action,applied=this.r2_last_action,
    delayed_action=this.action_delay.buffer[this.action_delay.steps,this.action_delay.ids],delay=this.action_delay.steps,
    command_before=command,command_after=this.command_pos,truth9=torch.cat(this._geometry(),-1),
    q=this.robot.data.joint_pos[:,this.arm_ids],qd=this.robot.data.joint_vel[:,this.arm_ids],
    preparing=this.step_preparing)
   parts=[];offset=0
   for n,v in fields.items():
    v=v.reshape(this.num_envs,-1);self.columns[n]=[offset,offset+v.shape[1]];offset+=v.shape[1];parts.append(v[self.ids].float())
   self.rows.append(torch.cat(parts,-1).cpu().numpy());self.tick+=1
  def on_reset(this,ids):
   if ids is None:ids=this.robot._ALL_INDICES
   reset(ids);self.generation[ids]+=1
  raw._pre_physics_step=types.MethodType(before,raw);raw._reset_idx=types.MethodType(on_reset,raw)
 def finalize(self):
  data=np.concatenate(self.rows);out=self.D/'initial_passive_trace.npz';np.savez_compressed(out,trace=data)
  assert out.stat().st_size<32*1024**2
  (self.D/'initial_passive_trace.json').write_text(json.dumps(dict(columns=self.columns,envs=self.ids.cpu().tolist(),ticks=self.tick,rows=len(data),bytes=out.stat().st_size,no_rng_calls=True,no_observation_refresh=True,no_control_change=True),indent=2))
