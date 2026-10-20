"""Passive contact/withdrawal/realignment records; no RNG or observation refresh."""
import json,types
from pathlib import Path
import numpy as np
import torch

class HandoffTrace:
 def __init__(self,raw,logdir):
  self.r=raw;self.D=Path(logdir);self.rows=[];self.columns={};self.tick=0
  self.ids=torch.tensor([120,123,124,125,126,127],device=raw.device)
  self.generation=torch.full((raw.num_envs,),-1,device=raw.device,dtype=torch.long);self.outputs={}
  raw.diagnostic_telemetry=True;raw.axial_trace_enabled=True
  for name in ['initial','late_initial','recovery','alignment','near_alignment']:
   policy=getattr(raw,name+'_policy');self.outputs[name]=torch.zeros(raw.num_envs,6,device=raw.device)
   def observe(obs,_policy=policy,_name=name):
    out=_policy(obs);self.outputs[_name].copy_(out);return out
   setattr(raw,name+'_policy',observe)
  pre=raw._pre_physics_step;reset=raw._reset_idx
  def before(this,actions):
   command=this.command_pos.clone();mode=this.supervisor.mode.clone();near=this.near_alignment_control_steps.clone();pre(actions)
   s=this.supervisor;x=this.r2_last_actor;b=this.boundary_controller
   q=this.robot.data.joint_pos[:,this.arm_ids];limits=this.robot.data.soft_joint_pos_limits[:,this.arm_ids]
   fields=dict(tick=torch.full_like(s.mode,self.tick),env=torch.arange(this.num_envs,device=this.device),generation=self.generation,
    step=this.episode_controller_steps,old_mode=mode,mode=s.mode,attempts=s.attempts,exits=s.exits,realignments=s.realignments,
    near_selected=this.near_alignment_control_steps-near,actor33=x,critic_features=(this.critic_shadow.feature if hasattr(this,'critic_shadow') else torch.zeros(this.num_envs,33,device=this.device)),critic_value=(this.critic_shadow.value if hasattr(this,'critic_shadow') else torch.zeros(this.num_envs,device=this.device)),post_features=b.features[:,33:],post_raw_action=b.last_base,applied=this.r2_last_action,boundary_delta=b.delta,boundary_mask=b.mask,
    delayed_action=this.action_delay.buffer[this.action_delay.steps,this.action_delay.ids],delay=this.action_delay.steps,
    command_before=command,command_after=this.command_pos,axial_integrated=this.axial_after_integrate,axial_before_stage=this.axial_before_stage,flange=this.axial_flange,flange_quat=this.axial_flange_quat,target_quat=this.axial_target_quat,hole_quat=this.hole_quat,delta_world=this.axial_command,truth9=torch.cat(this._geometry(),-1),filtered_force=s.force_ema,
    release_hold=s.release_hold,align_hold=s.align_hold,trigger_depth=s.trigger_depth,
    q=q,qd=this.robot.data.joint_vel[:,this.arm_ids],joint_margin=torch.minimum(q-limits[:,:,0],limits[:,:,1]-q),
    ik_target=this.diag_ik_target,jacobian=this.diag_jacobian,ik_pos=this.diag_ik_pos,ik_quat=this.diag_ik_quat,ik_command_pos=this.diag_ik_command_pos,ik_command_quat=this.diag_ik_command_quat,tool_offset=this.diag_tool_offset,root_quat=this.diag_root_quat,joint_limit_clip=this.diag_joint_limit_clip,joint_step_clip=this.diag_joint_step_clip,
    jacobian_sv=this.diag_jacobian_singular_values,preparing=this.step_preparing,**{k+'_action':v for k,v in self.outputs.items()})
   parts=[];offset=0
   for n,v in fields.items():
    v=v.reshape(this.num_envs,-1);self.columns[n]=[offset,offset+v.shape[1]];offset+=v.shape[1];parts.append(v[self.ids].float())
   self.rows.append(torch.cat(parts,-1).cpu().numpy());self.tick+=1
  def on_reset(this,ids):
   if ids is None:ids=this.robot._ALL_INDICES
   reset(ids);self.generation[ids]+=1
  raw._pre_physics_step=types.MethodType(before,raw);raw._reset_idx=types.MethodType(on_reset,raw)
 def finalize(self):
  data=np.concatenate(self.rows);out=self.D/'handoff_passive_trace.npz';np.savez_compressed(out,trace=data);assert out.stat().st_size<32*1024**2
  (self.D/'handoff_passive_trace.json').write_text(json.dumps(dict(columns=self.columns,envs=self.ids.cpu().tolist(),ticks=self.tick,rows=len(data),bytes=out.stat().st_size,no_rng_calls=True,no_observation_refresh=True,no_control_change=True,phase='after native command composition,before physics'),indent=2))
