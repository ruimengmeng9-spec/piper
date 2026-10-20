"""Passive full-batch hashes and selected deep-withdrawal telemetry. No RNG."""
import hashlib,json,types
from pathlib import Path
import numpy as np
import torch
from boundary_rl import model_hash

class Probe:
 def __init__(self,raw,boundary,logdir,detail):
  self.raw=raw;self.b=boundary;self.D=Path(logdir);self.detail=detail;self.tick=0;self.rows=[];self.columns={};self.resets=0;self.physical=[hashlib.sha256() for _ in range(raw.num_envs)];self.features=[hashlib.sha256() for _ in range(raw.num_envs)];self.models={n:model_hash(m) for n,m in boundary.models().items()};self.bhash=model_hash(boundary.model)
  model=raw.deep_recovery_model;assert model is not None and not model.actor_obs_normalizer.training
  self.deep=torch.zeros(raw.num_envs,6,device=raw.device);self.x=torch.zeros(raw.num_envs,33,device=raw.device);self.z=self.x.clone()
  if detail:
   original=raw.deep_recovery_policy
   def deep(obs):
    out=original(obs)
    self.deep.copy_(out);self.x.copy_(model.actor_obs_normalizer(model.get_actor_obs(obs)));self.z.copy_(model.current_features(self.x))
    return out
   raw.deep_recovery_policy=deep
  original_pre=raw._pre_physics_step;original_reset=raw._reset_idx
  def pre(this,actions):
   original_pre(actions);s=this.supervisor;truth=torch.cat(this._geometry(),-1)
   fields=[this.r2_last_actor,truth,this.r2_last_action,this.actions,this.action_delay.buffer[this.action_delay.steps,this.action_delay.ids],this.command_pos,this.command_quat,this.targets,this.robot.data.joint_pos[:,this.arm_ids],this.robot.data.joint_vel[:,this.arm_ids],s.mode,s.attempts,s.exits,s.realignments,s.streak,this.preparing,this.deep_recovery_latch]
   physical=torch.cat([v.reshape(this.num_envs,-1).float() for v in fields],-1).cpu().numpy();feature=torch.cat([boundary.features,boundary.delta,boundary.base.memory_a.hidden_state[0],this.deep_recovery_model.memory_a.hidden_state[0]],-1).cpu().numpy()
   for i in range(this.num_envs):self.physical[i].update(physical[i].tobytes());self.features[i].update(feature[i].tobytes())
   if self.detail:
    select=this.deep_recovery_latch&this.recovery_enabled&~this.preparing
    assert not (select&(s.mode!=1)).any()
    fields=dict(tick=torch.full_like(s.mode,self.tick),env=torch.arange(this.num_envs,device=this.device),step=this.episode_controller_steps,mode=s.mode,attempts=s.attempts,exits=s.exits,selected=select,public33=this.r2_last_actor,normalized33=self.x,masked33=self.z,deep_raw6=self.deep,composed6=this.r2_last_action,clipped6=this.actions,delayed6=this.action_delay.buffer[this.action_delay.steps,this.action_delay.ids],boundary_delta=boundary.delta,boundary_mask=boundary.mask,truth9=truth,filtered_force=s.force_ema,command_before_stage=this.axial_before_stage,command_position=this.command_pos,command_quat=this.command_quat,ee_position=this.axial_flange,ee_quat=this.axial_flange_quat,integrated_command=this.axial_after_integrate,delta_world=this.axial_command,ik_target=this.diag_ik_target,ik_pos=this.diag_ik_pos,ik_quat=this.diag_ik_quat,joint_limit_clip=this.diag_joint_limit_clip,joint_step_clip=this.diag_joint_step_clip,jacobian_sv=this.diag_jacobian_singular_values,q=this.robot.data.joint_pos[:,this.arm_ids],qd=this.robot.data.joint_vel[:,this.arm_ids])
    parts=[];offset=0
    for n,v in fields.items():
     v=v.reshape(this.num_envs,-1);self.columns[n]=[offset,offset+v.shape[1]];offset+=v.shape[1];parts.append(v.float())
    # Every selected deep-withdrawal step, all environments, including failures.
    if select.any():self.rows.append(torch.cat(parts,-1)[select].cpu().numpy())
   self.tick+=1
  def reset(this,ids):
   original_reset(ids);ids=this.robot._ALL_INDICES if ids is None else ids;self.resets+=len(ids)
   h=this.deep_recovery_model.memory_a.hidden_state
   if h is not None:assert not h[:,ids].any()
   assert not this.deep_recovery_latch[ids].any()
  raw._pre_physics_step=types.MethodType(pre,raw);raw._reset_idx=types.MethodType(reset,raw)
 def finalize(self):
  assert self.models=={n:model_hash(m) for n,m in self.b.models().items()} and self.bhash==model_hash(self.b.model)
  if self.detail:np.savez_compressed(self.D/'deep_withdrawal_forbidden_training.npz',rows=np.concatenate(self.rows) if self.rows else np.zeros((0,max((v[1] for v in self.columns.values()),default=0)),dtype=np.float32))
  v=dict(status='completed',diagnostic_only=True,training_forbidden=True,detail=self.detail,ticks=self.tick,physical_per_env_sha256=[x.hexdigest() for x in self.physical],feature_per_env_sha256=[x.hexdigest() for x in self.features],deep_native_resets_checked=self.resets,models_frozen=True,columns=self.columns,selected_rows=sum(len(x) for x in self.rows),no_RNG_calls=True,no_observation_refresh=True,no_physical_state_rewrite=True)
  (self.D/'withdrawal_probe.json').write_text(json.dumps(v,indent=2))
