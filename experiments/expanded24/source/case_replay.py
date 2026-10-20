"""Seen-case diagnostic only. Replace native reset state, never live episode state."""
import json,types
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply
class Replay:
 def __init__(self,raw,path,logdir):
  self.r=raw;self.D=Path(logdir);self.case=json.loads(Path(path).read_text());self.generation=-1;self.outcomes=[];self.first_error=None;i=127;self.i=i;device=raw.device;c=self.case
  assert raw.num_envs==128 and not raw.cfg.fixed_contact_cases
  self.noise=torch.tensor(c['additive_force_disturbance'],device=device);self.q=torch.tensor(c['joint_pos'],device=device);self.hp=torch.tensor(c['hole_pos'],device=device);self.hq=torch.tensor(c['hole_quat'],device=device)
  assert torch.allclose(raw.scene.env_origins[i],torch.tensor(c['expected_origin'],device=device),atol=1e-5)
  ids=torch.arange(raw.num_envs,dtype=torch.int32)
  ratio=c['friction_scale']/float(raw.friction_scales[i]);raw.friction_scales[i]=c['friction_scale']
  for asset in [raw.robot,*raw.walls]:
   material=asset.root_physx_view.get_material_properties().clone();material[i,:,:2]*=ratio;asset.root_physx_view.set_material_properties(material,ids)
   assert torch.allclose(asset.root_physx_view.get_material_properties()[i],material[i],atol=1e-6)
  raw.stiffness_scales[i]=c['stiffness_scale'];gains=raw.robot.root_physx_view.get_dof_stiffnesses().clone();gains[i,raw.arm_ids]=800*c['stiffness_scale'];raw.robot.write_joint_stiffness_to_sim(gains[i,raw.arm_ids][None].to(device),joint_ids=raw.arm_ids,env_ids=torch.tensor([i],device=device));raw.robot.actuators['arm'].stiffness[i]=800*c['stiffness_scale']
  measured=raw._measured_force;reset=raw._reset_idx;pre=raw._pre_physics_step
  def force(this,true_force):
   value=measured(true_force);tick=min(int(this.episode_controller_steps[i]),len(self.noise)-1);value[i]=(true_force[i]+self.noise[tick]).clamp(-100,100)*.02;return value
  def on_reset(this,env_ids):
   ids=this.robot._ALL_INDICES if env_ids is None else env_ids;selected=bool((ids==i).any())
   if selected and self.generation>=0:
    self.outcomes.append(dict(generation=self.generation,success=bool(this.ever_success[i]),end_reason=int(this.end_reason[i]),peak_force_n=float(this.peak_force[i]),steps=int(this.episode_controller_steps[i]),realignments=int(this.supervisor.realignments[i]),attempts=int(this.supervisor.attempts[i]),exits=int(this.supervisor.exits[i])))
   reset(env_ids)
   if selected:
    self.generation+=1;e=torch.tensor([i],device=device);q=this.robot.data.default_joint_pos[e].clone();q[:,this.arm_ids]=self.q
    this.targets[e]=q;this.robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=e);this.robot.set_joint_position_target(q,env_ids=e)
    this.hole_pos[i]=self.hp;this.hole_quat[i]=self.hq;this.hole_estimation_bias[i]=torch.tensor(c['observation_bias'],device=device);this.action_delay.steps[i]=c['delay_steps']
    for wall,offset in zip(this.walls,this.wall_offsets):
     shifted=quat_apply(self.hq[None],torch.tensor(offset,device=device)[None]);wall.write_root_pose_to_sim(torch.cat((self.hp[None]+this.scene.env_origins[e]+shifted,self.hq[None]),-1),env_ids=e)
  def before(this,actions):
   pre(actions)
   if self.generation==0 and int(this.episode_controller_steps[i])==1:
    expected=torch.tensor(c['expected_first_actor'],device=device);self.first_error=float((this.r2_last_actor[i]-expected).abs().max());assert self.first_error<2e-4,('Captured first actor mismatch',self.first_error)
  raw._measured_force=types.MethodType(force,raw);raw._reset_idx=types.MethodType(on_reset,raw);raw._pre_physics_step=types.MethodType(before,raw)
 def finalize(self):
  assert self.first_error is not None and any(v['generation']==0 for v in self.outcomes)
  v=dict(status='completed',diagnostic_only=True,training_forbidden=True,env=127,source=self.case['source'],first_actor_max_error=self.first_error,reset_override_only=True,live_physical_state_never_rewritten=True,force_disturbance_identical_by_logical_step=True,outcomes=self.outcomes)
  (self.D/'case_result.json').write_text(json.dumps(v,indent=2));print('CASE_REPLAY_RESULT',json.dumps(v),flush=True)
