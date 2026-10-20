"""Zero-update paired native episodes; every recorded item forbidden training."""
import hashlib,json,types
from pathlib import Path
import numpy as np
import torch
from boundary_rl import model_hash
from frozen_boundary_history import FrozenBoundaryHistory,precision,sequence_mean

def run(env,raw,boundary,logdir,args):
 assert args.history_proof in ['root','bridge'] and args.seed==1083 and raw.num_envs==128 and args.fixed_contact_cases
 assert not args.post_ppo_updates and not args.initial_case_replay and not args.paired_diagnostic and args.boundary_mode=='eval'
 assert raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry
 precision();D=Path(logdir);n=128;device=raw.device;bridge=FrozenBoundaryHistory(raw,boundary) if args.history_proof=='bridge' else None
 gen=torch.Generator(device=device).manual_seed(108301);bank=torch.randn((731,n,3),generator=gen,device=device);ids=torch.arange(n,device=device)
 def measured(this,true_force):return (true_force+this.force_bias+bank[this.episode_controller_steps.long().clamp(0,730),ids]*this.force_noise_scales).clamp(-100,100)*.02
 raw._measured_force=types.MethodType(measured,raw);torch.manual_seed(args.seed);env.seed(args.seed);obs,_=env.reset();boundary.reset(torch.ones(n,device=device,dtype=torch.bool))
 for m in boundary.models().values():m.reset(torch.ones(n,device=device,dtype=torch.bool))
 snapshot=dict(q=raw.robot.data.joint_pos[:,raw.arm_ids],qd=raw.robot.data.joint_vel[:,raw.arm_ids],targets=raw.targets,root_pos=raw.robot.data.root_pos_w,root_quat=raw.robot.data.root_quat_w,hole_pos=raw.hole_pos,hole_quat=raw.hole_quat,hole_bias=raw.hole_estimation_bias,force_bias=raw.force_bias,force_scales=raw.force_noise_scales,friction=raw.friction_scales.to(device),stiffness=raw.stiffness_scales,delay=raw.action_delay.steps,delay_buffer=raw.action_delay.buffer,command_pos=raw.command_pos,command_quat=raw.command_quat,command_initialized=raw.command_initialized,preparing=raw.preparing,offset=raw.training_episode_offset,first_actor=obs['policy'],first_critic=obs['critic'],post_left=boundary.post_left,prev_align=boundary.prev_align,prev_attempt=boundary.prev_attempt)
 for k,v in vars(raw.supervisor).items():
  if isinstance(v,torch.Tensor):snapshot['supervisor_'+k]=v
 for k,m in boundary.models().items():
  h=getattr(getattr(m,'memory_a',None),'hidden_state',None);snapshot['actor_history_'+k]=torch.zeros(1,n,33,device=device) if h is None else h
 snapshot['feature_history']=torch.zeros(1,n,33,device=device) if bridge is None else bridge.h
 snapshot={k:v.cpu().numpy().copy() for k,v in snapshot.items()};np.savez_compressed(D/'initial.npz',**snapshot)
 hashes={k:model_hash(m) for k,m in boundary.models().items()};bh=model_hash(boundary.model);cache={};role_names=[]
 for name in ['recovery_policy','alignment_policy','initial_policy','late_initial_policy','deep_recovery_policy','near_alignment_policy']:
  policy=getattr(raw,name,None)
  if policy is None:continue
  role_names.append(name)
  def wrap(policy,name):
   def call(obs):
    out=policy(obs);cache[name]=out.detach().clone();return out
   return call
  setattr(raw,name,wrap(policy,name))
 active=torch.ones(n,device=device,dtype=torch.bool);traces=[];digests=[];inputs=[];means=[];anchors=[];masks=[];alives=[];columns={};pre=raw._pre_physics_step;post_cache={}
 original=boundary.compose
 def compose(r,base,first):
  out=original(r,base,first);post_cache['mask']=r.training_student_control&active&~first;return out
 boundary.compose=compose
 def pre_step(this,action):
  alive=active.clone();pre(action);s=this.supervisor
  fields=dict(alive=alive,step=this.episode_controller_steps,mode=s.mode,attempts=s.attempts,exits=s.exits,realignments=s.realignments,preparing=this.step_preparing,truth9=torch.cat(this._geometry(),-1),applied=this.r2_last_action,delayed=this.action_delay.buffer[this.action_delay.steps,this.action_delay.ids],post=post_cache['mask'],q=this.robot.data.joint_pos[:,this.arm_ids],command=this.command_pos)
  pieces=[];offset=0
  for k,v in fields.items():v=v.reshape(n,-1);columns[k]=[offset,offset+v.shape[1]];offset+=v.shape[1];pieces.append(v.float())
  traces.append(torch.cat(pieces,-1).cpu().numpy())
  dense=[this.robot.data.joint_pos[:,this.arm_ids],this.robot.data.joint_vel[:,this.arm_ids],this.targets,this.command_pos,this.command_quat,this.action_delay.buffer.permute(1,0,2).reshape(n,-1),boundary.features,boundary.context,boundary.last_base,boundary.delta]
  for name in role_names:dense.append(cache[name])
  for name,m in boundary.models().items():
   h=getattr(getattr(m,'memory_a',None),'hidden_state',None);dense.append(torch.zeros(n,33,device=device) if h is None else h[0])
  if bridge is not None:dense.append(bridge.h[0])
  else:dense.append(boundary.base.memory_a.hidden_state[0])
  for k in sorted(vars(s)):
   v=getattr(s,k)
   if isinstance(v,torch.Tensor):dense.append(v.reshape(n,-1).float())
  data=torch.cat(dense,-1).cpu().numpy();digests.append(np.stack([np.frombuffer(hashlib.sha256(row.tobytes()).digest(),dtype='uint8') for row in data]))
 raw._pre_physics_step=types.MethodType(pre_step,raw);outcomes={}
 for tick in range(730):
  alive=active.clone()
  with torch.no_grad():
   x=boundary.base.actor_obs_normalizer(boundary.base.get_actor_obs(obs)).clone();action=boundary.act_inference(obs);mean=boundary.last_base.clone();anchor=mean.clone() if bridge is None else bridge.anchor.clone()
   obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
   for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
    if bool(alive[i]):outcomes[i]={k:(bool(v[j]) if k=='success' else float(v[j])) for k,v in term.items() if isinstance(v,torch.Tensor) and v.ndim==1 and len(v)==len(term['env_ids'])};outcomes[i]['env']=i
   inputs.append(x);means.append(mean);anchors.append(anchor);masks.append(post_cache['mask'].clone());alives.append(alive);boundary.reset(done);active&=~done
  if not active.any():break
 assert not active.any() and len(outcomes)==128
 assert hashes=={k:model_hash(m) for k,m in boundary.models().items()} and bh==model_hash(boundary.model)
 np.savez_compressed(D/'trajectory.npz',trace=np.stack(traces),dense_sha256=np.stack(digests));(D/'trace_columns.json').write_text(json.dumps(dict(columns=columns,roles=role_names,whole_dense_history_and_role_output_hashes=True,precision='TF32off'),indent=2))
 error=None;gradient=None
 if bridge is not None:
  bridge.verify();X=torch.stack(inputs);OLD=torch.stack(means);ROOT=torch.stack(anchors);Alive=torch.stack(alives);M=torch.stack(masks);h0=torch.zeros(1,n,33,device=device);errors=[]
  boundary.base.memory_a.rnn.train()
  with torch.no_grad():
   for e in torch.arange(n,device=device).split(16):errors.append(float(abs(sequence_mean(boundary.base,X[:,e],h0[:,e])-OLD[:,e])[Alive[:,e]].max()))
  error=max(errors);assert error<2e-5,error
  # Tests use a disposable clone only. No optimizer and no weights leave this test.
  clone=__import__('copy').deepcopy(boundary.base);clone.memory_a.rnn.flatten_parameters()
  for name,p in clone.named_parameters():p.requires_grad_(name.startswith(('actor.','memory_a.rnn.')))
  params=[*clone.memory_a.rnn.parameters(),*clone.actor.parameters()];gradient={};phase=torch.zeros_like(M,dtype=torch.float)
  for e in range(n):
   q=M[:,e].nonzero().flatten();phase[q,e]=torch.arange(len(q),device=device)/max(len(q)-1,1)
  for label,q in [('early',phase<=1/3),('middle',(phase>1/3)&(phase<2/3)),('late',phase>=2/3)]:
   mask=M[:,:16]&q[:,:16];assert mask.any()
   with torch.enable_grad():
    mu=sequence_mean(clone,X[:,:16],h0[:,:16]);loss=(mu-OLD[:,:16]-.01).square().sum(-1)[mask].mean();gs=torch.autograd.grad(loss,params);v=sum(float(g.abs().sum()) for g in gs[:4]);assert v>0;gradient[label]=v
  probe=torch.tensor([0,1,2,3,32,33,34,35],device=device);np.savez_compressed(D/'sequence_probe.npz',normalized_observations=X[:,probe].cpu().numpy(),h0=h0[:,probe].cpu().numpy(),old_mean=OLD[:,probe].cpu().numpy(),root_mean=ROOT[:,probe].cpu().numpy(),alive=Alive[:,probe].cpu().numpy(),selected=M[:,probe].cpu().numpy())
  assert hashes=={k:model_hash(m) for k,m in boundary.models().items()}
 result=dict(status='completed',diagnostic_only=True,training_forbidden=True,variant=args.history_proof,seed=1083,episodes=128,successes=sum(x['success'] for x in outcomes.values()),force_aborts=sum(x['peak_force_n']>100 for x in outcomes.values()),noise_bank_sha256=hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest(),all_weights_unchanged=True,physical_state_never_rewritten=True,initial_state_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in snapshot.items()},full_sequence_all128_replay_error=error,disposable_clone_GRU_gradient=gradient,outcomes=[outcomes[i] for i in range(n)])
 (D/'history_proof.json').write_text(json.dumps(result,indent=2));print('HISTORY_PROOF',json.dumps({k:v for k,v in result.items() if k not in ['outcomes','initial_state_hashes']}),flush=True)
