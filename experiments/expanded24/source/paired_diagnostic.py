"""Auxiliary common-random-number native episodes; explicitly forbidden training."""
import hashlib,json,types
from pathlib import Path
import numpy as np
import torch
from boundary_rl import model_hash
from frozen_boundary_history import FrozenBoundaryHistory,precision

def digest(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()

def run(env,raw,boundary,logdir,args):
 assert args.paired_diagnostic and args.pair_scale in [0.,1.] and args.seed==1085 and raw.num_envs==128
 assert args.fixed_contact_cases and not args.post_ppo_updates and not args.initial_case_replay and not args.critic_shadow and not args.alignment_axial_cap
 assert raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry and args.boundary_mode=='eval'
 precision();bridge=FrozenBoundaryHistory(raw,boundary)
 actor_sha=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest();assert actor_sha in ['d092d8c3dbb5a440d1f90f0e0c005bf1b754a5d3d7107234f7a50a13812abf1f','227e03dc9ca025819f82c5f285c93c20a4ed4831538bf64d3f58e392fbc90c24']
 root_actor=actor_sha.startswith('d092d8c3');label=('root' if root_actor else 'cand')+('det' if args.pair_scale==0 else 'noise')
 D=Path(logdir);assert D.name=="reliability_v2_guarded_recurrent_20261017_diag_"+label+"_random"
 n=raw.num_envs;device=raw.device;g=torch.Generator(device=device);g.manual_seed(108501)
 force_bank=torch.randn((731,n,3),generator=g,device=device);g.manual_seed(108502);epsilon=torch.randn((731,n,6),generator=g,device=device)
 sigma=torch.tensor([.01,.01,.01,.0075,.0075,.0075],device=device)*args.pair_scale;ids=torch.arange(n,device=device)
 def measured(this,true_force):
  t=this.episode_controller_steps.long().clamp(0,730)
  return (true_force+this.force_bias+force_bank[t,ids]*this.force_noise_scales).clamp(-100,100)*.02
 raw._measured_force=types.MethodType(measured,raw)
 torch.manual_seed(args.seed);env.seed(args.seed);obs,_=env.reset();boundary.reset(torch.ones(n,device=device,dtype=torch.bool))
 for m in boundary.models().values():m.reset(torch.ones(n,device=device,dtype=torch.bool))
 snapshot=dict(q=raw.robot.data.joint_pos[:,raw.arm_ids],qd=raw.robot.data.joint_vel[:,raw.arm_ids],targets=raw.targets,root_pos=raw.robot.data.root_pos_w,root_quat=raw.robot.data.root_quat_w,hole_pos=raw.hole_pos,hole_quat=raw.hole_quat,hole_bias=raw.hole_estimation_bias,force_bias=raw.force_bias,force_scales=raw.force_noise_scales,friction=raw.friction_scales.to(device),stiffness=raw.stiffness_scales,delay=raw.action_delay.steps,delay_buffer=raw.action_delay.buffer,command_pos=raw.command_pos,command_quat=raw.command_quat,command_initialized=raw.command_initialized,preparing=raw.preparing,offset=raw.training_episode_offset,mode=raw.supervisor.mode,attempts=raw.supervisor.attempts,exits=raw.supervisor.exits,realignments=raw.supervisor.realignments,first_actor=obs['policy'],first_critic=obs['critic'],boundary_post_left=boundary.post_left,boundary_prev_align=boundary.prev_align,boundary_prev_attempt=boundary.prev_attempt)
 snapshot["feature_history"]=bridge.h
 for k,v in vars(raw.supervisor).items():
  if isinstance(v,torch.Tensor):snapshot["supervisor_"+k]=v
 for name,m in boundary.models().items():
  if hasattr(m,'memory_a'):
   h=m.memory_a.hidden_state;snapshot['actor_history_'+name]=torch.zeros(1,n,33,device=device) if h is None else h
 snapshot={k:v.detach().cpu().numpy().copy() for k,v in snapshot.items()};np.savez_compressed(D/'paired_initial.npz',**snapshot)
 if label=="rootdet":np.savez_compressed(D/'diagnostic_banks_forbidden_training.npz',force=force_bank.cpu().numpy(),epsilon=epsilon.cpu().numpy())
 models={k:model_hash(v) for k,v in boundary.models().items()};boundary_hash=model_hash(boundary.model);active=torch.ones(n,device=device,dtype=torch.bool);post_mask=torch.zeros_like(active);original=boundary.compose
 def compose(r,base,first):
  nonlocal post_mask
  post_mask=r.training_student_control&active&~first
  if post_mask.any():assert torch.equal(base[post_mask],boundary.last_base[post_mask].clamp(-1,1))
  latent=boundary.last_base+sigma*epsilon[r.episode_controller_steps.long().clamp(0,730),ids];mixed=base.clone();mixed[post_mask]=latent[post_mask].clamp(-1,1)
  return original(r,mixed,first)
 boundary.compose=compose
 seen_post=torch.zeros(n,device=device,dtype=torch.bool);prefix_values=[];prefix_masks=[];prefix_deltas=[];own_prefix_masks=[];prefix_errors=dict(features66=0.,context_tail7=0.,residual6=0.);prefix_mask_exact=True;prefix_compared=0
 if label!='rootdet':
  ref=np.load(D.parent/"reliability_v2_guarded_recurrent_20261017_diag_rootdet_random/prefix_reference_forbidden_training.npz");reference_values=ref['values'];reference_masks=ref['mask']
 rows=[];columns={};original_pre=raw._pre_physics_step
 def pre(this,actions):
  nonlocal prefix_mask_exact,prefix_compared
  alive=active.clone();original_pre(actions);s=this.supervisor
  prefix=(alive&~seen_post&~post_mask).cpu().numpy();seen_post.logical_or_(post_mask)
  features=torch.cat((boundary.features,boundary.context[:,66:],boundary.delta),-1).cpu().numpy();assert features.shape==(n,79)
  if label=='rootdet':prefix_values.append(np.where(prefix[:,None],features,0));prefix_masks.append(prefix)
  else:
   tick=len(rows);rm=reference_masks[tick] if tick<len(reference_masks) else np.zeros(n,dtype=bool);prefix_mask_exact=prefix_mask_exact and np.array_equal(prefix,rm)
   keep=prefix&rm;delta=np.zeros_like(features)
   if keep.any():
    delta[keep]=features[keep]-reference_values[tick,keep];diff=abs(delta[keep]);prefix_compared+=int(keep.sum())
    for k,a,b in [('features66',0,66),('context_tail7',66,73),('residual6',73,79)]:prefix_errors[k]=max(prefix_errors[k],float(diff[:,a:b].max()))
   prefix_deltas.append(delta);own_prefix_masks.append(prefix)
  fields=dict(alive=alive,step=this.episode_controller_steps,mode=s.mode,attempts=s.attempts,exits=s.exits,realignments=s.realignments,preparing=this.step_preparing,truth9=torch.cat(this._geometry(),-1),applied=this.r2_last_action,delayed=this.action_delay.buffer[this.action_delay.steps,this.action_delay.ids],post=post_mask,q=this.robot.data.joint_pos[:,this.arm_ids],command=this.command_pos)
  parts=[];offset=0
  for k,v in fields.items():v=v.reshape(n,-1);columns[k]=[offset,offset+v.shape[1]];offset+=v.shape[1];parts.append(v.float())
  rows.append(torch.cat(parts,-1).cpu().numpy())
 raw._pre_physics_step=types.MethodType(pre,raw);outcomes={}
 for tick in range(730):
  alive=active.clone()
  with torch.no_grad():
   action=boundary.act_inference(obs);obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
   for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
    if not bool(alive[i]):continue
    outcomes[i]={k:(bool(v[j]) if k=='success' else float(v[j])) for k,v in term.items() if isinstance(v,torch.Tensor) and v.ndim==1 and len(v)==len(term['env_ids'])};outcomes[i]['env']=i
   boundary.reset(done);active&=~done
  if not active.any():break
 assert not active.any() and len(outcomes)==128
 data=np.stack(rows);np.savez_compressed(D/'paired_trace.npz',trace=data);(D/'paired_trace.json').write_text(json.dumps(dict(columns=columns,shape=list(data.shape),before_physics=True,terminal_outcomes_separately_recorded=True),indent=2))
 bridge.verify()
 if label!='rootdet':
  if len(rows)<len(reference_masks):prefix_mask_exact=prefix_mask_exact and not reference_masks[len(rows):].any()
  np.savez_compressed(D/'prefix_delta_forbidden_training.npz',delta=np.stack(prefix_deltas),mask=np.stack(own_prefix_masks))
 if label=='rootdet':
  np.savez_compressed(D/'prefix_reference_forbidden_training.npz',values=np.stack(prefix_values),mask=np.stack(prefix_masks));prefix_compared=int(np.stack(prefix_masks).sum())
 assert {k:model_hash(v) for k,v in boundary.models().items()}==models and model_hash(boundary.model)==boundary_hash
 outcomes=[outcomes[i] for i in range(n)];result=dict(status='completed',diagnostic_only=True,training_forbidden=True,seed=args.seed,scale=args.pair_scale,condition=label,actor_checkpoint_sha256=actor_sha,qualified_feature_prefix=dict(mask_exact=prefix_mask_exact,maximum_errors=prefix_errors,compared_steps=prefix_compared,passed=prefix_mask_exact and all(v<=2e-5 for v in prefix_errors.values()),all_exact=prefix_mask_exact and all(v==0 for v in prefix_errors.values())),complete_episodes=128,successes=sum(x['success'] for x in outcomes),force_aborts=sum(x['peak_force_n']>100 for x in outcomes),mean_seconds=sum(x['seconds'] for x in outcomes)/n,force_bank_sha256=digest(force_bank),epsilon_bank_sha256=digest(epsilon),initial_state_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in snapshot.items()},actor_and_boundary_weights_unchanged=True,physical_state_never_rewritten=True,native_gates_unchanged=True,outcomes=outcomes)
 (D/'paired_result.json').write_text(json.dumps(result,indent=2));print('PAIRED_DIAGNOSTIC',json.dumps({k:v for k,v in result.items() if k not in ['outcomes','initial_state_hashes']}),flush=True)
