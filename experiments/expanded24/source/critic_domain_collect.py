"""One complete frozen-policy native cohort in a fresh physical domain."""
import copy,hashlib,json,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from boundary_rl import model_hash
from context_ppo import Shadow,metrics,predict,sha,utc
from frozen_boundary_history import FrozenBoundaryHistory,precision
from joint_input_upgrade import expose_joint5
from deep_joint_hook import DeepHook
from brake_wrapper import BrakeHook
from native_groups import arrays
from joint_storage import cpu,MINFREE
from guarded_value import value_gate_passed

SEEDS=[1125,1127,1129,1133,1135,1137]
DOMAIN_NAMES=['friction_scales','stiffness_scales','peg_masses','grip_axial_offsets','force_noise_scales']

def collect(env,raw,boundary,logdir,args):
 assert args.critic_domain_collect and args.seed in SEEDS and args.joint5_input and not args.deep_ppo_updates and not args.critic_phase_pilot
 assert args.fixed_contact_cases and args.boundary_mode=='eval' and raw.num_envs==128 and raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry
 assert not any([args.initial_case_replay,args.post_ppo_updates,args.critic_shadow,args.paired_diagnostic,args.fixed_final_retry_brake,args.independent_boundary_history])
 precision();D=Path(logdir);index=SEEDS.index(args.seed);blind=index>=4;start=time.monotonic();source=Path(args.deep_recovery_checkpoint);assert sha(source)=='055571a29e42fc6e36b347823cb5aa57e6ea1505af6fb59226ab5108d33eabdf'
 model=raw.deep_recovery_model;original=torch.load(source,map_location=raw.device,weights_only=False)['model_state_dict'];assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items());root=copy.deepcopy(model).eval();expose_joint5(model)
 for m in [root,model,boundary.model]:
  for p in m.parameters():p.requires_grad_(False)
 initial=cpu(model.state_dict());frozen={n:model_hash(m) for n,m in boundary.models().items()};boundary_hash=model_hash(boundary.model)
 bridge=FrozenBoundaryHistory(raw,boundary);shadow=Shadow(raw,boundary);hook=DeepHook(raw,boundary,root);brake=BrakeHook(raw,boundary,hook);raw.fixed_final_retry_hook=brake
 domain={n:cpu(getattr(raw,n)).numpy() for n in DOMAIN_NAMES};domain_hashes={n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in domain.items()};np.savez_compressed(D/'domain_parameters.npz',**domain)
 torch.manual_seed(args.seed);env.seed(args.seed);obs,_=env.reset();boundary.reset(torch.ones(128,dtype=torch.bool,device=raw.device));assert not bridge.h.any();active=torch.ones(128,dtype=torch.bool,device=raw.device);hook.active=active.clone();records=[];ends={}
 (D/'domain_progress.json').write_text(json.dumps(dict(phase='collecting',seed=args.seed,index=index,blind=blind,actor_updates=0),indent=2))
 for tick in range(730):
  alive=active.clone()
  with torch.no_grad():
   action=boundary.act_inference(obs);obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
   for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
    if bool(alive[i]):
     ends[i]=dict(env=index*128+i,native_env=i,seed=args.seed,success=bool(term['success'][j]),end_reason=int(term['end_reason'][j]),peak_force_n=float(term['peak_force_n'][j]),seconds=float(term['seconds'][j]),attempts=int(term['recovery_attempts'][j]),exits=int(term['recovery_exits'][j]),realignments=int(term['recovery_realignments'][j]),end_mode=int(term['recovery_end_mode'][j]))
   records.append((hook.cache,shadow.feature.clone(),shadow.value.clone(),reward.float().clone()*alive,done.clone(),alive,boundary.public.clone(),hook.compose_cache,hook.compose_cache['clipped'],hook.compose_cache['delayed']))
   boundary.reset(done);active&=~done;hook.active=active.clone()
  if not active.any():break
 assert not active.any() and len(ends)==128 and not brake.brake.latched.any()
 rows=[ends[i] for i in range(128)];(D/'episode_outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));M=torch.stack([r[0]['selected']&r[5] for r in records]);C=torch.stack([r[1] for r in records]);V=torch.stack([r[2] for r in records]);rewards=torch.stack([r[3] for r in records]);done=torch.stack([r[4] for r in records]);returns=torch.zeros_like(V);future=torch.zeros(128,device=raw.device);phase=torch.zeros_like(V)
 for t in reversed(range(len(records))):future=rewards[t]+.99*future*(~done[t]);returns[t]=future
 for e in range(128):
  q=M[:,e].nonzero().flatten();phase[q,e]=torch.arange(len(q),device=raw.device)/max(len(q)-1,1)
 eid=(torch.arange(128,device=raw.device)+index*128)[None,:].expand_as(M)[M];held=((eid//4)%4)==0;counts=M.sum(0).cpu().tolist();extra=sorted([e for e in range(128) if ((e//4)%4)==0 and counts[e]>0],key=lambda e:(-counts[e],e))[:4];probes=[0,1,2,3,32,33,34,35];probes+=[e for e in extra if e not in probes];probe=torch.tensor(probes,device=raw.device)
 np.savez_compressed(D/'native_probes.npz',**arrays(records,probe,hook,brake));np.savez_compressed(D/'selected_data.npz',critic_features=C[M].cpu().numpy(),prior_value=V[M].cpu().numpy(),target=returns[M].cpu().numpy(),phase=phase[M].cpu().numpy(),env=eid.cpu().numpy())
 score=metrics(V[M],returns[M],phase[M],held) if held.any() else None;coverage=int(held.sum())>200 and int((~held).sum())>500;forces=sum(r['peak_force_n']>100 for r in rows)
 bridge.verify();assert {n:model_hash(m) for n,m in boundary.models().items()}==frozen;assert model_hash(boundary.model)==boundary_hash;assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items());assert all(np.array_equal(cpu(getattr(raw,n)).numpy(),domain[n]) for n in DOMAIN_NAMES)
 assert not any(p.requires_grad for m in [root,model,boundary.model] for p in m.parameters())
 torch.save(dict(deep_state_dict=initial,boundary_state_dict=cpu(boundary.model.state_dict()),domain={n:cpu(getattr(raw,n)) for n in DOMAIN_NAMES},torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),all_actor_and_critic_frozen=True),D/'frozen_snapshot.pt')
 result=dict(status='collected',seed=args.seed,index=index,blind=blind,episodes=128,successes=sum(r['success'] for r in rows),force_aborts=forces,force_gate_passed=forces<=12,coverage_passed=coverage,heldout_steps=int(held.sum()),train_steps=int((~held).sum()),selected_steps=int(M.sum()),selected_episodes=int((M.sum(0)>0).sum()),selected_steps_by_native_env=counts,selected_steps_by_case={str(c):sum(counts[c::4]) for c in range(4)},native_ticks=len(records),native_live_steps=int(torch.stack([r[5] for r in records]).sum()),prior_metrics=score,value_gate_passed=bool(score and value_gate_passed(score)),blind_gate_passed=bool(blind and forces<=12 and coverage and score and value_gate_passed(score)),actor_updates=0,value_fit_attempts=0,frozen_role_hashes=frozen,boundary_model_hash=boundary_hash,deep_state_hash=model_hash(model),domain_hashes=domain_hashes,domain_joint_hash=hashlib.sha256(''.join(domain_hashes.values()).encode()).hexdigest(),boundary_checkpoint_sha256=sha(args.boundary_checkpoint),source_deep_sha256=sha(source),new_joint_column_max_abs=float(model.actor[0].weight[:,10].abs().max()),all_actor_and_critic_frozen=True,wall_seconds=time.monotonic()-start)
 assert result['new_joint_column_max_abs']==0;assert shutil.disk_usage(D).free>=MINFREE
 (D/'domain_collect_result.json').write_text(json.dumps(result,indent=2));(D/'domain_progress.json').write_text(json.dumps(dict(phase='completed',seed=args.seed,episodes=128,actor_updates=0),indent=2));print('CRITIC_DOMAIN_COLLECT',json.dumps(result),flush=True)
