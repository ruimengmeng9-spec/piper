"""Fit one actor MLP using successful, force-bounded native training episodes.

The actor GRU, critic, normalizers, exploration, and all five frozen models are
unchanged. Case labels only balance training examples and choose supervision.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from masked_ppo import model_hash

def fit(env,raw,runner,logdir,args):
 assert args.mode=='eval' and args.iterations==0 and args.seed not in range(700,706) and args.seed not in range(800,804)
 assert raw.cfg.fixed_contact_cases and not raw.cfg.lateral_cases_only and not raw.cfg.replay_original_axial_cases and not raw.cfg.replay_original_positive_lateral
 assert abs(raw.cfg.recovery_offset-.005)<1e-9 and abs(raw.cfg.recovery_center_x+.0045)<1e-9
 assert raw.cfg.training_joint_margin_cost==0 and not raw.cfg.train_stop_failed_retry
 m=runner.alg.policy;m.eval();assert m.is_recurrent
 before={k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('actor.')}
 frozen=[('initial',raw.initial_model),('withdrawal',raw.recovery_model),('alignment',raw.alignment_model),('plain_teacher',raw.training_teacher_model),('mirrored_teacher',raw.mirror_model)]
 frozen_hashes={n:model_hash(model) for n,model in frozen};gru_hash=model_hash(m.memory_a)
 for parameter in m.parameters():parameter.requires_grad_(False)
 for parameter in m.actor.parameters():parameter.requires_grad_(True)
 opt=torch.optim.Adam(m.actor.parameters(),lr=3e-4)
 captured={};hook=m.actor.register_forward_pre_hook(lambda module,inputs:captured.update(z=inputs[0].detach().clone()))
 D=Path(logdir);reports=[];aggregate={'z':[],'y':[],'case':[]};mixes=[1.,.7,.3,0.,0.,0.];quota=2
 def progress(phase,**extra):
  (D/'safe_mirror_progress.json').write_text(json.dumps(dict(phase=phase,rounds=reports,actor_inputs=33,actor_GRU_frozen=True,actor_GRU_hash=gru_hash,frozen_model_hashes=frozen_hashes,final_seeds_unread=[800,801,802,803],**extra),indent=2))
 progress('collecting_first_safe_bank',training_seed=args.seed)
 for ri,mix in enumerate(mixes):
  raw.training_teacher_mix=mix;m.eval();env.seed(args.seed+ri);obs,_=env.reset();m.memory_a.hidden_state=None;m.memory_c.hidden_state=None
  count=torch.zeros(env.num_envs,device=env.device,dtype=torch.long);env_index=torch.arange(env.num_envs,device=env.device)
  batches={'z':[],'y':[],'x':[],'uid':[]};terminals=[];safe_uids=[]
  with torch.no_grad():
   for step in range(1700):
    if step%128==0:progress('collecting_round',current_round=ri+1,training_seed=args.seed+ri,teacher_action_fraction=mix,physics_frames=step,completed_episode_quota=int(count.sum()))
    x=obs['policy'].clone();learner=m.act_inference(obs).clone();raw.unclipped_policy_actions=learner.detach().clone();z=captured['z'].clone()
    obs,rewards,dones,info=env.step(learner)
    selected=raw.training_student_control&(count<quota)
    if selected.any():
     batches['z'].append(z[selected].clone());batches['x'].append(x[selected].clone());batches['y'].append(raw.training_teacher_target[selected].clone());batches['uid'].append((env_index*quota+count)[selected].clone())
    m.reset(dones.bool())
    if dones.any():
     t=info['terminal'];ids=t['env_ids'];assert torch.equal(ids,dones.nonzero().flatten())
     valid=count[ids]<quota
     for j in valid.nonzero().flatten().cpu().tolist():
      i=int(ids[j]);uid=int(i*quota+count[i]);peak=float(t['peak_force_n'][j]);success=bool(t['success'][j])
      chain=bool(t['recovery_true_exits'][j]>=1 and t['recovery_true_released_exits'][j]>=1 and t['recovery_realignments'][j]>=1 and t['handover_force_n'][j]>=5.)
      safe=success and peak<=40. and chain
      row=dict(uid=uid,env=i,case=int(raw.case_ids[i]),success=success,peak_force_n=peak,native_chain_verified=chain,retained_for_training=safe,end_reason=int(t['end_reason'][j]));terminals.append(row)
      if safe:safe_uids.append(uid)
     count[ids[valid]]+=1
    if (count>=quota).all():break
   assert (count>=quota).all(),('Incomplete training collection',ri,count.tolist())
  all_data={k:torch.cat(v) for k,v in batches.items()};uids=all_data['uid'];keep=torch.isin(uids,torch.tensor(safe_uids,device=env.device,dtype=torch.long));cases=raw.case_ids[(uids//quota).long()]
  safe_episode_counts=[sum(t['case']==c and t['retained_for_training'] for t in terminals) for c in range(4)]
  sample_counts=[int((keep&(cases==c)).sum()) for c in range(4)]
  retained_peaks=[t['peak_force_n'] for t in terminals if t['retained_for_training']]
  bank=dict(training_seed=args.seed+ri,teacher_action_fraction=mix,all_episodes=terminals,safe_episode_counts=safe_episode_counts,safe_sample_counts=sample_counts,maximum_retained_peak_force_n=max(retained_peaks,default=None),filter='complete native success, genuine released exit+realignment, entire-episode true peak<=40N',formal_acceptance=False)
  (D/f'training_bank_round{ri+1}.json').write_text(json.dumps(bank,indent=2))
  np.savez_compressed(D/f'training_bank_round{ri+1}.npz',**{k:v.detach().cpu().numpy() for k,v in all_data.items()},case=cases.cpu().numpy(),retained=keep.cpu().numpy())
  if ri==0:
   gate=dict(passed=min(safe_episode_counts)>=16 and min(sample_counts)>=1024,safe_episode_counts=safe_episode_counts,safe_sample_counts=sample_counts,maximum_retained_peak_force_n=max(retained_peaks,default=None),unsafe_or_incomplete_episodes_excluded=True,weights_updated_before_gate=False,teacher_over100N_all_episodes=sum(t['peak_force_n']>100 for t in terminals),source='new training seed, not development/final seeds')
   (D/'safe_demonstration_gate.json').write_text(json.dumps(gate,indent=2));progress('safe_bank_gate_checked',gate=gate)
   assert gate['passed'],('Insufficient four-case safe demonstrations; no fitting allowed',gate)
  if keep.any():
   aggregate['z'].append(all_data['z'][keep].clone());aggregate['y'].append(all_data['y'][keep].clone());aggregate['case'].append(cases[keep].clone())
  Z=torch.cat(aggregate['z']);Y=torch.cat(aggregate['y']);C=torch.cat(aggregate['case']);counts=torch.bincount(C,minlength=4);assert (counts>0).all()
  weights=len(C)/(4*counts[C].float());m.actor.train();losses=[]
  for epoch in range(10):
   for ids in torch.randperm(len(Z),device=env.device).split(1024):
    w=weights[ids];loss=((m.actor(Z[ids])-Y[ids]).square().mean(-1)*w).sum()/w.sum()
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.actor.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  m.eval()
  assert model_hash(m.memory_a)==gru_hash
  for k,v in m.state_dict().items():
   if k in before:assert torch.equal(v,before[k]),('Non-actor state changed',k)
  for n,model in frozen:assert model_hash(model)==frozen_hashes[n],('Frozen model changed',n)
  report=dict(round=ri+1,training_seed=args.seed+ri,teacher_action_fraction=mix,all_episode_successes=sum(t['success'] for t in terminals),all_episode_over100N=sum(t['peak_force_n']>100 for t in terminals),safe_episode_counts=safe_episode_counts,safe_sample_counts=sample_counts,aggregate_samples=len(Z),aggregate_case_samples=counts.tolist(),last_mean_squared_error=sum(losses[-64:])/min(64,len(losses)))
  reports.append(report)
  if not hasattr(runner,'logger_type'):runner.logger_type='tensorboard'
  runner.save(str(D/f'model_post_round{ri+1}.pt'))
  progress('round_completed',current_round=ri+1);print('SAFE_MIRROR_HEAD_ROUND',json.dumps(report),flush=True)
  del all_data,batches
 hook.remove();raw.training_teacher_mix=None;raw.mirror_policy=None;raw.training_teacher_policy=None
 for parameter in m.actor.parameters():
  for key in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:
   value=runner.alg.optimizer.state[parameter].get(key)
   if value is not None:value.zero_()
 m.memory_a.hidden_state=None;m.memory_c.hidden_state=None
 if not hasattr(runner,'logger_type'):runner.logger_type='tensorboard'
 cp=D/'fitted_post_actor.pt';runner.save(str(cp));torch.save(opt.state_dict(),D/'head_fit_optimizer.pt')
 result=dict(status='completed',method='balanced successful-native-episode DAgger, one actor MLP; fixed original GRU',checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),actor_inputs=33,actor_GRU_hash=gru_hash,GRU_critic_normalizers_exploration_unchanged=True,frozen_models_unchanged=frozen_hashes,teacher_absent_from_evaluation=True,physical_and_joint_limits_unchanged=True,rounds=reports,formal_acceptance=False)
 (D/'safe_mirror_fit_result.json').write_text(json.dumps(result,indent=2));progress('fit_completed',checkpoint=str(cp));print('SAFE_MIRROR_HEAD_FIT_COMPLETED',json.dumps(result),flush=True)
