"""Native full-episode PPO for the existing deep MLP with public joint5 input."""
import copy,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from boundary_rl import model_hash
from context_ppo import Shadow,metrics,predict,sha,utc
from guarded_value import fit_value,value_gate_passed,actor_permitted,optimizer_digest
from frozen_boundary_history import FrozenBoundaryHistory,precision
from joint_input_upgrade import expose_joint5
from deep_joint_hook import DeepHook
from joint_storage import durable,MINFREE,cpu
from brake_wrapper import BrakeHook
from native_groups import combine,compare_first

def train(env,raw,boundary,logdir,args):
 assert args.seed==1119 and args.deep_ppo_updates==4 and args.joint5_input and args.fixed_contact_cases
 assert not any([args.initial_case_replay,args.post_ppo_updates,args.critic_shadow,args.paired_diagnostic,args.withdrawal_guard if hasattr(args,'withdrawal_guard') else False,args.fixed_final_retry_brake,args.independent_boundary_history])
 assert args.boundary_mode=='eval' and raw.num_envs==128 and raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry
 precision();D=Path(logdir);start=time.monotonic();source=Path(args.deep_recovery_checkpoint);source_sha=sha(source);assert source_sha=='055571a29e42fc6e36b347823cb5aa57e6ea1505af6fb59226ab5108d33eabdf';model=raw.deep_recovery_model;cp=torch.load(source,map_location=raw.device,weights_only=False)
 assert all(torch.equal(v,cp['model_state_dict'][k]) for k,v in model.state_dict().items())
 root=copy.deepcopy(model).eval()
 for p in root.parameters():p.requires_grad_(False)
 expose_joint5(model);initial={k:v.detach().clone() for k,v in model.state_dict().items()};frozen={n:model_hash(m) for n,m in boundary.models().items() if n!='deep_recovery_model'};boundary_initial=cpu(boundary.model.state_dict())
 for p in model.parameters():p.requires_grad_(False)
 for n,p in boundary.model.named_parameters():p.requires_grad_(n.startswith('critic.'))
 bridge=FrozenBoundaryHistory(raw,boundary);shadow=Shadow(raw,boundary);hook=DeepHook(raw,boundary,root);brake=BrakeHook(raw,boundary,hook);raw.fixed_final_retry_hook=brake;params=list(model.actor.parameters());actor_opt=torch.optim.Adam(params,lr=1e-5);value_opt=torch.optim.Adam(boundary.model.critic.parameters(),lr=1e-3);history=[];completed=0;fits=0;attempts=0;gate=None;stop=None
 def verify():
  bridge.verify();assert sha(source)==source_sha;assert {n:model_hash(m) for n,m in boundary.models().items() if n!='deep_recovery_model'}==frozen
  for k,v in model.state_dict().items():
   if not k.startswith('actor.') or completed==0:assert torch.equal(v,initial[k]),k
  for k,v in boundary.model.state_dict().items():
   if not k.startswith('critic.'):assert torch.equal(v.cpu(),boundary_initial[k]),k
 def progress(phase,**kw):
  v=dict(server_utc=utc(),phase=phase,completed_updates=completed,value_fits=fits,value_fit_attempts=attempts,collected_batches=len(history),**kw);(D/'deep_progress.json').write_text(json.dumps(v,indent=2));print('DEEP_JOINT_PPO',json.dumps(v),flush=True)
 audit=dict(source_sha256=source_sha,initial_feature_mask=[2,3,4,5,10,27,28,29,30,31,32],intentional_initial_changes=['mask.index10=1','unused actor.0.weight column10=0'],zero_initial_policy_change_proven_by='reliability_v2_joint_native_20261017',algorithm='selected-deep latent Gaussian full native episode MonteCarlo PPO;existing current observation MLP,no GRU gradients',actor_lr=1e-5,value_lr=1e-3,sigma=hook.sigma.tolist(),actor_epochs=2,env_minibatch=32,cohorts_per_learning_group=2,episodes_per_group=256,physical_vector_width=128,maximum_native_cohorts=12,maximum_native_episodes=1536,probe_rule="fixed8plus up to4 most-exposed heldout envs,exposure-only,no outcome selection",empty_selected_env_groups_skipped=True,anchor=20,root_anchor_feature_indices=[2,3,4,5,27,28,29,30,31,32],gamma=.99,clip=.1,KL_stop=.01,grad_norm_clip=.25,cold_new_actor_and_value_optimizers=True,all_native_reward_physics_success_rules_unchanged=True,all_GRUs_normalizers_std_and_other_skills_frozen=True,diagnostic_data_forbidden=True,maximum_batches=6,maximum_updates=4)
 audit.update(fixed_final_retry_brake=True,brake_source_sha256=sha(Path(__file__).parent/'retry_brake.py'),validation_group='reliability_v2_deep_brake_pair_20261017',first1117unsafe_run_not_resumed=True)
 (D/'deep_resume_audit.json').write_text(json.dumps(audit,indent=2));torch.manual_seed(1119);env.seed(1119)
 for batch in range(6):
  if shutil.disk_usage(D).free<=MINFREE+32*1024**2:stop='Storage reserve unavailable before new batch';break
  torch.save(dict(deep_state_dict=cpu(model.state_dict()),critic=cpu(boundary.model.critic.state_dict()),actor_optimizer=cpu(actor_opt.state_dict()),value_optimizer=cpu(value_opt.state_dict())),D/f'weights_before_batch{batch+1}.pt')
  actor_before=model_hash(model);opt_before=optimizer_digest(actor_opt);actor_opt.zero_grad(set_to_none=True);units=[];unit_stats=[];all_rows=[]
  for unit in range(2):
   obs,_=env.reset();boundary.reset(torch.ones(128,dtype=torch.bool,device=raw.device));assert not bridge.h.any();h=model.memory_a.hidden_state;assert h is None or not h.any();active=torch.ones(128,dtype=torch.bool,device=raw.device);hook.active=active.clone();records=[];ends={};progress('collecting_value_warmup' if batch<2 else 'collecting_actor_batch',batch=batch+1,unit=unit+1,cohorts_per_group=2)
   for tick in range(730):
    alive=active.clone()
    with torch.no_grad():
     action=boundary.act_inference(obs);obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
     for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
      if bool(alive[i]):ends[i]=dict(batch=batch+1,env=i,success=bool(term['success'][j]),end_reason=int(term['end_reason'][j]),peak_force_n=float(term['peak_force_n'][j]),seconds=float(term['seconds'][j]),attempts=int(term['recovery_attempts'][j]),exits=int(term['recovery_exits'][j]),realignments=int(term['recovery_realignments'][j]),end_mode=int(term['recovery_end_mode'][j]))
     records.append((hook.cache,shadow.feature.clone(),shadow.value.clone(),reward.float().clone()*alive,done.clone(),alive,boundary.public.clone(),hook.compose_cache,hook.compose_cache['clipped'],hook.compose_cache['delayed']))
     boundary.reset(done);active&=~done;hook.active=active.clone()
    if not active.any():break
   assert not active.any() and len(ends)==128
   rows=[dict(ends[i],unit=unit+1,native_env=i,env=unit*128+i) for i in range(128)];count=sum(x['peak_force_n']>100 for x in rows);counts=torch.stack([r[0]['selected']&r[5] for r in records]).sum(0).cpu().tolist();unit_stats.append(dict(unit=unit+1,episodes=128,successes=sum(r['success'] for r in rows),over100N=count,controlled_steps=sum(counts),selected_steps_by_native_env=counts,native_ticks=len(records)));units.append(records);all_rows.extend(rows)
   if batch==0 and unit==0:
    if not compare_first(records,rows,hook,brake,D,actor_before,opt_before):stop='Replication check failed before gradients';break
   if count>12:stop='Force gate exceeded in native128 cohort before all group gradients';break
  records=combine(units);EPISODES=128*len(units);rows=all_rows;over=sum(x['peak_force_n']>100 for x in rows)
  with (D/'deep_episode_outcomes.jsonl').open('a') as f:
   for r in rows:f.write(json.dumps(r)+'\n')
  K={k:torch.stack([r[0][k] for r in records]) for k in records[0][0]};C=torch.stack([r[1] for r in records]);V=torch.stack([r[2] for r in records]);Alive=torch.stack([r[5] for r in records]);M=K['selected']&Alive;phase=torch.zeros_like(V)
  for e in range(EPISODES):
   q=M[:,e].nonzero().flatten();phase[q,e]=torch.arange(len(q),device=raw.device)/max(len(q)-1,1)
  probe_ids=[]
  for unit,info in enumerate(unit_stats):
   base_probe=[0,1,2,3,32,33,34,35];counts=info['selected_steps_by_native_env'];eligible=[e for e in range(128) if ((e//4)%4)==0 and counts[e]>0];extra=sorted(eligible,key=lambda e:(-counts[e],e))[:4];probe_ids.extend(unit*128+e for e in base_probe+[e for e in extra if e not in base_probe])
  probe=torch.tensor(probe_ids,device=raw.device);public=torch.stack([r[6] for r in records]);shared=torch.equal(public,K['public'])
  arrays=dict(public33=public[:,probe].cpu().numpy(),deep_features=K['features'][:,probe].cpu().numpy(),old_mean=K['mean'][:,probe].cpu().numpy(),root_mean=K['root'][:,probe].cpu().numpy(),latent=K['latent'][:,probe].cpu().numpy(),old_logp=K['logp'][:,probe].cpu().numpy(),selected=M[:,probe].cpu().numpy(),native=K['native'][:,probe].cpu().numpy(),alive=Alive[:,probe].cpu().numpy(),critic_features=C[:,probe].cpu().numpy(),prior_value=V[:,probe].cpu().numpy(),reward=torch.stack([r[3] for r in records])[:,probe].cpu().numpy(),done=torch.stack([r[4] for r in records])[:,probe].cpu().numpy(),phase=phase[:,probe].cpu().numpy(),before_brake=torch.stack([r[7]['before_brake'] for r in records])[:,probe].cpu().numpy(),brake_state=torch.stack([r[7]['brake_state'] for r in records])[:,probe].cpu().numpy(),brake_columns=np.array(brake.columns),composed=torch.stack([r[7]['out'] for r in records])[:,probe].cpu().numpy(),base=torch.stack([r[7]['base'] for r in records])[:,probe].cpu().numpy(),delta=torch.stack([r[7]['delta'] for r in records])[:,probe].cpu().numpy(),boundary_mask=torch.stack([r[7]['mask'] for r in records])[:,probe].cpu().numpy(),clipped=torch.stack([r[8] for r in records])[:,probe].cpu().numpy(),delayed=torch.stack([r[9] for r in records])[:,probe].cpu().numpy(),delay_steps=torch.stack([r[7]['delay'] for r in records])[:,probe].cpu().numpy(),refresh=torch.stack([r[7]['refresh'] for r in records])[:,probe].cpu().numpy(),sigma=hook.sigma.cpu().numpy(),env=probe.cpu().numpy(),public_shared=np.array(shared))
  if not shared:arrays['deep_public33']=K['public'][:,probe].cpu().numpy()
  np.savez_compressed(D/f'native_probes_batch{batch+1}.npz',**arrays)
  row=dict(batch=batch+1,kind='value_warmup' if batch<2 else 'actor',full_episodes=EPISODES,cohorts=unit_stats,physical_envs=128,successes=sum(x['success'] for x in rows),over100N=over,mean_seconds=sum(x['seconds'] for x in rows)/EPISODES,controlled_steps=int(M.sum()),total_native_steps=int(Alive.sum()),optimized=False,value_fit_attempted=False,value_optimized=False,optimizer_steps=0,actor_sha256_before=actor_before,actor_optimizer_sha256_before=opt_before)
  def finish_row():
   row.update(actor_sha256_after=model_hash(model),actor_optimizer_sha256_after=optimizer_digest(actor_opt))
   if not row['optimized']:assert actor_before==row['actor_sha256_after'] and opt_before==row['actor_optimizer_sha256_after']
   verify();history.append(row);durable(D,model,boundary,actor_opt,value_opt,history,gate,completed,fits,attempts,raw);(D/'deep_training_history.json').write_text(json.dumps(dict(batches=history),indent=2));progress('batch_completed',latest=row)
  if any(u['over100N']>12 for u in unit_stats):stop='Force gate exceeded in native128 cohort before all group gradients';finish_row();break
  if stop and stop.startswith('Replication'):finish_row();break
  rewards=torch.stack([r[3] for r in records]);dones=torch.stack([r[4] for r in records]);returns=torch.zeros_like(rewards);future=torch.zeros(EPISODES,device=raw.device)
  for t in reversed(range(len(records))):future=rewards[t]+.99*future*(~dones[t]);returns[t]=future
  eid=torch.arange(EPISODES,device=raw.device)[None,:].expand_as(M)[M];held=((eid//4)%4)==0
  if int(held.sum())<=200 or int((~held).sum())<=500:stop='Insufficient native selected-deep heldout/train coverage before gradients';row.update(heldout_steps=int(held.sum()),train_steps=int((~held).sum()));finish_row();break
  X=K['features'];OLD=K['mean'];ROOT=K['root'];A=K['latent'];LP=K['logp'];cf=C[M];T=returns[M];ph=phase[M]
  with torch.no_grad():
   predicted=model.actor(X.reshape(-1,33)).reshape(*X.shape[:2],6);anchor=root.actor(root.current_features(root.actor_obs_normalizer(K['public'].reshape(-1,33)))).reshape(*OLD.shape);actor_err=float(abs(predicted-OLD)[M].max());anchor_err=float(abs(anchor-ROOT)[M].max());critic_err=float(abs(predict(boundary.model,cf)-V[M]).max())
  assert max(actor_err,anchor_err,critic_err)<2e-5,(actor_err,anchor_err,critic_err)
  prior=metrics(V[M],T,ph,held);passed=value_gate_passed(prior);permitted=actor_permitted(batch,prior);row.update(prior_value_metrics=prior,actor_value_gate_passed=passed,actor_update_authorized=permitted,actor_replay_error=actor_err,original10_anchor_replay_error=anchor_err,critic_replay_error=critic_err)
  batch_gate=dict(batch=batch+1,passed=passed,actor_update_authorized=permitted,metrics=prior,predicted_before_any_fit_on_this_batch=True);(D/f'value_gate_batch{batch+1}.json').write_text(json.dumps(batch_gate,indent=2));np.savez_compressed(D/f'value_gate_data_batch{batch+1}.npz',critic_features=cf[held].cpu().numpy(),prior_value=V[M][held].cpu().numpy(),target=T[held].cpu().numpy(),phase=ph[held].cpu().numpy(),env=eid[held].cpu().numpy())
  if batch==1:
   gate=batch_gate
   if not passed:stop='Independent second-batch prefit value gate failed;actor unchanged';finish_row();break
  if batch>=2 and not passed:row['kind']='value_calibration'
  ADV=returns-V;adv_selected=T-V[M];ADV=(ADV-adv_selected.mean())/adv_selected.std(unbiased=False).clamp_min(.05)
  if permitted:
   assert gate['passed']
   for p in params:p.requires_grad_(True)
   coverage={}
   for name,mask in [('early',phase<=1/3),('middle',(phase>1/3)&(phase<2/3)),('late',phase>=2/3)]:
    q=M&mask;assert q.any();mu=model.actor(X[q]);loss=-(Normal(mu,hook.sigma).log_prob(A[q]).sum(-1)*ADV[q]).mean();grads=torch.autograd.grad(loss,params);coverage[name]=dict(MLP_gradient_l1=sum(float(g.abs().sum()) for g in grads),new_joint5_column_gradient_l1=float(grads[0][:,10].abs().sum()));assert coverage[name]['MLP_gradient_l1']>0
   assert sum(x['new_joint5_column_gradient_l1'] for x in coverage.values())>0
   losses=[];early=False
   for epoch in range(2):
    for ids in torch.randperm(EPISODES,device=raw.device).split(32):
     mm=M[:,ids]
     if not mm.any():continue
     mu=model.actor(X[:,ids].reshape(-1,33)).reshape(len(X),len(ids),6);lr=Normal(mu,hook.sigma).log_prob(A[:,ids]).sum(-1)-LP[:,ids];ratio=lr.exp();kl=float((ratio-1-lr)[mm].mean().detach())
     if kl>.01:early=True;break
     pg=torch.maximum(-ADV[:,ids]*ratio,-ADV[:,ids]*ratio.clamp(.9,1.1))[mm].mean();anchor=(mu-ROOT[:,ids]).square().mean(-1)[mm].mean();loss=pg+20*anchor;assert torch.isfinite(loss);actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,.25);actor_opt.step();losses.append(dict(policy=float(pg.detach()),anchor=float(anchor.detach()),kl=kl))
    if early:break
   assert losses;completed+=1;row.update(optimized=True,optimizer_steps=len(losses),early_KL_stop=early,last_loss=losses[-1],actual_MLP_gradient_coverage=coverage)
  fit=fit_value(boundary.model,value_opt,cf,T,eid,ph);attempts+=1;fits+=int(fit['retained_fit']);row.update(value_fit_attempted=True,value_optimized=fit['retained_fit'],value_fit=fit,value_optimizer_steps=fit['optimizer_steps']);torch.save(dict(critic=cpu(boundary.model.critic.state_dict()),optimizer=cpu(value_opt.state_dict())),D/f'value_fit_retained_batch{batch+1}.pt')
  if batch==0:np.savez_compressed(D/'value_fit_data_batch1.npz',critic_features=cf.cpu().numpy(),target=T.cpu().numpy(),phase=ph.cpu().numpy(),env=eid.cpu().numpy())
  finish_row()
 assert not brake.brake.latched.any()
 verify();out=D/'deep';out.mkdir();cp['model_state_dict']=cpu(model.state_dict())
 if 'optimizer_state_dict' in cp:cp['optimizer_state_dict']['state']={}
 cp['joint5_deep_PPO_lineage']=dict(source_sha256=source_sha,seed=1119,completed_updates=completed,value_fits=fits,fit_attempts=attempts,original10_feature_anchor=True,intentional_initial_zero_contribution_input_upgrade=True)
 path=out/'model_deep_joint.pt';torch.save(cp,path);shutil.copy2(source.parent/'agent.json',out/'agent.json');torch.save(cpu(actor_opt.state_dict()),D/'actor_optimizer.pt');torch.save(cpu(value_opt.state_dict()),D/'value_optimizer.pt');torch.save(dict(state_dict=cpu(boundary.model.state_dict()),source_sha256=sha(args.boundary_checkpoint),boundary_actor_frozen=True),D/'boundary_value.pt')
 result=dict(status=('completed' if completed==4 else 'stopped_budget') if not stop else 'stopped_safety' if stop.startswith('Force') else 'stopped_storage' if stop.startswith('Storage') else 'stopped_coverage' if stop.startswith('Insufficient') else 'stopped_replication' if stop.startswith('Replication') else 'stopped_value_gate',stop_reason=stop,completed_updates=completed,completed_value_fits=fits,completed_value_fit_attempts=attempts,full_episodes=sum(r['full_episodes'] for r in history),batches=history,value_gate=gate,checkpoint=str(path),checkpoint_sha256=sha(path),source_sha256=source_sha,learned_actor_changes_after_upgrade={k:float(abs(v-initial[k]).max()) for k,v in model.state_dict().items() if not torch.equal(v,initial[k])},boundary_critic_changes={k:float(abs(v.cpu()-boundary_initial[k]).max()) for k,v in boundary.model.state_dict().items() if not torch.equal(v.cpu(),boundary_initial[k])},new_joint_column_max_abs=float(model.actor[0].weight[:,10].abs().max()),fixed_final_retry_brake=True,brake_source_sha256=sha(Path(__file__).parent/'retry_brake.py'),native_rules_unchanged=True,wall_seconds=time.monotonic()-start)
 (D/'deep_train_result.json').write_text(json.dumps(result,indent=2));progress(result['status'],checkpoint_sha256=result['checkpoint_sha256'])
