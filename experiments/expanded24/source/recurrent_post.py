"""Full native episode BPTT; fixed qualified residual features and gated value."""
import copy,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from boundary_rl import model_hash
from brake_wrapper import BrakeHook
from recovery_storage import RecoveryStorage
from context_ppo import Shadow,metrics,predict,sha,utc
from guarded_value import fit_value,value_gate_passed,actor_permitted,optimizer_digest
from frozen_boundary_history import FrozenBoundaryHistory,RecurrentHook,precision,sequence_mean

def train(env,raw,boundary,logdir,args):
 assert args.seed==1099 and args.post_ppo_updates==4 and args.fixed_contact_cases and not args.initial_case_replay and not args.critic_shadow and not args.paired_diagnostic
 assert args.boundary_mode=='eval' and raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry and not args.alignment_axial_cap
 precision();D=Path(logdir);start=time.monotonic();model=boundary.base;source=Path(args.checkpoint);cp=torch.load(source,map_location=raw.device,weights_only=False);before={k:v.detach().clone() for k,v in model.state_dict().items()};assert all(torch.equal(v,cp['model_state_dict'][k]) for k,v in before.items())
 assert sha(source)=='d092d8c3dbb5a440d1f90f0e0c005bf1b754a5d3d7107234f7a50a13812abf1f'
 frozen={k:model_hash(v) for k,v in boundary.models().items() if k!='base'};boundary_before={k:v.detach().clone() for k,v in boundary.model.state_dict().items()}
 for p in model.parameters():p.requires_grad_(False)
 for n,p in boundary.model.named_parameters():p.requires_grad_(n.startswith('critic.'))
 bridge=FrozenBoundaryHistory(raw,boundary);hook=RecurrentHook(raw,boundary,bridge);shadow=Shadow(raw,boundary);brake=BrakeHook(raw,boundary,hook)
 params=[*model.memory_a.rnn.parameters(),*model.actor.parameters()]
 actor_opt=torch.optim.Adam([dict(params=list(model.memory_a.rnn.parameters()),lr=1e-6),dict(params=list(model.actor.parameters()),lr=1e-5)])
 value_opt=torch.optim.Adam(boundary.model.critic.parameters(),lr=1e-3);history=[];completed=0;value_fits=0;fit_attempts=0;gate=None;stop=None;storage=RecoveryStorage(D)
 def verify():
  assert sha(Path(__file__).parent/'retry_brake.py')=='c94e1fcee9ec3c400f9fec822e24c72b5c88703b539a5302054f5318b6181d05'
  bridge.verify();assert {k:model_hash(v) for k,v in boundary.models().items() if k!='base'}==frozen
  for k,v in boundary.model.state_dict().items():
   if not k.startswith('critic.'):assert torch.equal(v,boundary_before[k]),k
  for k,v in model.state_dict().items():
   if not k.startswith(('actor.','memory_a.rnn.')) or completed==0:assert torch.equal(v,before[k]),k
 def progress(phase,**kw):
  v=dict(server_utc=utc(),phase=phase,completed_updates=completed,value_fits=value_fits,value_fit_attempts=fit_attempts,collected_batches=len(history),target_updates=4,**kw);(D/'post_progress.json').write_text(json.dumps(v,indent=2));print('RECURRENT_POST',json.dumps(v),flush=True)
 audit=dict(source_sha256=sha(source),source_exact=True,root_lineage_only=True,algorithm='native complete-episode MC PPO; full GRU observation history from native reset; no detached recurrent feature input',boundary_feature_root_sha256=bridge.source_sha256,actor_gru_lr=1e-6,actor_mlp_lr=1e-5,value_lr=1e-3,sequence_minibatch_envs=16,actor_epochs=2,value_epochs=40,maximum_actor_updates=4,value_warmup_batches=2,sigma=hook.sigma.tolist(),gamma=.99,clip=.1,anchor=20,KL_stop=.01,gradient_norm_clip=.25,external_actor_inputs=33,actions=6,new_heads=0,frozen='post critic/memory_c/normalizers/std; other roles; boundary actor/std/scale; independent qualified feature history',diagnostic_data_forbidden=True,native_reward_unchanged=True,epoch0_candidate=True,per_actor_batch_fresh_value_gate=True,maximum_collected_batches=6,required_updates_for_qualification=4)
 audit.update(brake_source_sha256=sha(Path(__file__).parent/'retry_brake.py'),fixed_brake_mapping=True,brake_uses_measured33_and_native_history_only=True,brake_validation_group='reliability_v2_brake_global_20261017',latent_policy_likelihood_unchanged=True)
 (D/'post_resume_audit.json').write_text(json.dumps(audit,indent=2));torch.manual_seed(args.seed);env.seed(args.seed)
 for batch in range(6):
  if not storage.wait_space(batch,progress):stop='Storage reserve unavailable before a new batch;latest learner preserved';break
  torch.save(dict(actor=copy.deepcopy(model.actor.state_dict()),rnn=copy.deepcopy(model.memory_a.rnn.state_dict()),value=copy.deepcopy(boundary.model.critic.state_dict()),value_optimizer=copy.deepcopy(value_opt.state_dict()),actor_optimizer=copy.deepcopy(actor_opt.state_dict())),D/f'sequence_weights_before_batch{batch+1}.pt')
  actor_opt.zero_grad(set_to_none=True);actor_before=model_hash(model);actor_opt_before=optimizer_digest(actor_opt)
  model.memory_a.rnn.eval();obs,_=env.reset();boundary.reset(torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool));h0=torch.zeros(1,raw.num_envs,33,device=raw.device);assert model.memory_a.hidden_state is None or (model.memory_a.hidden_state==0).all();assert (bridge.h==0).all();assert not brake.brake.latched.any()
  active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool);hook.active=active.clone();records=[];outcomes={};previous_h=None;progress('collecting_value_warmup' if batch<2 else 'collecting_actor_batch',batch=batch+1)
  for tick in range(730):
   alive=active.clone()
   with torch.no_grad():
    action=boundary.act_inference(obs)
    if previous_h is not None:assert torch.equal(bridge.actor_h0[:,alive],previous_h[:,alive]),'Unexpected live-episode GRU reset'
    previous_h=model.memory_a.hidden_state.clone()
    obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
    for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
     if bool(alive[i]):outcomes[i]=dict(batch=batch+1,env=i,success=bool(term['success'][j]),end_reason=int(term['end_reason'][j]),peak_force_n=float(term['peak_force_n'][j]),seconds=float(term['seconds'][j]),attempts=int(term['recovery_attempts'][j]),realignments=int(term['recovery_realignments'][j]))
    records.append((hook.cache,shadow.feature.clone(),shadow.value.clone(),reward.detach().float().clone()*alive,done.clone(),alive,brake.cache.clone()));boundary.reset(done);active&=~done;hook.active=active.clone()
   if not active.any():break
  assert not active.any() and len(outcomes)==128
  rows=list(outcomes.values());over=sum(x['peak_force_n']>100 for x in rows)
  with (D/'post_episode_outcomes.jsonl').open('a') as f:
   for x in rows:f.write(json.dumps(x)+'\n')
  row=dict(batch=batch+1,kind='value_warmup' if batch<2 else 'actor',full_episodes=128,successes=sum(x['success'] for x in rows),over100N=over,total_native_steps=int(torch.stack([r[5] for r in records]).sum()),mean_seconds=sum(x['seconds'] for x in rows)/128,optimized=False,optimizer_steps=0,value_optimized=False,value_optimizer_steps=0,value_fit_attempted=False,actor_sha256_before=actor_before,actor_optimizer_sha256_before=actor_opt_before)
  X,A,LP,M,OLD,ROOT=[torch.stack([r[0][k] for r in records]) for k in range(6)];C=torch.stack([r[1] for r in records]);V=torch.stack([r[2] for r in records]);Alive=torch.stack([r[5] for r in records]);M&=Alive
  probe=torch.tensor([0,1,2,3,32,33,34,35],device=raw.device)
  np.savez_compressed(D/f'sequence_probe_batch{batch+1}.npz',normalized_observations=X[:,probe].cpu().numpy(),h0=h0[:,probe].cpu().numpy(),old_mean=OLD[:,probe].cpu().numpy(),root_mean=ROOT[:,probe].cpu().numpy(),latent=A[:,probe].cpu().numpy(),old_logp=LP[:,probe].cpu().numpy(),selected=M[:,probe].cpu().numpy(),alive=Alive[:,probe].cpu().numpy(),critic_features=C[:,probe].cpu().numpy(),prior_value=V[:,probe].cpu().numpy(),sigma=hook.sigma.cpu().numpy(),brake_state=torch.stack([r[6] for r in records])[:,probe].cpu().numpy(),brake_columns=np.array(brake.columns))
  brake_values=torch.stack([r[6] for r in records]);row.update(brake_selected_steps=int(((brake_values[:,:,7]>.5)&Alive).sum()),brake_triggered_episodes=int(((brake_values[:,:,7]>.5)&Alive).any(0).sum()))
  # This gate precedes every value or actor gradient, including warmup.
  if over>12:
   row.update(actor_sha256_after=model_hash(model),actor_optimizer_sha256_after=optimizer_digest(actor_opt));assert actor_before==row['actor_sha256_after'] and actor_opt_before==row['actor_optimizer_sha256_after'];stop='Force gate exceeded before actor or value gradients';history.append(row);storage.checkpoint(model,boundary,actor_opt,value_opt,completed,value_fits,fit_attempts,history,gate,raw);storage.compare_and_deduplicate(row,rows);break
  rewards=torch.stack([r[3] for r in records]);dones=torch.stack([r[4] for r in records]);returns=torch.zeros_like(rewards);future=torch.zeros(128,device=raw.device)
  for t in reversed(range(len(records))):future=rewards[t]+.99*future*(~dones[t]);returns[t]=future
  model.memory_a.rnn.train();actor_errors=[];root_errors=[]
  with torch.no_grad():
   for ids in torch.arange(128,device=raw.device).split(16):
    pred=sequence_mean(model,X[:,ids],h0[:,ids]);actor_errors.append(float(abs(pred-OLD[:,ids])[Alive[:,ids]].max()))
    y,_=bridge.rnn(X[:,ids],h0[:,ids]);anchor=bridge.actor(y.reshape(-1,33)).reshape(len(X),len(ids),6);root_errors.append(float(abs(anchor-ROOT[:,ids])[Alive[:,ids]].max()))
   critic_error=float(abs(predict(boundary.model,C[M])-V[M]).max())
  actor_error=max(actor_errors);root_error=max(root_errors);assert max(actor_error,root_error,critic_error)<2e-5,(actor_error,root_error,critic_error)
  eid=torch.arange(128,device=raw.device)[None,:].expand_as(M)[M];phase=torch.zeros_like(returns)
  for e in range(128):
   q=M[:,e].nonzero().flatten();phase[q,e]=torch.arange(len(q),device=raw.device)/max(len(q)-1,1)
  T=returns[M];cf=C[M];ph=phase[M];held=((eid//4)%4)==0;prior=metrics(V[M],T,ph,held);row.update(controlled_steps=int(M.sum()),full_sequence_replay_max_error=actor_error,frozen_anchor_replay_max_error=root_error,critic_feature_replay_max_error=critic_error,prior_value_metrics=prior,sequence_length=len(X),initial_history_zero=True)
  prediction_ok=value_gate_passed(prior);row['actor_value_gate_passed']=prediction_ok;row['actor_update_authorized']=actor_permitted(batch,prior)
  np.savez_compressed(D/f'value_gate_data_batch{batch+1}.npz',critic_features=cf[held].cpu().numpy(),prior_value=V[M][held].cpu().numpy(),target=T[held].cpu().numpy(),phase=ph[held].cpu().numpy(),env=eid[held].cpu().numpy())
  batch_gate=dict(batch=batch+1,passed=prediction_ok,metrics=prior,heldout_only_data=True,mse_ratio_limit=.9,variance_ratio_limit=.9,each_phase_mse_ratio_limit=1.05,predicted_before_any_fit_on_this_batch=True,actor_update_authorized=actor_permitted(batch,prior));(D/f'value_gate_batch{batch+1}.json').write_text(json.dumps(batch_gate,indent=2))
  if batch==1:
   gate=batch_gate;(D/'value_gate.json').write_text(json.dumps(gate,indent=2))
   if not prediction_ok:
    row.update(actor_sha256_after=model_hash(model),actor_optimizer_sha256_after=optimizer_digest(actor_opt));stop='Independent next-batch value gate failed;actor remains qualified root';history.append(row);storage.checkpoint(model,boundary,actor_opt,value_opt,completed,value_fits,fit_attempts,history,gate,raw);storage.compare_and_deduplicate(row,rows);break
  if batch>=2 and not prediction_ok:row['kind']='value_calibration'
  ADV=returns-V;ADV=(ADV-T.sub(V[M]).mean())/T.sub(V[M]).std(unbiased=False).clamp_min(.05)
  if actor_permitted(batch,prior):
   assert gate['passed']
   for p in params:p.requires_grad_(True)
   # Actual native early/middle/late samples must reach the existing GRU.
   coverage={};ids=torch.arange(16,device=raw.device)
   for label,q in [('early',phase<=1/3),('middle',(phase>1/3)&(phase<2/3)),('late',phase>=2/3)]:
    mm=M[:,ids]&q[:,ids];assert mm.any();mu=sequence_mean(model,X[:,ids],h0[:,ids]);loss=-(Normal(mu,hook.sigma).log_prob(A[:,ids]).sum(-1)*ADV[:,ids])[mm].mean();gg=torch.autograd.grad(loss,params);g=sum(float(a.abs().sum()) for a in gg[:len(list(model.memory_a.rnn.parameters()))]);assert g>0;coverage[label]=g
   losses=[];early=False
   for epoch in range(2):
    for ids in torch.randperm(128,device=raw.device).split(16):
     mm=M[:,ids];assert mm.any();mu=sequence_mean(model,X[:,ids],h0[:,ids]);lr=Normal(mu,hook.sigma).log_prob(A[:,ids]).sum(-1)-LP[:,ids];ratio=lr.exp();kl=float(((ratio-1-lr)[mm]).mean().detach())
     if kl>.01:early=True;break
     pg=torch.maximum(-ADV[:,ids]*ratio,-ADV[:,ids]*ratio.clamp(.9,1.1))[mm].mean();anchor=(mu-ROOT[:,ids]).square().mean(-1)[mm].mean();loss=pg+20*anchor;assert torch.isfinite(loss);actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,.25);actor_opt.step();losses.append(dict(policy=float(pg.detach()),anchor=float(anchor.detach()),kl=kl))
    if early:break
   assert losses;completed+=1;row.update(optimized=True,optimizer_steps=len(losses),early_KL_stop=early,last_loss=losses[-1],actual_GRU_gradient_coverage=coverage)
  model.memory_a.rnn.eval();fit=fit_value(boundary.model,value_opt,cf,T,eid,ph);fit_attempts+=1;value_fits+=int(fit['retained_fit']);row.update(value_optimized=fit['retained_fit'],value_optimizer_steps=fit['optimizer_steps'],value_fit=fit,value_fit_attempted=True,actor_sha256_after=model_hash(model),actor_optimizer_sha256_after=optimizer_digest(actor_opt))
  if not row['optimized']:assert actor_before==row['actor_sha256_after'] and actor_opt_before==row['actor_optimizer_sha256_after']
  torch.save(dict(critic=copy.deepcopy(boundary.model.critic.state_dict()),optimizer=copy.deepcopy(value_opt.state_dict())),D/f'value_fit_retained_batch{batch+1}.pt')
  if batch==0:
   torch.save(dict(critic=boundary.model.critic.state_dict(),optimizer=value_opt.state_dict()),D/'value_warmup_critic.pt');np.savez_compressed(D/'value_fit_data_batch1.npz',critic_features=cf.cpu().numpy(),target=T.cpu().numpy(),phase=ph.cpu().numpy(),env=eid.cpu().numpy())
  verify();history.append(row);storage.checkpoint(model,boundary,actor_opt,value_opt,completed,value_fits,fit_attempts,history,gate,raw);storage.compare_and_deduplicate(row,rows);(D/'post_training_history.json').write_text(json.dumps(dict(batches=history),indent=2));progress('batch_completed',latest=row)
 verify();cp['model_state_dict']={k:v.detach().clone() for k,v in model.state_dict().items()}
 if 'optimizer_state_dict' in cp:cp['optimizer_state_dict']['state']={}
 cp['full_recurrent_post_lineage']=dict(source_sha256=sha(source),seed=1099,completed_actor_updates=completed,value_fits=value_fits,value_fit_attempts=fit_attempts,boundary_feature_root_checkpoint=str(bridge.source),boundary_feature_root_sha256=bridge.source_sha256,complete_sequence_from_native_reset=True)
 out=D/'post';out.mkdir();path=out/'model_post_recurrent.pt';torch.save(cp,path);shutil.copy2(source.parent/'agent.json',out/'agent.json');torch.save(actor_opt.state_dict(),D/'actor_optimizer.pt');torch.save(value_opt.state_dict(),D/'value_optimizer.pt');torch.save(dict(state_dict=boundary.model.state_dict(),source_sha256=sha(args.boundary_checkpoint),value_output_scale=100.,boundary_actor_frozen=True),D/'boundary_value.pt')
 result=dict(status=('completed' if completed==4 else 'stopped_budget') if not stop else 'stopped_safety' if 'Force gate' in stop else 'stopped_storage' if stop.startswith('Storage') else 'stopped_value_gate',stop_reason=stop,completed_updates=completed,completed_value_fits=value_fits,completed_value_fit_attempts=fit_attempts,full_episodes=sum(x['full_episodes'] for x in history),batches=history,value_gate=gate,checkpoint=str(path),checkpoint_sha256=sha(path),source_sha256=sha(source),changed_tensor_maximums={k:float(abs(v-before[k]).max()) for k,v in model.state_dict().items() if not torch.equal(v,before[k])},boundary_critic_changes={k:float(abs(v-boundary_before[k]).max()) for k,v in boundary.model.state_dict().items() if not torch.equal(v,boundary_before[k])},frozen_root_feature_hashes=bridge.hashes,native_reward_and_gates_unchanged=True,wall_seconds=time.monotonic()-start)
 (D/'post_train_result.json').write_text(json.dumps(result,indent=2));progress(result['status'],checkpoint_sha256=result['checkpoint_sha256'])
