"""Native independent critic history, value-only warmup and gated post PPO."""
import copy,datetime as dt,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from boundary_rl import model_hash
from post_ppo import Hook

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

class Shadow:
 def __init__(self,raw,boundary):
  self.model=boundary.base;self.feature=torch.zeros(raw.num_envs,33,device=raw.device);self.value=torch.zeros(raw.num_envs,device=raw.device)
  def capture(module,inputs,output):self.feature=inputs[0].detach().clone();self.value=output.detach().flatten().clone()
  self.model.critic.register_forward_hook(capture);original=boundary.act_inference
  def act(obs):
   with torch.no_grad():self.model.evaluate(obs)
   return original(obs)
  boundary.act_inference=act;raw.critic_shadow=self

def metrics(pred,target,phase,mask):
 def report(q):
  y=target[q];e=y-pred[q];const=float((y-y.mean()).square().mean());return dict(samples=int(q.sum()),mse=float(e.square().mean()),constant_mse=const,mse_ratio=float(e.square().mean())/max(const,1e-6),residual_variance_ratio=float(e.var(unbiased=False))/max(const,1e-6),bias=float(e.mean()))
 out=report(mask);out['phases']={n:report(mask&q) for n,q in [('early',phase<=1/3),('middle',(phase>1/3)&(phase<2/3)),('late',phase>=2/3)]};return out

def fit_value(model,opt,C,T,eid,phase,epochs=40):
 held=((eid//4)%4)==0;train=~held;assert int(held.sum())>200 and int(train.sum())>500
 with torch.no_grad():before=metrics(model.critic(C).flatten(),T,phase,held)
 best=float('inf');state=None;steps=0
 for epoch in range(epochs):
  ids=train.nonzero().flatten();ids=ids[torch.randperm(len(ids),device=C.device)]
  for q in ids.split(1024):
   prediction=model.critic(C[q]).flatten();loss=((prediction-T[q])/100.).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.critic.parameters(),1.);opt.step();steps+=1
  with torch.no_grad():score=float((model.critic(C[held]).flatten()-T[held]).square().mean())
  if score<best:best=score;state=copy.deepcopy(model.critic.state_dict());moment=copy.deepcopy(opt.state_dict());best_epoch=epoch+1
 model.critic.load_state_dict(state);opt.load_state_dict(moment)
 with torch.no_grad():after=metrics(model.critic(C).flatten(),T,phase,held)
 return dict(before=before,after=after,best_epoch=best_epoch,optimizer_steps=steps,epochs=epochs,heldout_rule='env//4 modulo4 equals0; all four contact cases retained in both splits')

def train(env,raw,boundary,logdir,args):
 assert args.seed==1071 and args.post_ppo_updates==4 and args.fixed_contact_cases and not args.initial_case_replay and not args.critic_shadow
 assert args.boundary_mode=='eval' and raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry and not args.alignment_axial_cap
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 D=Path(logdir);start=time.monotonic();model=boundary.base;source=Path(args.checkpoint);cp=torch.load(source,map_location=raw.device,weights_only=False);before={k:v.detach().clone() for k,v in model.state_dict().items()};assert all(torch.equal(v,cp['model_state_dict'][k]) for k,v in before.items())
 frozen={k:model_hash(v) for k,v in boundary.models().items() if k!='base'};residual_hash=model_hash(boundary.model);root=copy.deepcopy(model.actor).eval()
 for p in root.parameters():p.requires_grad_(False)
 for n,p in model.named_parameters():p.requires_grad_(n.startswith('critic.'))
 hook=Hook(raw,boundary);shadow=Shadow(raw,boundary);actor_opt=torch.optim.Adam(model.actor.parameters(),lr=1e-5);value_opt=torch.optim.Adam(model.critic.parameters(),lr=1e-3)
 history=[];completed=0;value_fits=0;gate=None;stop=None
 def verify():
  assert {k:model_hash(v) for k,v in boundary.models().items() if k!='base'}==frozen and model_hash(boundary.model)==residual_hash
  for k,v in model.state_dict().items():
   if not k.startswith(('actor.','critic.')):assert torch.equal(v,before[k]),k
   if completed==0 and k.startswith('actor.'):assert torch.equal(v,before[k]),k
 def progress(phase,**kw):
  v=dict(server_utc=utc(),phase=phase,completed_updates=completed,value_fits=value_fits,collected_batches=len(history),target_updates=4,**kw);(D/'post_progress.json').write_text(json.dumps(v,indent=2));print('VALUE_PPO',json.dumps(v),flush=True)
 (D/'post_resume_audit.json').write_text(json.dumps(dict(source_sha256=sha(source),actor_source_is_qualified=True,source_weights_exact=True,value_only_warmup_batches=2,maximum_actor_updates=4,critic_history='native independent memory_c with normalizer frozen and native done resets',architecture_unchanged=True,actor_lr=1e-5,value_lr=1e-3,value_epochs=40,actor_optimizer_new=True,value_optimizer_new=True,prior_value_predictions_used=True,native_reward_unchanged=True,case_replay_forbidden=True),indent=2))
 torch.manual_seed(args.seed);env.seed(args.seed)
 for batch in range(6):
  assert shutil.disk_usage(D).free>5045649408+32*1024**2
  obs,_=env.reset();boundary.reset(torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool));active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool);hook.active=active.clone();records=[];outcomes={};progress('collecting_value_warmup' if batch<2 else 'collecting_actor_batch',batch=batch+1)
  for tick in range(730):
   alive=active.clone()
   with torch.no_grad():
    action=boundary.act_inference(obs);obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
    for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
     if bool(alive[i]):outcomes[i]=dict(batch=batch+1,env=i,success=bool(term['success'][j]),end_reason=int(term['end_reason'][j]),peak_force_n=float(term['peak_force_n'][j]),seconds=float(term['seconds'][j]),attempts=int(term['recovery_attempts'][j]),realignments=int(term['recovery_realignments'][j]))
    records.append((hook.cache,shadow.feature.clone(),shadow.value.clone(),reward.detach().float().clone()*alive,done.clone(),alive));boundary.reset(done);active&=~done;hook.active=active.clone()
   if not active.any():break
  assert not active.any() and len(outcomes)==raw.num_envs
  rows=list(outcomes.values());over=sum(x['peak_force_n']>100 for x in rows)
  with (D/'post_episode_outcomes.jsonl').open('a') as f:
   for x in rows:f.write(json.dumps(x)+'\n')
  row=dict(batch=batch+1,kind='value_warmup' if batch<2 else 'actor',full_episodes=len(rows),successes=sum(x['success'] for x in rows),over100N=over,total_native_steps=int(torch.stack([r[5] for r in records]).sum()),mean_seconds=sum(x['seconds'] for x in rows)/len(rows),optimized=False,optimizer_steps=0,value_optimized=False,value_optimizer_steps=0)
  if over>raw.num_envs*.10:stop='Force gate exceeded before actor or value gradients';history.append(row);break
  rewards=torch.stack([r[3] for r in records]);done=torch.stack([r[4] for r in records]);returns=torch.zeros_like(rewards);future=torch.zeros(raw.num_envs,device=raw.device)
  for t in reversed(range(len(records))):future=rewards[t]+.99*future*(~done[t]);returns[t]=future
  fs=[];cs=[];aa=[];ll=[];tt=[];mm=[];vv=[];ee=[]
  for t,r in enumerate(records):
   f,a,l,mask,mu=r[0];mask=mask&r[5]
   if mask.any():fs.append(f[mask]);cs.append(r[1][mask]);aa.append(a[mask]);ll.append(l[mask]);tt.append(returns[t][mask]);mm.append(mu[mask]);vv.append(r[2][mask]);ee.append(mask.nonzero().flatten())
  F,C,A,LP,T,OLD,V,eid=map(torch.cat,(fs,cs,aa,ll,tt,mm,vv,ee));phase=torch.empty_like(T)
  for e in eid.unique():
   q=(eid==e).nonzero().flatten();phase[q]=torch.arange(len(q),device=raw.device)/max(len(q)-1,1)
  with torch.no_grad():actor_err=float((model.actor(F)-OLD).abs().max());critic_err=float((model.critic(C).flatten()-V).abs().max())
  assert actor_err<2e-5 and critic_err<2e-5,(actor_err,critic_err)
  held=((eid//4)%4)==0;prior=metrics(V,T,phase,held);row.update(controlled_steps=len(F),feature_replay_max_error=actor_err,critic_feature_replay_max_error=critic_err,prior_value_metrics=prior)
  if batch==1:
   np.savez_compressed(D/'value_gate_data.npz',critic_features=C.cpu().numpy(),prior_value=V.cpu().numpy(),target=T.cpu().numpy(),phase=phase.cpu().numpy(),env=eid.cpu().numpy())
   passed=prior['mse_ratio']<=.90 and prior['residual_variance_ratio']<=.90 and all(v['mse_ratio']<=1.05 for v in prior['phases'].values())
   gate=dict(passed=passed,kind='fresh second native batch predicted before any fit on that batch',metrics=prior,mse_ratio_limit=.90,variance_ratio_limit=.90,each_phase_mse_ratio_limit=1.05);(D/'value_gate.json').write_text(json.dumps(gate,indent=2))
   if not passed:stop='Independent next-batch value gate failed;actor remains original qualified';history.append(row);break
  adv=T-V;adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(.05)
  ids=torch.arange(0,len(F),max(1,len(F)//256),device=raw.device)[:256];np.savez_compressed(D/f'value_probe_batch{batch+1}.npz',actor_features=F[ids].cpu().numpy(),critic_features=C[ids].cpu().numpy(),latent=A[ids].cpu().numpy(),old_mean=OLD[ids].cpu().numpy(),old_logp=LP[ids].cpu().numpy(),returns=T[ids].cpu().numpy(),prior_value=V[ids].cpu().numpy(),advantage=adv[ids].cpu().numpy(),sigma=hook.sigma.cpu().numpy())
  if batch>=2:
   assert gate['passed']
   for p in model.actor.parameters():p.requires_grad_(True)
   with torch.no_grad():anchor_mu=root(F)
   losses=[];early=False
   for epoch in range(2):
    for q in torch.randperm(len(F),device=raw.device).split(2048):
     mu=model.actor(F[q]);lr=Normal(mu,hook.sigma).log_prob(A[q]).sum(-1)-LP[q];ratio=lr.exp();pg=torch.maximum(-adv[q]*ratio,-adv[q]*ratio.clamp(.9,1.1)).mean();anchor=(mu-anchor_mu[q]).square().mean();loss=pg+20*anchor;assert torch.isfinite(loss);actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.actor.parameters(),.25);actor_opt.step();kl=float(((ratio-1)-lr).mean());losses.append(dict(policy=float(pg.detach()),anchor=float(anchor.detach()),kl=kl))
     if kl>.01:early=True;break
    if early:break
   completed+=1;row.update(optimized=True,optimizer_steps=len(losses),last_loss=losses[-1],early_KL_stop=early)
  fit=fit_value(model,value_opt,C,T,eid,phase);value_fits+=1;row.update(value_optimized=True,value_optimizer_steps=fit['optimizer_steps'],value_fit=fit)
  if batch==0:
   torch.save(dict(critic=model.critic.state_dict(),optimizer=value_opt.state_dict()),D/'value_warmup_critic.pt');np.savez_compressed(D/'value_fit_data_batch1.npz',critic_features=C.cpu().numpy(),target=T.cpu().numpy(),phase=phase.cpu().numpy(),env=eid.cpu().numpy())
  verify();history.append(row);(D/'post_training_history.json').write_text(json.dumps(dict(batches=history),indent=2));progress('batch_completed',latest=row)
 verify();cp['model_state_dict']={k:v.detach().clone() for k,v in model.state_dict().items()}
 if 'optimizer_state_dict' in cp:cp['optimizer_state_dict']['state']={}
 cp['value_post_lineage']=dict(source_sha256=sha(source),training_seed=args.seed,completed_actor_updates=completed,value_fits=value_fits,optimizers_saved_separately=True)
 out=D/'post';out.mkdir();path=out/'model_post_ppo.pt';torch.save(cp,path);shutil.copy2(source.parent/'agent.json',out/'agent.json');torch.save(actor_opt.state_dict(),D/'actor_optimizer.pt');torch.save(value_opt.state_dict(),D/'value_optimizer.pt')
 changes={k:float((v-before[k]).abs().max()) for k,v in model.state_dict().items() if not torch.equal(v,before[k])};result=dict(status='completed' if not stop else ('stopped_safety' if 'Force gate' in stop else 'stopped_value_gate'),stop_reason=stop,completed_updates=completed,completed_value_fits=value_fits,full_episodes=sum(x['full_episodes'] for x in history),batches=history,value_gate=gate,checkpoint=str(path),checkpoint_sha256=sha(path),source_sha256=sha(source),changed_tensor_maximums=changes,GRUs_normalizers_std_and_other_roles_exact=True,native_reward_unchanged=True,all_native_gates_unchanged=True,wall_seconds=time.monotonic()-start)
 (D/'post_train_result.json').write_text(json.dumps(result,indent=2));progress(result['status'],checkpoint_sha256=result['checkpoint_sha256'])
