"""Actual retry-insertion actor PPO with frozen native GRU and normalization."""
import copy,datetime as dt,hashlib,json,shutil,time
from pathlib import Path
import torch
import numpy as np
from torch import nn
from torch.distributions import Normal
from boundary_rl import model_hash

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

class Hook:
 def __init__(self,raw,boundary):
  self.r=raw;self.model=boundary.base;self.active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool)
  self.sigma=torch.tensor([.02,.02,.02,.015,.015,.015],device=raw.device)
  def capture(module,inputs,output):self.feature=inputs[0].detach().clone();self.mean=output.detach().clone()
  self.model.actor.register_forward_hook(capture)
  original=boundary.compose
  def compose(r,base,just_exited):
   mask=r.training_student_control&self.active&~just_exited
   assert not (mask&r.preparing).any() and not (mask&(r.supervisor.mode!=0)).any()
   if mask.any():assert torch.equal(base[mask],self.mean[mask].clamp(-1,1)),'Selected retry action differs from native clipped post actor'
   latent=self.mean.clone()
   if mask.any():latent[mask]=Normal(self.mean[mask],self.sigma).sample()
   lp=Normal(self.mean,self.sigma).log_prob(latent).sum(-1)
   mixed=base.clone();mixed[mask]=latent[mask].clamp(-1,1)
   out=original(r,mixed,just_exited)
   assert torch.equal(out[~mask],(base+torch.where(boundary.mask,boundary.delta,0.))[~mask])
   self.cache=(self.feature,latent.detach(),lp.detach(),mask.clone(),self.mean.clone())
   return out
  boundary.compose=compose

def train(env,raw,boundary,logdir,args):
 assert args.seed==1069 and args.post_ppo_updates==8 and args.fixed_contact_cases and not args.initial_case_replay and not args.alignment_axial_cap
 assert args.boundary_mode=='eval' and raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 D=Path(logdir);start=time.monotonic();source=Path(args.checkpoint);cp=torch.load(source,map_location=raw.device,weights_only=False)
 model=boundary.base;before={k:v.detach().clone() for k,v in model.state_dict().items()}
 assert all(torch.equal(v,cp['model_state_dict'][k]) for k,v in before.items())
 frozen={k:model_hash(v) for k,v in boundary.models().items() if k!='base'};residual_hash=model_hash(boundary.model)
 root=copy.deepcopy(model.actor).eval()
 for p in root.parameters():p.requires_grad_(False)
 for name,p in model.named_parameters():p.requires_grad_(name.startswith('actor.'))
 hook=Hook(raw,boundary);params=list(model.actor.parameters());opt=torch.optim.Adam(params,lr=1e-5);history=[];completed=0;stop=None
 def verify():
  assert {k:model_hash(v) for k,v in boundary.models().items() if k!='base'}==frozen
  assert model_hash(boundary.model)==residual_hash
  for k,v in model.state_dict().items():
   if not k.startswith('actor.'):assert torch.equal(v,before[k]),k
 def progress(phase,**kw):
  v=dict(server_utc=utc(),phase=phase,completed_updates=completed,collected_batches=len(history),target_updates=8,**kw);(D/'post_progress.json').write_text(json.dumps(v,indent=2));print('POST_PPO',json.dumps(v),flush=True)
 audit=dict(source=str(source),source_sha256=sha(source),initial_weights_exact=True,new_actor_heads=0,scope='existing effective post actor MLP body and all6 output rows; actual retry actions only',frozen='postGRU,N,std,critic,allotherroles,boundaryresidual',algorithm='on-policy complete native MonteCarlo PPO with frozen online GRU features and sample-meanbaseline',learning_rate=1e-5,sigma=hook.sigma.tolist(),anchor=20,clip=.1,gamma=.99,epochs=2,native_reward_unchanged=True,safety_gate_before_gradient=True,training_precision='TF32off',training_case_replay=False)
 (D/'post_resume_audit.json').write_text(json.dumps(audit,indent=2));torch.manual_seed(args.seed);env.seed(args.seed)
 for batch in range(8):
  assert shutil.disk_usage(D).free>5045649408+32*1024**2
  obs,_=env.reset();boundary.reset(torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool));active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool);hook.active=active.clone();records=[];outcomes={}
  progress('collecting_complete_native_contact_episodes',batch=batch+1)
  for tick in range(730):
   alive=active.clone()
   with torch.no_grad():
    action=boundary.act_inference(obs);obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
    for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
     if bool(alive[i]):outcomes[i]=dict(batch=batch+1,env=i,success=bool(term['success'][j]),end_reason=int(term['end_reason'][j]),peak_force_n=float(term['peak_force_n'][j]),seconds=float(term['seconds'][j]),attempts=int(term['recovery_attempts'][j]),realignments=int(term['recovery_realignments'][j]))
    records.append((hook.cache,reward.detach().float().clone()*alive,done.clone(),alive))
    boundary.reset(done);active&=~done;hook.active=active.clone()
   if not active.any():break
  assert not active.any() and len(outcomes)==raw.num_envs
  rows=list(outcomes.values());over=sum(x['peak_force_n']>100 for x in rows)
  with (D/'post_episode_outcomes.jsonl').open('a') as f:
   for x in rows:f.write(json.dumps(x)+'\n')
  row=dict(batch=batch+1,full_episodes=len(rows),successes=sum(x['success'] for x in rows),over100N=over,total_native_steps=int(torch.stack([r[3] for r in records]).sum()),mean_seconds=sum(x['seconds'] for x in rows)/len(rows),optimized=False,optimizer_steps=0)
  if over>raw.num_envs*.10:
   stop='Contact batch exceeded10percent force-abort gate before gradients';history.append(row);progress('stopped_safety',latest=row);break
  rewards=torch.stack([r[1] for r in records]);done=torch.stack([r[2] for r in records]);returns=torch.zeros_like(rewards);future=torch.zeros(raw.num_envs,device=raw.device)
  for t in reversed(range(len(records))):future=rewards[t]+.99*future*(~done[t]);returns[t]=future
  fs=[];aa=[];ll=[];tt=[];means=[];times=[];env_ids=[]
  for t,r in enumerate(records):
   f,a,l,mask,mu=r[0];mask=mask&r[3]
   if mask.any():fs.append(f[mask]);aa.append(a[mask]);ll.append(l[mask]);tt.append(returns[t][mask]);means.append(mu[mask]);times.append(torch.full_like(l[mask],t));env_ids.append(mask.nonzero().flatten())
  assert fs,'No effective retry control samples'
  F,A,LP,T,OLD=map(torch.cat,(fs,aa,ll,tt,means));adv=(T-T.mean())/T.std(unbiased=False).clamp_min(.05)
  with torch.no_grad():
   err=float((model.actor(F)-OLD).abs().max());anchor_mu=root(F)
  assert err<2e-5,err
  timing=torch.cat(times);eid=torch.cat(env_ids);terminal_tick=torch.stack([r[3] for r in records]).long().sum(0)-1
  phase=torch.empty_like(timing)
  for e in eid.unique():
   q=(eid==e).nonzero().flatten();phase[q]=torch.arange(len(q),device=raw.device)/max(len(q)-1,1)
  credit={}
  for name,sel in [('early',phase<=1/3),('middle',(phase>1/3)&(phase<2/3)),('late',phase>=2/3)]:
   ids=sel.nonzero().flatten();assert len(ids)>0
   probe=ids[::max(1,len(ids)//1024)][:1024]
   mu=model.actor(F[probe]);lp=Normal(mu,hook.sigma).log_prob(A[probe]).sum(-1);loss=-(adv[probe]*lp).mean()
   grads=torch.autograd.grad(loss,params);gn=float(torch.sqrt(sum(g.square().sum() for g in grads)));assert np.isfinite(gn) and gn>0
   discount=.99**(terminal_tick[eid[ids]]-timing[ids]);credit[name]=dict(samples=len(ids),probe_samples=len(probe),policy_gradient_norm=gn,return_mean=float(T[ids].mean()),return_std=float(T[ids].std(unbiased=False)),mean_terminal_discount=float(discount.mean()),mean_steps_to_terminal=float((terminal_tick[eid[ids]]-timing[ids]).mean()))
  audit_ids=torch.arange(0,len(F),max(1,len(F)//256),device=raw.device)[:256]
  np.savez_compressed(D/f'post_credit_probe_batch{batch+1}.npz',features=F[audit_ids].cpu().numpy(),latent=A[audit_ids].cpu().numpy(),old_mean=OLD[audit_ids].cpu().numpy(),old_logp=LP[audit_ids].cpu().numpy(),returns=T[audit_ids].cpu().numpy(),advantage=adv[audit_ids].cpu().numpy(),phase=phase[audit_ids].cpu().numpy(),sigma=hook.sigma.cpu().numpy())
  losses=[];early=False
  for epoch in range(2):
   for q in torch.randperm(len(F),device=raw.device).split(2048):
    mu=model.actor(F[q]);newlp=Normal(mu,hook.sigma).log_prob(A[q]).sum(-1);lr=newlp-LP[q];ratio=lr.exp();pg=torch.maximum(-adv[q]*ratio,-adv[q]*ratio.clamp(.9,1.1)).mean();anchor=(mu-anchor_mu[q]).square().mean();loss=pg+20*anchor
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,.25);opt.step();kl=float(((ratio-1)-lr).mean());losses.append(dict(policy=float(pg.detach()),anchor=float(anchor.detach()),kl=kl))
    if kl>.01:early=True;break
   if early:break
  completed+=1;verify()
  with torch.no_grad():change=float((model.actor(F)-anchor_mu).abs().max())
  row.update(credit_diagnostics=credit,optimized=True,optimizer_steps=len(losses),controlled_steps=len(F),feature_replay_max_error=err,last_loss=losses[-1],early_KL_stop=early,maximum_action_change_from_root=change);history.append(row)
  (D/'post_training_history.json').write_text(json.dumps(dict(batches=history),indent=2));progress('update_completed',latest=row)
 verify();cp['model_state_dict']={k:v.detach().clone() for k,v in model.state_dict().items()}
 if 'optimizer_state_dict' in cp:cp['optimizer_state_dict']['state']={}
 cp['post_ppo_lineage']=dict(source_sha256=sha(source),training_seed=args.seed,completed_updates=completed,optimizer_saved_separately=True)
 out=D/'post';out.mkdir();path=out/'model_post_ppo.pt';torch.save(cp,path);shutil.copy2(source.parent/'agent.json',out/'agent.json');torch.save(opt.state_dict(),D/'actor_optimizer.pt')
 changes={k:float((v-before[k]).abs().max()) for k,v in model.state_dict().items() if not torch.equal(v,before[k])}
 result=dict(status='stopped_safety' if stop else 'completed',stop_reason=stop,completed_updates=completed,full_episodes=sum(x['full_episodes'] for x in history),batches=history,checkpoint=str(path),checkpoint_sha256=sha(path),source_sha256=sha(source),changed_tensor_maximums=changes,all_other_tensors_exact=True,boundary_feature_function_preserved=True,native_reward_unchanged=True,all_native_gates_unchanged=True,wall_seconds=time.monotonic()-start)
 (D/'post_train_result.json').write_text(json.dumps(result,indent=2));progress(result['status'],checkpoint_sha256=result['checkpoint_sha256'])
