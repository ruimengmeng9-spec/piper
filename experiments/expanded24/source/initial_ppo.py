"""Full-native-episode clipped policy gradient on existing initial XYZ decoders.

Frozen recurrent/body features are captured online, so no recurrent replay or
new actor is needed. Monte Carlo returns use a per-role sample-mean baseline;
the existing critics are frozen. Exploration is training-only.
"""
import datetime as dt,hashlib,json,shutil,time
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from boundary_rl import model_hash

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

class InitialHook:
 def __init__(self,raw):
  self.r=raw;self.cache={};self.layer_cache={};self.active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool)
  self.original={};self.decoders={};self.roots={};self.sigma={'early':.03,'late':.08}
  for role,name in [('early','initial'),('late','late_initial')]:
   model=getattr(raw,name+'_model');policy=getattr(raw,name+'_policy');self.original[role]=policy
   decoder=model.actor[-1] if role=='early' else model.position_head[-1]
   self.decoders[role]=decoder;self.roots[role]=(decoder.weight.detach().clone(),decoder.bias.detach().clone())
   def capture(module,inputs,output,_role=role):self.layer_cache[_role]=(inputs[0].detach().clone(),output.detach().clone())
   decoder.register_forward_hook(capture)
   def sample(obs,_role=role,_model=model,_policy=policy):
    out=_policy(obs);x=obs['policy'];s=raw.supervisor;depth=-x[:,2]/20;force=x[:,24:27]
    eligible=(s.mode==0)&(s.exits==0)&~raw.preparing
    late=eligible&(depth>=.020)&(depth<=.050)&(x[:,3:6].norm(dim=-1)<.04)&(force.abs().amax(-1)<=2)&(force.norm(dim=-1)/.02<8)
    mask=(eligible&~late if _role=='early' else torch.zeros_like(late))&self.active
    feature,decoded=self.layer_cache[_role];mean=decoded[:,:3]
    if _role=='early':assert torch.equal(mean,out[:,:3])
    else:assert torch.allclose(.25*torch.tanh(mean),out[:,:3],atol=2e-6,rtol=1e-6)
    latent=mean.clone()
    if mask.any():latent[mask]=Normal(mean[mask],self.sigma[_role]).sample()
    logp=Normal(mean,self.sigma[_role]).log_prob(latent).sum(-1)
    result=out.clone();xyz=latent if _role=='early' else .25*torch.tanh(latent)
    result[mask,:3]=xyz[mask];assert torch.equal(result[:,3:],out[:,3:])
    self.cache[_role]=(feature.detach().clone(),latent.detach().clone(),logp.detach().clone(),mask.clone())
    return result
   setattr(raw,name+'_policy',sample)

def train(env,raw,boundary,logdir,args):
 assert args.seed==1063 and args.initial_ppo_updates==4 and not args.initial_case_replay and not args.fixed_contact_cases
 assert args.boundary_mode=='eval' and raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry
 from boundary_curriculum import install
 install(raw)
 D=Path(logdir);start=time.monotonic();history=[]
 sources={'initial':Path(args.initial_insertion_checkpoint),'late_initial':Path(args.late_initial_checkpoint)}
 checkpoints={k:torch.load(p,map_location=raw.device,weights_only=False) for k,p in sources.items()}
 hashes={k:sha(p) for k,p in sources.items()}
 assert hashes['initial']=='c81dba13c3eab9e1fe66976d8eac2235ca1460b72e5946353ed30e92c8e4719d'
 originals={name:{k:v.detach().clone() for k,v in getattr(raw,name+'_model').state_dict().items()} for name in sources}
 for name in sources:
  assert all(torch.equal(v,checkpoints[name]['model_state_dict'][k]) for k,v in originals[name].items())
 frozen={k:model_hash(v) for k,v in boundary.models().items() if k not in ['initial_model','late_initial_model']}
 boundary_sha=model_hash(boundary.model);hook=InitialHook(raw)
 for d in [hook.decoders['early']]:
  d.weight.requires_grad_(True);d.bias.requires_grad_(True)
 optimizers={k:torch.optim.Adam(d.parameters(),lr=1e-5) for k,d in hook.decoders.items() if k=='early'}
 audit=dict(server_utc=utc(),source_sha256=hashes,initial_weights_exact=True,new_actor_heads=0,
  trained_scope='only existing early actor final XYZ rows; late effective policy frozen; all rotation,GRU,body,normalizers,critics,other skills and boundary residual frozen',
  optimizer='new decoder-only Adam; original supervised/mixed optimizer moments are not reused or claimed resumed',
  algorithm='on-policy full-native-episode Monte Carlo clipped PPO,per-role sample-mean baseline,no new critic',
  training_std=hook.sigma,learning_rate=1e-5,clip=.1,anchor_coefficient=20.,discount=.99,native_reward_unchanged=False,evaluation_native_reward_unchanged=True,training_only_time_cost=.01,training_only_recovery_entry_cost=5.,training_reset_boundary_fraction=.25,original_randomization_bounds_unchanged=True)
 (D/'initial_resume_audit.json').write_text(json.dumps(audit,indent=2))
 def verify():
  assert model_hash(boundary.model)==boundary_sha
  assert {k:model_hash(v) for k,v in boundary.models().items() if k not in ['initial_model','late_initial_model']}==frozen
  for name,before in originals.items():
   for k,v in getattr(raw,name+'_model').state_dict().items():
    allowed=name=='initial' and k in ['actor.6.weight','actor.6.bias']
    if allowed and name=='initial':assert torch.equal(v[3:],before[k][3:]),('Rotation changed',k)
    elif not allowed:assert torch.equal(v,before[k]),(name,k)
 def progress(phase,**kw):
  x=dict(server_utc=utc(),phase=phase,completed_updates=len(history),target_updates=4,**kw)
  (D/'initial_progress.json').write_text(json.dumps(x,indent=2));print('INITIAL_PPO_PROGRESS',json.dumps(x),flush=True)
 torch.manual_seed(args.seed);env.seed(args.seed);progress('collecting_native_episodes')
 stop_reason=None
 for update in range(4):
  assert shutil.disk_usage(D).free>5045649408+32*1024**2
  obs,_=env.reset();boundary.reset(torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool))
  active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool);hook.active=active.clone()
  records=[];outcomes={};prev_attempts=torch.zeros(raw.num_envs,device=raw.device,dtype=torch.long);native_total=torch.zeros(raw.num_envs,device=raw.device);shaped_total=native_total.clone()
  for tick in range(730):
   alive=active.clone()
   with torch.no_grad():
    action=boundary.act_inference(obs);obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
    for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
     if not bool(alive[i]):continue
     outcomes[i]=dict(env=i,success=bool(term['success'][j]),end_reason=int(term['end_reason'][j]),peak_force_n=float(term['peak_force_n'][j]),seconds=float(term['seconds'][j]),attempts=int(term['recovery_attempts'][j]))
    current_attempts=raw.supervisor.attempts.clone();terminal_ids=term.get('env_ids',torch.empty(0,device=raw.device,dtype=torch.long))
    if len(terminal_ids):current_attempts[terminal_ids]=term['recovery_attempts'].long()
    entries=(current_attempts-prev_attempts).clamp_min(0);prev_attempts.copy_(raw.supervisor.attempts)
    shaped=reward.detach().float()-.01-5.*entries.float();native_total+=reward*alive;shaped_total+=shaped*alive
    records.append((dict(hook.cache),shaped.clone()*alive,done.clone(),alive))
    boundary.reset(done);active&=~done;hook.active=active.clone()
   if not active.any():break
  assert not active.any() and len(outcomes)==raw.num_envs
  rewards=torch.stack([r[1] for r in records]);done=torch.stack([r[2] for r in records]);returns=torch.zeros_like(rewards);future=torch.zeros(raw.num_envs,device=raw.device)
  for t in reversed(range(len(records))):future=rewards[t]+.99*future*(~done[t]);returns[t]=future
  reports={}
  for role,decoder in hook.decoders.items():
   fs=[];lat=[];lp=[];targets=[]
   for t,r in enumerate(records):
    f,a,l,mask=r[0][role];mask=mask&r[3]
    if mask.any():fs.append(f[mask]);lat.append(a[mask]);lp.append(l[mask]);targets.append(returns[t][mask])
   if not fs:reports[role]=dict(controlled_steps=0,optimizer_steps=0);continue
   F,A,LP,T=map(torch.cat,(fs,lat,lp,targets));adv=(T-T.mean())/T.std(unbiased=False).clamp_min(.05)
   oldw,oldb=hook.roots[role];oldmu=torch.nn.functional.linear(F,oldw,oldb)[:,:3];early_stop=False;losses=[]
   for epoch in range(3):
    for q in torch.randperm(len(F),device=raw.device).split(4096):
     mu=decoder(F[q])[:,:3];newlp=Normal(mu,hook.sigma[role]).log_prob(A[q]).sum(-1);logratio=newlp-LP[q];ratio=logratio.exp()
     pg=torch.maximum(-adv[q]*ratio,-adv[q]*ratio.clamp(.9,1.1)).mean()
     mean_action=mu if role=='early' else .25*torch.tanh(mu);anchor_action=oldmu[q] if role=='early' else .25*torch.tanh(oldmu[q])
     anchor=(mean_action-anchor_action).square().mean();loss=pg+20.*anchor
     assert torch.isfinite(loss);opt=optimizers[role];opt.zero_grad();loss.backward()
     if role=='early':decoder.weight.grad[3:]=0;decoder.bias.grad[3:]=0
     nn.utils.clip_grad_norm_(decoder.parameters(),.25);opt.step()
     kl=float(((ratio-1)-logratio).mean());losses.append(dict(policy=float(pg.detach()),anchor=float(anchor.detach()),kl=kl))
     if kl>.01:early_stop=True;break
    if early_stop:break
   with torch.no_grad():
    now=decoder(F)[:,:3];now=now if role=='early' else .25*torch.tanh(now);root=oldmu if role=='early' else .25*torch.tanh(oldmu)
   reports[role]=dict(controlled_steps=len(F),optimizer_steps=len(losses),early_kl_stop=early_stop,maximum_sampled_action_change_from_root=float((now-root).abs().max()),last_loss=losses[-1])
  verify();rows=list(outcomes.values());row=dict(update=update+1,full_episodes=len(rows),successes=sum(x['success'] for x in rows),over100N=sum(x['peak_force_n']>100 for x in rows),
   mean_seconds=sum(x['seconds'] for x in rows)/len(rows),total_native_steps=int(torch.stack([r[3] for r in records]).sum()),role_updates=reports,recovery_attempts=sum(v['attempts'] for v in rows),mean_native_return=float(native_total.mean()),mean_shaped_return=float(shaped_total.mean()))
  history.append(row);(D/'initial_training_history.json').write_text(json.dumps(dict(updates=history,last_episode_outcomes=rows),indent=2));progress('update_completed',latest=row)
  if row['over100N']>raw.num_envs*.10:
   stop_reason='Training boundary-mixture batch exceeded10percent force-abort episodes;preserve qualified release';break
 verify();outputs={}
 for name,cp in checkpoints.items():
  state={k:v.detach().clone() for k,v in getattr(raw,name+'_model').state_dict().items()}
  # The late model's inherited XYZ base is unused by its existing XYZ head.
  # Synchronize those rows only to retain the loader's exact shared-base check.
  if name=='late_initial':
   early=raw.initial_model.state_dict()
   for k in ['actor.6.weight','actor.6.bias']:state[k]=early[k].detach().clone()
  cp['model_state_dict']=state
  if 'optimizer_state_dict' in cp:cp['optimizer_state_dict']['state']={}
  cp['initial_stage_ppo_lineage']=dict(source_sha256=hashes[name],source_pair_sha256=hashes,training_seed=args.seed,completed_updates=len(history),optimizer_saved_separately=True)
  out=D/name;out.mkdir();p=out/'model_initial_ppo.pt';torch.save(cp,p);shutil.copy2(sources[name].parent/'agent.json',out/'agent.json')
  outputs[name]=dict(path=str(p),sha256=sha(p))
 torch.save({k:v.state_dict() for k,v in optimizers.items()},D/'decoder_optimizers.pt')
 result=dict(status='stopped_safety' if stop_reason else 'completed',stop_reason=stop_reason,server_utc=utc(),updates=history,outputs=outputs,source_sha256=hashes,
  full_native_episodes=True,native_reward_unchanged=False,evaluation_native_reward_unchanged=True,training_time_cost=.01,training_recovery_entry_cost=5.,training_reset_boundary_fraction=.25,all_native_gates_unchanged=True,new_actor_heads=0,
  decoder_only_optimizer_initialized=True,wall_seconds=time.monotonic()-start)
 (D/'initial_train_result.json').write_text(json.dumps(result,indent=2));progress('stopped_safety' if stop_reason else 'completed',outputs=outputs)
