"""PPO on existing far-alignment XY decoder; complete native contact episodes."""
import datetime as dt,hashlib,json,shutil,time
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from boundary_rl import model_hash

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

class AlignmentHook:
 def __init__(self,raw):
  self.r=raw;self.model=raw.alignment_model;self.decoder=self.model.actor[-1]
  self.active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool)
  self.previous_exits=torch.zeros(raw.num_envs,device=raw.device,dtype=torch.long)
  self.sigma=.03;original=raw.alignment_policy
  def capture(module,inputs,output):self.feature=inputs[0].detach().clone();self.decoded=output.detach().clone()
  self.decoder.register_forward_hook(capture)
  def sample(obs):
   output=original(obs);x=obs['policy'];s=raw.supervisor;f=x[:,24:27]
   just_exited=s.exits>self.previous_exits;self.previous_exits.copy_(s.exits)
   near=(x[:,2]/20>=.003)&(x[:,2]/20<=.050)&(x[:,:2].norm(dim=-1)/20<.002)&(x[:,3:6].norm(dim=-1)<.04)&(f.abs().amax(-1)<=2)&(f.norm(dim=-1)/.02<8)
   mask=(s.mode==2)&~raw.preparing&~just_exited&~near&self.active
   feature=self.feature;mean=self.decoded[:,:2];assert torch.equal(mean,output[:,:2])
   latent=mean.clone()
   if mask.any():latent[mask]=Normal(mean[mask],self.sigma).sample()
   logp=Normal(mean,self.sigma).log_prob(latent).sum(-1);result=output.clone();result[mask,:2]=latent[mask]
   assert torch.equal(result[:,2:],output[:,2:])
   self.cache=(feature,latent,logp,mask.clone(),just_exited.clone(),near.clone());return result
  raw.alignment_policy=sample

def train(env,raw,boundary,logdir,args):
 assert args.seed==1059 and args.handoff_ppo_updates==4 and args.alignment_axial_cap and args.fixed_contact_cases
 assert raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry and args.boundary_mode=='eval'
 D=Path(logdir);start=time.monotonic();source=Path(args.alignment_checkpoint);cp=torch.load(source,map_location=raw.device,weights_only=False)
 model=raw.alignment_model;before={k:v.detach().clone() for k,v in model.state_dict().items()}
 assert all(torch.equal(v,cp['model_state_dict'][k]) for k,v in before.items())
 frozen={k:model_hash(v) for k,v in boundary.models().items() if k!='alignment_model'};residual_sha=model_hash(boundary.model)
 hook=AlignmentHook(raw);decoder=hook.decoder;decoder.weight.requires_grad_(True);decoder.bias.requires_grad_(True)
 optimizer=torch.optim.Adam(decoder.parameters(),lr=1e-5);rootw=decoder.weight.detach().clone();rootb=decoder.bias.detach().clone();history=[]
 audit=dict(server_utc=utc(),source_sha256=sha(source),source=str(source),initial_weights_exact=True,
  scope='only existing far-alignment actor final XY rows; native near domain and first exit handover excluded',
  new_actor_heads=0,optimizer='new decoder-only Adam; no original mixed optimizer resume claim',learning_rate=1e-5,sigma=.03,anchor=20.,clip=.1,
  algorithm='full-native Monte Carlo clipped PPO, sample-mean baseline, frozen critic',discount=.99,native_reward_unchanged=True,
  frozen_role_hashes=frozen,boundary_residual_sha256=residual_sha)
 (D/'handoff_resume_audit.json').write_text(json.dumps(audit,indent=2))
 def verify():
  assert {k:model_hash(v) for k,v in boundary.models().items() if k!='alignment_model'}==frozen
  assert model_hash(boundary.model)==residual_sha
  for k,v in model.state_dict().items():
   if k in ['actor.6.weight','actor.6.bias']:assert torch.equal(v[2:],before[k][2:]),('Z or rotation changed',k)
   else:assert torch.equal(v,before[k]),k
 def progress(phase,**kw):
  x=dict(server_utc=utc(),phase=phase,completed_updates=len(history),target_updates=4,**kw);(D/'handoff_progress.json').write_text(json.dumps(x,indent=2));print('HANDOFF_PPO_PROGRESS',json.dumps(x),flush=True)
 torch.manual_seed(args.seed);env.seed(args.seed);progress('collecting_complete_native_contact_episodes')
 for update in range(4):
  assert shutil.disk_usage(D).free>5045649408+32*1024**2
  obs,_=env.reset();boundary.reset(torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool));active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool)
  hook.active=active.clone();hook.previous_exits.zero_();records=[];outcomes={}
  for tick in range(730):
   alive=active.clone()
   with torch.no_grad():
    action=boundary.act_inference(obs);obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
    for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
     if bool(alive[i]):outcomes[i]=dict(env=i,success=bool(term['success'][j]),end_reason=int(term['end_reason'][j]),peak_force_n=float(term['peak_force_n'][j]),seconds=float(term['seconds'][j]),exits=float(term['recovery_exits'][j]),realignments=float(term['recovery_realignments'][j]))
    feature,latent,lp,mask,first,near=hook.cache;assert not (mask&first).any() and not (mask&near).any()
    records.append(((feature,latent,lp,mask),reward.detach().float().clone()*alive,done.clone(),alive))
    boundary.reset(done);active&=~done;hook.active=active.clone()
   if not active.any():break
  assert not active.any() and len(outcomes)==raw.num_envs
  rewards=torch.stack([r[1] for r in records]);done=torch.stack([r[2] for r in records]);returns=torch.zeros_like(rewards);future=torch.zeros(raw.num_envs,device=raw.device)
  for t in reversed(range(len(records))):future=rewards[t]+.99*future*(~done[t]);returns[t]=future
  features=[];latents=[];logp=[];targets=[]
  for t,record in enumerate(records):
   f,a,l,mask=record[0];mask=mask&record[3]
   if mask.any():features.append(f[mask]);latents.append(a[mask]);logp.append(l[mask]);targets.append(returns[t][mask])
  assert features,'Prepared-contact curriculum produced no far-alignment actions'
  F,A,LP,T=map(torch.cat,(features,latents,logp,targets));adv=(T-T.mean())/T.std(unbiased=False).clamp_min(.05)
  old=torch.nn.functional.linear(F,rootw,rootb)[:,:2];losses=[];early_stop=False
  for epoch in range(3):
   for q in torch.randperm(len(F),device=raw.device).split(4096):
    mu=decoder(F[q])[:,:2];newlp=Normal(mu,hook.sigma).log_prob(A[q]).sum(-1);logratio=newlp-LP[q];ratio=logratio.exp()
    pg=torch.maximum(-adv[q]*ratio,-adv[q]*ratio.clamp(.9,1.1)).mean();anchor=(mu-old[q]).square().mean();loss=pg+20.*anchor
    assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();decoder.weight.grad[2:]=0;decoder.bias.grad[2:]=0
    nn.utils.clip_grad_norm_(decoder.parameters(),.25);optimizer.step();kl=float(((ratio-1)-logratio).mean());losses.append(dict(policy=float(pg.detach()),anchor=float(anchor.detach()),kl=kl))
    if kl>.01:early_stop=True;break
   if early_stop:break
  verify();rows=list(outcomes.values())
  with torch.no_grad():maximum=float((decoder(F)[:,:2]-old).abs().max())
  row=dict(update=update+1,full_episodes=len(rows),successes=sum(r['success'] for r in rows),over100N=sum(r['peak_force_n']>100 for r in rows),peak40=sum(r['peak_force_n']>40 for r in rows),
   mean_seconds=sum(r['seconds'] for r in rows)/len(rows),controlled_steps=len(F),total_native_steps=int(torch.stack([r[3] for r in records]).sum()),optimizer_steps=len(losses),early_kl_stop=early_stop,last_loss=losses[-1],maximum_sampled_action_change_from_root=maximum)
  history.append(row);(D/'handoff_training_history.json').write_text(json.dumps(dict(updates=history,last_episode_outcomes=rows),indent=2));progress('update_completed',latest=row)
  if row['over100N']>raw.num_envs*.10:raise RuntimeError('Training force gate exceeded;retain qualified release')
 verify();cp['model_state_dict']={k:v.detach().clone() for k,v in model.state_dict().items()}
 if 'optimizer_state_dict' in cp:cp['optimizer_state_dict']['state']={}
 cp['handoff_ppo_lineage']=dict(source_sha256=sha(source),training_seed=args.seed,completed_updates=4,trained_scope='actor final XY only',decoder_optimizer_saved_separately=True)
 output=D/'alignment';output.mkdir();path=output/'model_alignment_ppo.pt';torch.save(cp,path);shutil.copy2(source.parent/'agent.json',output/'agent.json');torch.save(optimizer.state_dict(),D/'decoder_optimizer.pt')
 result=dict(status='completed',server_utc=utc(),updates=history,checkpoint=str(path),checkpoint_sha256=sha(path),source_sha256=sha(source),new_actor_heads=0,
  native_reward_unchanged=True,all_native_gates_unchanged=True,full_native_contact_episodes=True,frozen_role_hashes=frozen,wall_seconds=time.monotonic()-start)
 (D/'handoff_train_result.json').write_text(json.dumps(result,indent=2));progress('completed',checkpoint=str(path))
