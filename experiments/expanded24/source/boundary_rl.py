"""Bounded residual PPO with complete native episodes and frozen recurrent features."""
import datetime as dt,hashlib,json,shutil,time
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def model_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

class Residual(nn.Module):
 def __init__(self):
  super().__init__();self.actor=nn.Sequential(nn.Linear(66,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,6))
  self.critic=nn.Sequential(nn.Linear(73,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,2))
  nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
  self.register_buffer('scale',torch.tensor([.35,.35,.35,.12,.12,.12]));self.register_buffer('std',torch.full((6,),.30))
 def mean(self,x):return self.actor(x)

class Controller:
 def __init__(self,raw,runner,args):
  self.raw=raw;self.base=runner.alg.policy;self.base.eval();self.base_policy=runner.get_inference_policy(device=raw.device);self.args=args
  for p in self.base.parameters():p.requires_grad_(False)
  self.frozen={k:model_hash(v) for k,v in self.models().items()}
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   torch.manual_seed(1042);self.model=Residual().to(raw.device)
  if args.boundary_checkpoint:
   cp=torch.load(args.boundary_checkpoint,map_location=raw.device,weights_only=False);self.model.load_state_dict(cp['state_dict'])
  n=raw.num_envs;self.post_left=torch.zeros(n,device=raw.device,dtype=torch.long);self.prev_align=self.post_left.clone();self.prev_attempt=self.post_left.clone()
  self.enabled=torch.ones(n,device=raw.device,dtype=torch.bool);self.stochastic=args.boundary_mode=='train';self.total_control=0;self.max_residual=torch.zeros(6,device=raw.device)
  self.stage_steps=torch.zeros(3,device=raw.device);self.n=0;raw.boundary_controller=self
 def models(self):
  return {k:m for k in ['base','recovery_model','alignment_model','initial_model','late_initial_model','deep_recovery_model','near_alignment_model'] if (m:=self.base if k=='base' else getattr(self.raw,k,None)) is not None}
 def reset(self,dones=None):
  self.base.reset(dones)
  if dones is None:self.post_left.zero_();self.prev_align.zero_();self.prev_attempt.zero_()
  else:
   q=dones.bool().flatten();self.post_left[q]=0;self.prev_align[q]=0;self.prev_attempt[q]=0
 def act_inference(self,obs):
  base=self.base_policy(obs);self.last_base=base.detach().clone()
  x=self.base.actor_obs_normalizer(self.base.get_actor_obs(obs));h=self.base.memory_a.hidden_state
  assert isinstance(h,torch.Tensor) and h.shape==(1,self.raw.num_envs,33)
  self.features=torch.cat((x.clamp(-8,8),h[0]),-1).detach();self.public=obs['policy'].detach().clone()
  self.mu=self.model.mean(self.features);dist=Normal(self.mu,self.model.std);self.latent=dist.sample() if self.stochastic else self.mu
  self.logp=dist.log_prob(self.latent);self.delta=self.model.scale*torch.tanh(self.latent)
  return base
 def compose(self,raw,base,just_exited):
  s=raw.supervisor;resume=s.realignments>self.prev_align;self.prev_align.copy_(s.realignments)
  self.post_left=torch.where(resume,torch.full_like(self.post_left,48),(self.post_left-1).clamp_min(0))
  selected=((s.mode==1)|(s.mode==2)|((s.mode==0)&(self.post_left>0)))&~raw.preparing&~just_exited&self.enabled
  self.mask=selected[:,None].expand(-1,6).clone();self.mask[:,2]&=s.mode!=2
  delta=torch.where(self.mask,self.delta,0.)
  assert (delta.abs()<=self.model.scale+1e-6).all()
  self.last_retry=(s.attempts-self.prev_attempt).clamp_min(0).float();self.prev_attempt.copy_(s.attempts)
  self.context=torch.cat((self.features,torch.nn.functional.one_hot(s.mode,3).float(),
       (raw.episode_controller_steps.float()/720)[:,None],(s.attempts.float()/3)[:,None],(s.exits.float()/3)[:,None],(self.post_left.float()/48)[:,None]),-1)
  assert self.context.shape[-1]==73
  self.lead_cost=self.public[:,27:30].square().sum(-1)*2500*selected.float()
  self.total_control+=int(selected.sum());self.max_residual=torch.maximum(self.max_residual,delta.abs().amax(0));self.n+=1
  for i in range(3):self.stage_steps[i]+=((s.mode==i)&selected).sum()
  result=base+delta
  assert (result[just_exited]==base[just_exited]).all() and (result[s.mode==2,2]==0).all()
  return result
 def verify_frozen(self):
  assert {k:model_hash(v) for k,v in self.models().items()}==self.frozen,'Frozen base/skill/normalizer changed'
 def finalize(self,logdir):
  self.verify_frozen();report=dict(server_utc=utc(),external_actor_inputs=33,actions=6,current_and_frozen_GRU_features=66,critic_context_training_only=True,
   total_control_steps=self.total_control,stage_control_steps=self.stage_steps.tolist(),maximum_raw_residual_by_axis=self.max_residual.tolist(),scale=self.model.scale.tolist(),all_native_gates_unchanged=True,
   alignment_z_zero=True,first_exit_zero_preserved=True,history_preserved=True,continuous_base_GRU=True,frozen_model_hashes=self.frozen,stochastic=self.stochastic)
  (Path(logdir)/'boundary_interface_audit.json').write_text(json.dumps(report,indent=2));return report

def train(env,raw,runner,logdir,args):
 assert args.seed in range(1042,1100) and args.boundary_updates<=18 and args.hole_width_mm==14 and args.mode=='eval'
 assert not raw.cfg.train_stop_failed_retry and raw.cfg.episode_length_s==12.
 D=Path(logdir);controller=Controller(raw,runner,args);m=controller.model;opt=torch.optim.Adam(m.parameters(),lr=3e-5);reports=[]
 if args.boundary_checkpoint:
  cp=torch.load(args.boundary_checkpoint,map_location=raw.device,weights_only=False)
  if 'optimizer' in cp:opt.load_state_dict(cp['optimizer'])
 # All learned weights must come from the explicitly supplied checkpoint.
 assert all(torch.equal(m.state_dict()[k],v) for k,v in cp['state_dict'].items())
 root=Path('/workspace/piper-insertion/logs/reliability_v2_boundary_pilot1_random/boundary_last.pt')
 root_sha=hashlib.sha256(root.read_bytes()).hexdigest()
 assert root_sha=='84ca1a797d5992bcee0a922020a1f8d7464a9c9849033c4819686546d47196a8'
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  torch.manual_seed(1050);anchor_model=Residual().to(raw.device)
 root_cp=torch.load(root,map_location=raw.device,weights_only=False);anchor_model.load_state_dict(root_cp['state_dict']);anchor_model.eval()
 for p in anchor_model.parameters():p.requires_grad_(False)
 anchor_hash=model_hash(anchor_model)
 loaded_steps=[int(v['step']) for v in opt.state.values() if 'step' in v]
 assert loaded_steps and min(loaded_steps)>0,'Optimizer was not resumed'
 audit=dict(source_checkpoint=args.boundary_checkpoint,source_sha256=hashlib.sha256(Path(args.boundary_checkpoint).read_bytes()).hexdigest(),
  root_accepted_sha256=root_sha,all_initial_weights_exact=True,architecture_unchanged=True,optimizer_resumed=True,
  minimum_loaded_optimizer_step=min(loaded_steps),maximum_loaded_optimizer_step=max(loaded_steps),learning_rate=3e-5,
  exploration_std=.10,regularization='original accepted actor on same public features,not zero',new_actor_heads=0,time_cost_per_native_step=.03,previous_time_cost=.005)
 (D/'direct_resume_audit.json').write_text(json.dumps(audit,indent=2))
 for group in opt.param_groups:group['lr']=3e-5
 with torch.no_grad():m.std.fill_(.10)
 torch.manual_seed(args.seed);env.seed(args.seed);start=time.monotonic()
 def progress(phase,**extra):
  v=dict(server_utc=utc(),phase=phase,seed=args.seed,profile='fixed' if args.fixed_contact_cases else 'random',completed_updates=len(reports),target_updates=args.boundary_updates,**extra)
  (D/'boundary_progress.json').write_text(json.dumps(v,indent=2));print('BOUNDARY_RL_PROGRESS',json.dumps(v),flush=True)
 progress('starting_complete_episode_ppo',actor_features='current public33 + frozen base GRU33',new_training='full-task Monte Carlo PPO, not BC or offline surrogate optimization')
 for update in range(args.boundary_updates):
  assert shutil.disk_usage(D).free>5045649408+16*1024**2
  obs,_=env.reset();controller.reset(torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device));active=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
  controller.enabled=active.clone();records=[];ends={};ticks=0
  for step in range(730):
   alive=active.clone()
   with torch.no_grad():
    action=controller.act_inference(obs);obs,_,done,info=env.step(action);done=done.bool();terminal=info.get('terminal',{})
    force=raw._geometry()[2].norm(dim=-1);rewards=-.03-.002*controller.lead_cost-.5*controller.last_retry
    finish=done&alive
    if finish.any():
     ids=terminal['env_ids'].long()
     force[ids]=terminal['contact_force_n']
     for j,i in enumerate(ids.cpu().tolist()):
      if not bool(alive[i]):continue
      success=bool(terminal['success'][j]);reason=int(terminal['end_reason'][j]);peak=float(terminal['peak_force_n'][j])
      rewards[i]+=100. if success else (-200. if reason==1 else -50. if reason==2 else -20.)
      ends[i]=dict(env=i,success=success,reason=reason,peak_force_n=peak,seconds=float(terminal['seconds'][j]),attempts=float(terminal['recovery_attempts'][j]),realignments=float(terminal['recovery_realignments'][j]))
    rewards-=.02*((force-40).clamp_min(0)/10).square()
    records.append((controller.features.clone(),controller.context.clone(),controller.latent.clone(),controller.logp.clone(),controller.mask.clone(),rewards*alive,done.clone(),alive))
    controller.reset(done);active&=~done;controller.enabled=active.clone();ticks+=1
   if step%180==179:progress('collecting_full_native_episodes',update=update+1,tick=ticks,complete_episodes=len(ends))
   if not active.any():break
  assert not active.any() and len(ends)==raw.num_envs,'Native full-episode quota incomplete'
  X,C,A,LP,M,R,Done,Alive=[torch.stack([r[k] for r in records]) for k in range(8)]
  returns=torch.zeros_like(R);future=torch.zeros(raw.num_envs,device=raw.device)
  for t in reversed(range(len(records))):future=R[t]+.999*future*(~Done[t]);returns[t]=future
  unsafe=torch.tensor([not ends[i]['success'] or ends[i]['peak_force_n']>40 for i in range(raw.num_envs)],device=raw.device).float()
  keep=Alive.flatten();xf=X.flatten(0,1)[keep];cf=C.flatten(0,1)[keep];af=A.flatten(0,1)[keep];lp=LP.flatten(0,1)[keep];mask=M.flatten(0,1)[keep];target=returns.flatten()[keep]/100
  labels=unsafe[None,:].expand_as(R).flatten()[keep];controlled=mask.any(-1)
  with torch.no_grad():
   oldvalue=m.critic(cf)[:,0];adv=target-oldvalue
   if controlled.any():adv=(adv-adv[controlled].mean())/adv[controlled].std(unbiased=False).clamp_min(.05)
  training=[];early_kl=False
  for epoch in range(3):
   order=torch.randperm(len(xf),device=raw.device)
   for q in order.split(4096):
    dist=Normal(m.mean(xf[q]),m.std);newlp=dist.log_prob(af[q]);logratio=((newlp-lp[q])*mask[q]).sum(-1);ratio=logratio.exp();sel=controlled[q].float()
    pg=(torch.maximum(-adv[q]*ratio,-adv[q]*ratio.clamp(.9,1.1))*sel).sum()/sel.sum().clamp_min(1)
    values=m.critic(cf[q]);vf=torch.nn.functional.smooth_l1_loss(values[:,0],target[q]);risk=torch.nn.functional.binary_cross_entropy_with_logits(values[:,1],labels[q])
    anchor=((m.mean(xf[q])-anchor_model.mean(xf[q]).detach()).square().mean(-1)*sel).sum()/sel.sum().clamp_min(1)
    loss=pg+.5*vf+.05*risk+.1*anchor
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step()
    kl=float((((ratio-1)-logratio)*sel).sum()/sel.sum().clamp_min(1));training.append(dict(policy=float(pg.detach()),value=float(vf.detach()),risk=float(risk.detach()),kl=kl))
    if kl>.01:early_kl=True;break
   if early_kl:break
  controller.verify_frozen();assert model_hash(anchor_model)==anchor_hash
  e=list(ends.values());report=dict(update=update+1,full_episodes=len(e),successes=sum(x['success'] for x in e),over100N=sum(x['peak_force_n']>100 for x in e),bad_or_peak40=sum(not x['success'] or x['peak_force_n']>40 for x in e),
   mean_seconds=sum(x['seconds'] for x in e)/len(e),mean_attempts=sum(x['attempts'] for x in e)/len(e),controlled_steps=int(controlled.sum()),total_native_steps=int(Alive.sum()),mean_training_objective=float(R.sum()/raw.num_envs),ppo_minibatches=len(training),last_losses=training[-1],early_KL_stop=early_kl)
  reports.append(report);(D/'boundary_training_history.json').write_text(json.dumps(dict(training_seed=args.seed,profile='fixed' if args.fixed_contact_cases else 'random',updates=reports,episode_outcomes_last=e),indent=2))
  torch.save(dict(recipe='accepted_time_aware_resume_v1',time_cost_per_native_step=.03,root_accepted_sha256=root_sha,lineage_parent_sha256=audit['source_sha256'],cumulative_updates_since_accepted=cp.get('cumulative_updates_since_accepted',0)+update+1,state_dict=m.state_dict(),optimizer=opt.state_dict(),training_seed=args.seed,completed_updates=update+1,base_frozen_hashes=controller.frozen),D/'boundary_last.pt')
  progress('complete_episode_update_finished',latest=report)
  if report['over100N']>raw.num_envs*.10:
   progress('stopped_excess_training_force',latest=report);(D/'boundary_train_result.json').write_text(json.dumps(dict(status='stopped_training_force_gate',updates=reports),indent=2));return
 controller.finalize(D);result=dict(status='completed',server_utc=utc(),training_seed=args.seed,updates=reports,checkpoint=str(D/'boundary_last.pt'),checkpoint_sha256=hashlib.sha256((D/'boundary_last.pt').read_bytes()).hexdigest(),runtime_teacher=False,BC_samples=0,root_accepted_sha256=root_sha,direct_weight_and_optimizer_resume=True,unsafe_episodes_negative_only=True,time_cost_per_native_step=.03,
   complete_native_episodes_only=True,all_native_gates_unchanged=True,wall_seconds=time.monotonic()-start)
 (D/'boundary_train_result.json').write_text(json.dumps(result,indent=2));progress('training_completed',checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'])
