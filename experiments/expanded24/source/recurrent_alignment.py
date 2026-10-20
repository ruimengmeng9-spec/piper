"""Truncated recurrent PPO on real shadow histories, full native task returns."""
import copy,hashlib,json,time,shutil
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from boundary_rl import model_hash

DIMS=[0,1,3,4,5]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replay(rnn,actor,x,h,resets):
 outputs=[]
 for t in range(len(x)):
  h=h*(~resets[t])[None,:,None];y,h=rnn(x[t:t+1],h);outputs.append(actor(y[0])[:,DIMS])
 return torch.stack(outputs)

class Hook:
 def __init__(self,raw):
  self.r=raw;self.m=raw.alignment_model;self.active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool);self.previous_exits=torch.zeros(raw.num_envs,device=raw.device,dtype=torch.long)
  assert isinstance(self.m.memory_a.rnn,nn.GRU) and self.m.memory_a.rnn.num_layers==1 and self.m.memory_a.rnn.dropout==0
  self.root_rnn=copy.deepcopy(self.m.memory_a.rnn).eval();self.root_rnn.flatten_parameters();self.root_actor=copy.deepcopy(self.m.actor).eval()
  for p in [*self.root_rnn.parameters(),*self.root_actor.parameters()]:p.requires_grad_(False)
  self.root_h=torch.zeros(1,raw.num_envs,self.m.memory_a.rnn.hidden_size,device=raw.device);self.sigma=torch.tensor([.03,.03,.04,.04,.04],device=raw.device);original=raw.alignment_policy
  def sample(obs):
   h=self.m.memory_a.hidden_state
   if h is None:h=torch.zeros_like(self.root_h)
   assert isinstance(h,torch.Tensor) and h.shape==self.root_h.shape
   h0=h.detach().clone();reset=(h0==0).all(0).all(-1);x=self.m.actor_obs_normalizer(self.m.get_actor_obs(obs)).detach().clone();output=original(obs)
   with torch.no_grad():
    self.root_h*=~reset[None,:,None];y,self.root_h=self.root_rnn(x[None],self.root_h);anchor=self.root_actor(y[0])[:,DIMS]
   s=raw.supervisor;v=obs['policy'];force=v[:,24:27];first=s.exits>self.previous_exits;self.previous_exits.copy_(s.exits)
   near=(v[:,2]/20>=.003)&(v[:,2]/20<=.050)&(v[:,:2].norm(dim=-1)/20<.002)&(v[:,3:6].norm(dim=-1)<.04)&(force.abs().amax(-1)<=2)&(force.norm(dim=-1)/.02<8)
   mask=(s.mode==2)&~raw.preparing&~first&~near&self.active;mean=output[:,DIMS];latent=mean.clone()
   if mask.any():latent[mask]=Normal(mean[mask],self.sigma).sample()
   lp=Normal(mean,self.sigma).log_prob(latent).sum(-1);out=output.clone();out[:,DIMS]=torch.where(mask[:,None],latent,mean)
   assert torch.equal(out[:,2],output[:,2]) and not (mask&first).any() and not (mask&near).any()
   self.cache=(x,h0[0],reset,latent.detach(),lp.detach(),mask.detach().clone(),mean.detach(),anchor.detach())
   return out
  raw.alignment_policy=sample

def pack(data,segments,indices,length=32):
 X,H,R,A,LP,M,OLD,ROOT,ADV=data;out=[]
 for source in [X,R,A,LP,M,OLD,ROOT,ADV]:
  rows=[]
  for index in indices:
   env,start,end=segments[int(index)];v=source[start:end,env];pad=torch.zeros((length-len(v),*v.shape[1:]),device=v.device,dtype=v.dtype)
   if source is R:pad.fill_(True)
   rows.append(torch.cat((v,pad)))
  out.append(torch.stack(rows,1))
 h=torch.stack([H[segments[int(i)][1],segments[int(i)][0]] for i in indices],0)[None]
 return out,h

def train(env,raw,boundary,logdir,args):
 assert args.seed==1065 and args.recurrent_alignment_updates==8 and args.fixed_contact_cases and not args.initial_case_replay
 assert args.boundary_mode=='eval' and raw.cfg.episode_length_s==12 and not raw.cfg.train_stop_failed_retry and not args.alignment_axial_cap
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 D=Path(logdir);start=time.monotonic();m=raw.alignment_model;source=Path(args.alignment_checkpoint);cp=torch.load(source,map_location=raw.device,weights_only=False);before={k:v.detach().clone() for k,v in m.state_dict().items()};assert all(torch.equal(v,cp['model_state_dict'][k]) for k,v in before.items())
 frozen={k:model_hash(v) for k,v in boundary.models().items() if k!='alignment_model'};boundary_hash=model_hash(boundary.model);hook=Hook(raw)
 for name,p in m.named_parameters():p.requires_grad_(name.startswith(('actor.','memory_a.rnn.')))
 parameters=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.Adam(parameters,lr=1e-5);history=[];stop=None;root_hashes=(model_hash(hook.root_rnn),model_hash(hook.root_actor))
 def verify():
  assert model_hash(boundary.model)==boundary_hash and {k:model_hash(v) for k,v in boundary.models().items() if k!='alignment_model'}==frozen
  assert (model_hash(hook.root_rnn),model_hash(hook.root_actor))==root_hashes
  for k,v in m.state_dict().items():
   if not k.startswith(('actor.','memory_a.rnn.')):assert torch.equal(v,before[k]),k
   elif k in ['actor.6.weight','actor.6.bias']:assert torch.equal(v[2],before[k][2]),k
 def progress(phase,**kw):
  v=dict(phase=phase,completed_updates=sum(x['optimized'] for x in history),collected_batches=len(history),target_updates=8,**kw);(D/'recurrent_progress.json').write_text(json.dumps(v,indent=2));print('RECURRENT_PPO',json.dumps(v),flush=True)
 audit=dict(source=str(source),source_sha256=sha(source),source_exact=True,algorithm='full-native MC clipped PPO with length32 truncated recurrent replay;actor/body/GRU train;critic and normalizers frozen',new_actor_heads=0,training_TF32_disabled_for_replay_consistency=True,learning_rate=1e-5,discount=.99,epochs=2,sequences_per_minibatch=16,exploration_std=hook.sigma.tolist(),anchor=20,actor_Z_decoder_row_frozen=True,applied_alignment_Z_zero_by_original_supervisor=True,native_reward_unchanged=True,recorded_diagnostic_case_not_used=True,frozen_roles=frozen)
 (D/'recurrent_resume_audit.json').write_text(json.dumps(audit,indent=2));torch.manual_seed(args.seed);env.seed(args.seed)
 for update in range(8):
  assert shutil.disk_usage(D).free>5045649408+32*1024**2
  m.memory_a.rnn.eval();obs,_=env.reset();boundary.reset(torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool));active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool);hook.active=active.clone();hook.previous_exits.zero_();hook.root_h.zero_();records=[];outcomes={};progress('collecting',batch=update+1)
  for tick in range(730):
   alive=active.clone()
   with torch.no_grad():
    action=boundary.act_inference(obs);obs,reward,done,info=env.step(action);done=done.bool();term=info.get('terminal',{})
    for j,i in enumerate(term.get('env_ids',torch.empty(0,dtype=torch.long)).cpu().tolist()):
     if bool(alive[i]):outcomes[i]=dict(env=i,success=bool(term['success'][j]),peak_force_n=float(term['peak_force_n'][j]),reason=int(term['end_reason'][j]),seconds=float(term['seconds'][j]))
    assert not (raw.r2_last_action[(raw.training_control_mode==2)&~raw.training_control_preparing,2]!=0).any()
    records.append((hook.cache,reward.detach().float().clone()*alive,done.clone(),alive));boundary.reset(done);active&=~done;hook.active=active.clone()
   if not active.any():break
  assert not active.any() and len(outcomes)==raw.num_envs;rows=list(outcomes.values());row=dict(batch=update+1,full_episodes=len(rows),successes=sum(v['success'] for v in rows),over100N=sum(v['peak_force_n']>100 for v in rows),native_steps=int(torch.stack([v[3] for v in records]).sum()),optimized=False,optimizer_steps=0)
  with (D/'recurrent_episode_outcomes.jsonl').open('a') as f:
   for v in rows:f.write(json.dumps(dict(batch=update+1,**v))+'\n')
  if row['over100N']>raw.num_envs*.10:
   stop='Native contact batch exceeded10percent force-abort episodes';history.append(row);(D/'recurrent_history.json').write_text(json.dumps(dict(batches=history,last_outcomes=rows),indent=2));progress('stopped_safety',latest=row);break
  rewards=torch.stack([v[1] for v in records]);dones=torch.stack([v[2] for v in records]);returns=torch.zeros_like(rewards);future=torch.zeros(raw.num_envs,device=raw.device)
  for t in reversed(range(len(records))):future=rewards[t]+.99*future*(~dones[t]);returns[t]=future
  fields=[torch.stack([v[0][k] for v in records]) for k in range(8)];X,H,R,A,LP,M,OLD,ROOT=fields;M=M&torch.stack([v[3] for v in records]);assert M.any();values=returns[M];ADV=(returns-values.mean())/values.std(unbiased=False).clamp_min(.05);data=[X,H,R,A,LP,M,OLD,ROOT,ADV]
  segments=[(j,t,min(t+32,len(records))) for j in range(raw.num_envs) for t in range(0,len(records),32) if bool(M[t:t+32,j].any())];assert segments
  m.memory_a.rnn.train();errors=[]
  with torch.no_grad():
   indices=torch.linspace(0,len(segments)-1,min(64,len(segments))).long().tolist()
   for pos in range(0,len(indices),16):
    b,h=pack(data,segments,indices[pos:pos+16]);xs,rs,aa,ll,mm,old,root,adv=b;pred=replay(m.memory_a.rnn,m.actor,xs,h,rs);errors.append(float((pred-old).abs()[mm].max()))
  maxerr=max(errors);assert maxerr<2e-5,('Recurrent sequence does not reproduce native mean',maxerr)
  losses=[];early=False
  for epoch in range(2):
   for ids in torch.randperm(len(segments)).split(16):
    b,h=pack(data,segments,ids.tolist());xs,rs,aa,ll,mm,old,root,adv=b;mu=replay(m.memory_a.rnn,m.actor,xs,h,rs);newlp=Normal(mu,hook.sigma).log_prob(aa).sum(-1);lr=newlp-ll;ratio=lr.exp();kl=float((((ratio-1)-lr)[mm]).mean().detach())
    if kl>.01:early=True;break
    pg=torch.maximum(-adv*ratio,-adv*ratio.clamp(.9,1.1))[mm].mean();anchor=(mu-root).square().mean(-1)[mm].mean();loss=pg+20.*anchor;assert torch.isfinite(loss);opt.zero_grad();loss.backward();m.actor[-1].weight.grad[2]=0;m.actor[-1].bias.grad[2]=0;nn.utils.clip_grad_norm_(parameters,.25);opt.step();losses.append(dict(policy=float(pg.detach()),anchor=float(anchor.detach()),kl=kl))
   if early:break
  assert losses;verify();m.memory_a.rnn.eval();row.update(optimized=True,optimizer_steps=len(losses),controlled_steps=int(M.sum()),segments=len(segments),replay_max_error=maxerr,early_kl_stop=early,last_loss=losses[-1]);history.append(row);(D/'recurrent_history.json').write_text(json.dumps(dict(batches=history,last_outcomes=rows),indent=2));progress('update_completed',latest=row)
 verify();cp['model_state_dict']={k:v.detach().clone() for k,v in m.state_dict().items()}
 if 'optimizer_state_dict' in cp:cp['optimizer_state_dict']['state']={}
 cp['recurrent_PPO_lineage']=dict(source_sha256=sha(source),seed=args.seed,completed_updates=sum(v['optimized'] for v in history),truncated_sequence_length=32)
 out=D/'alignment';out.mkdir();path=out/'model_recurrent_alignment.pt';torch.save(cp,path);shutil.copy2(source.parent/'agent.json',out/'agent.json');torch.save(opt.state_dict(),D/'actor_optimizer.pt')
 near_source=Path(args.near_alignment_checkpoint);nearcp=torch.load(near_source,map_location=raw.device,weights_only=False)
 for k,v in m.state_dict().items():
  if k.startswith('memory_a.rnn.'):nearcp['model_state_dict'][k]=v.detach().clone()
 nearcp['compatibility_lineage']=dict(source_sha256=sha(near_source),unused_shadow_GRU_synchronized=True,effective_current_observation_actor_exact=True)
 nearout=D/'near_alignment';nearout.mkdir();nearpath=nearout/'model_near_compatible.pt';torch.save(nearcp,nearpath);shutil.copy2(near_source.parent/'agent.json',nearout/'agent.json')
 result=dict(status='stopped_safety' if stop else 'completed',stop_reason=stop,batches=history,completed_updates=sum(v['optimized'] for v in history),full_episodes=sum(v['full_episodes'] for v in history),source_sha256=sha(source),checkpoint_sha256=sha(path),near_source_sha256=sha(near_source),near_checkpoint_sha256=sha(nearpath),architecture_unchanged=True,new_actor_heads=0,native_reward_unchanged=True,native_physical_gates_unchanged=True,wall_seconds=time.monotonic()-start)
 (D/'recurrent_train_result.json').write_text(json.dumps(result,indent=2));progress('stopped_safety' if stop else 'completed',result=result)
