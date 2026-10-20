"""PPO likelihood terms apply only when the insertion policy controls physics."""
import hashlib,inspect,json,textwrap,types
from pathlib import Path
import torch
from rsl_rl.utils import split_and_pad_trajectories,unpad_trajectories

def model_hash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def install(runner,raw,logdir,env,args):
 alg=runner.alg;assert alg.policy.is_recurrent and not alg.rnd and not alg.symmetry and not alg.is_multi_gpu
 assert 'training_control_mask' in alg.storage.observations
 # Verify storage's exact padding/unpadding order before applying any gradients.
 x=torch.arange(24.,device=alg.device).reshape(4,6,1);d=torch.zeros(4,6,1,device=alg.device,dtype=torch.uint8);d[1,2]=1;d[2,4]=1
 padded,masks=split_and_pad_trajectories(x,d);assert torch.equal(unpad_trajectories(padded,masks),x)
 normalizer=alg.policy.actor_obs_normalizer
 normalizer.update=types.MethodType(lambda self,*a,**k:None,normalizer)
 frozen_normalizer={k:v.detach().clone() for k,v in normalizer.state_dict().items()}
 action_audit={'samples':0,'mean_clipped':torch.zeros(6,device=alg.device),'sample_clipped':torch.zeros(6,device=alg.device)}
 teachers=[];teacher_hashes={};teacher_sources=[]
 if args.training_bc:
  from rsl_rl.runners import OnPolicyRunner
  roots=Path('/workspace/piper-insertion/logs')
  for label,relative in [('axial','controlled_ppo_20261015_r1_train200/model_199.pt'),('lateral',args.training_bc_lateral_checkpoint)]:
   cp=roots/relative;config=json.loads((cp.parent/'agent.json').read_text())
   expert=OnPolicyRunner(env,config,log_dir=None,device=env.device);expert.load(str(cp),load_optimizer=False)
   policy=expert.get_inference_policy(device=env.device);model=expert.alg.policy
   for parameter in model.parameters():parameter.requires_grad_(False)
   teachers.append((label,model,policy));teacher_hashes[label]=model_hash(model)
   teacher_sources.append(dict(role=label,path=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
 alg.training_bc_coefficient=2. if teachers else 0.
 alg.training_bc_losses=[]
 original_process=alg.process_env_step
 def process(self,obs,rewards,dones,extras):
  selected=getattr(raw,'training_student_control',(raw.training_control_mode==0)&~raw.training_control_preparing)
  self.transition.observations['training_control_mask'].copy_(selected.float()[:,None])
  if teachers:
   with torch.no_grad():
    predictions={label:policy(self.transition.observations).detach().clamp(-1,1) for label,model,policy in teachers}
    target=torch.where((raw.case_ids<2)[:,None],predictions['axial'],predictions['lateral'])
    self.transition.observations['training_bc_target'].copy_(target)
    trusted=(raw.case_ids<2)|(raw.training_control_offset*1000<=args.training_bc_trusted_offset_mm+1e-5)
    if args.training_bc_trust_case2:trusted|=raw.case_ids==2
    self.transition.observations['training_bc_weight'].copy_((selected&trusted).float()[:,None])
    for label,model,policy in teachers:model.reset(dones.bool())

  action_audit['samples']+=int(selected.sum())
  action_audit['mean_clipped']+=((self.transition.action_mean.abs()>1)&selected[:,None]).float().sum(0)
  action_audit['sample_clipped']+=((self.transition.actions.abs()>1)&selected[:,None]).float().sum(0)
  return original_process(obs,rewards,dones,extras)
 alg.process_env_step=types.MethodType(process,alg)
 source=textwrap.dedent(inspect.getsource(type(alg).update))
 needle='        num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.'
 assert source.count(needle)==1
 source=source.replace(needle,"""        control=obs_batch['training_control_mask']
        if masks_batch is not None:control=unpad_trajectories(control,masks_batch)
        control=control.squeeze(-1)
        def controlled_mean(value):
            assert value.shape==control.shape,(value.shape,control.shape)
            return (value*control).sum()/control.sum().clamp_min(1.)
"""+needle)
 for before,after in [('kl_mean = torch.mean(kl)','kl_mean = controlled_mean(kl)'),('surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()','surrogate_loss = controlled_mean(torch.max(surrogate, surrogate_clipped))'),('- self.entropy_coef * entropy_batch.mean()','- self.entropy_coef * controlled_mean(entropy_batch)')]:
  assert source.count(before)==1;source=source.replace(before,after)
 loss_needle='        # Symmetry loss\n        if self.symmetry:'
 assert source.count(loss_needle)==1
 source=source.replace(loss_needle,'''        bc_target=obs_batch['training_bc_target']
        bc_weight=obs_batch['training_bc_weight']
        if masks_batch is not None:
            bc_target=unpad_trajectories(bc_target,masks_batch)
            bc_weight=unpad_trajectories(bc_weight,masks_batch)
        bc_weight=bc_weight.squeeze(-1)
        assert mu_batch.shape==bc_target.shape and bc_weight.shape==control.shape
        bc_loss=((mu_batch-bc_target).square().mean(-1)*bc_weight).sum()/bc_weight.sum().clamp_min(1.)
        loss=loss+self.training_bc_coefficient*bc_loss
        self.training_bc_losses.append(float(bc_loss.detach()))

'''+loss_needle)
 namespace=dict(type(alg).update.__globals__);namespace['unpad_trajectories']=unpad_trajectories;exec(compile(source,'<private_controlled_ppo>','exec'),namespace)
 patched=namespace['update'];count=0
 frozen={n:model_hash(m) for n,m in [('withdrawal',raw.recovery_model),('alignment',raw.alignment_model)]+([('initial_insertion',raw.initial_model)] if raw.initial_model is not None else [])}
 def update(self):
  nonlocal count
  mask=self.storage.observations['training_control_mask'];controlled=int(mask.sum());total=mask.numel()
  self.training_bc_losses=[]
  result=patched(self);count+=1
  for n,m,policy in teachers:assert model_hash(m)==teacher_hashes[n],('Frozen training supervisor changed',n)
  for k,v in normalizer.state_dict().items():assert torch.equal(v,frozen_normalizer[k]),('Actor normalization changed',k)
  for n,m in [('withdrawal',raw.recovery_model),('alignment',raw.alignment_model)]+([('initial_insertion',raw.initial_model)] if raw.initial_model is not None else []):assert model_hash(m)==frozen[n],('Frozen skill changed',n)
  audit=dict(update=count,controlled_samples=controlled,total_samples=total,frozen_skill_hashes=frozen,actor_inputs=33,phase_mask_is_not_actor_input=True,actor_normalizer_frozen=True,learning_rate=self.learning_rate)
  audit['training_behavior_constraint']=dict(active=bool(teachers),coefficient=self.training_bc_coefficient,mean_squared_error=sum(self.training_bc_losses)/max(1,len(self.training_bc_losses)),frozen_teacher_hashes=teacher_hashes,case_labels_training_only=True,actions_applied='learner exclusively; teachers shadow observations only')
  audit['unclipped_policy_action_audit']={'controlled_samples':action_audit['samples'],'mean_clip_fraction_by_axis':(action_audit['mean_clipped']/max(1,action_audit['samples'])).tolist(),'sample_clip_fraction_by_axis':(action_audit['sample_clipped']/max(1,action_audit['samples'])).tolist()}
  (Path(logdir)/'controlled_ppo_progress.json').write_text(json.dumps(audit,indent=2))
  if count==1 or count%10==0:print('CONTROLLED_PPO_UPDATE',json.dumps(audit),flush=True)
  return result
 alg.update=types.MethodType(update,alg);alg.schedule='fixed';alg.learning_rate=1e-4
 for group in alg.optimizer.param_groups:group['lr']=1e-4
 (Path(logdir)/'controlled_ppo_implementation.py').write_text(source)
 (Path(logdir)/'controlled_ppo_protocol.json').write_text(json.dumps(dict(status='installed',policy_gradient='main insertion mode only, never preparation/withdrawal/realignment',critic='all actual transitions',phase_mask='training storage only; excluded from policy/critic groups',physical_gates_unchanged=True,frozen_skill_hashes=frozen,learning_rate=1e-4,padding_roundtrip_passed=True),indent=2))
 (Path(logdir)/'retry_curriculum_protocol.json').write_text(json.dumps(dict(training_only=True,early_failure='second actor-only blocked trigger after an actual realignment',success_claim=False,native_force_guard_unchanged=True,full_native_evaluation=True,reason_code=5),indent=2))
 (Path(logdir)/'behavior_constraint_protocol.json').write_text(json.dumps(dict(active=bool(teachers),sources=teacher_sources,trusted_lateral_offset_mm=args.training_bc_trusted_offset_mm,trust_case2=args.training_bc_trust_case2,coefficient=alg.training_bc_coefficient,teachers_frozen=True,actor_inputs=33,case_selection_training_only=True,no_runtime_teacher=True,actual_training_actions='learner exclusively',target='bounded frozen actor output; no true physical state'),indent=2))
 (Path(logdir)/'joint_margin_curriculum_protocol.json').write_text(json.dumps(dict(weight=raw.cfg.training_joint_margin_cost,training_only=True,joint5_risk='clamp((abs(q5)-.65)/.511,0,2)^2',applied_to='actual student-controlled transitions only',actor_input_already_contains_joint5=True,native_success_and_abort_unchanged=True,evaluation_cost=0.),indent=2))
 print('CONTROLLED_PPO_INSTALLED',flush=True)
