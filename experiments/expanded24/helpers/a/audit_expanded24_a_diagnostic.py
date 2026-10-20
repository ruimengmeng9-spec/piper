import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCriticRecurrent
L=Path('/workspace/piper-insertion/logs');G='reliability_v2_expanded24_a_native_20261020';D=L/G;S=L/(G+'_source/executable');sys.path.insert(0,str(S))
from current_recovery import CurrentObsRecoveryActorCriticRecurrent
from joint_input_upgrade import expose_joint5
from boundary_rl import Residual,model_hash
from context_ppo import predict,metrics
from retry_brake import FinalRetryAxialBrake
from guarded_value import value_gate_passed,optimizer_digest
from frozen_boundary_history import precision
from domain_value_upgrade import upgrade_critic,append_domain_context
from history_value import upgrade_history,history_inputs
from history_data import read_npz,load_history,dataset,predict_dataset
from group_balanced_value import assess,rng_digest
from expanded24_a_volume_checks import probe_joint_target,verify_joint_target_arrays,probe_queue,verify_queue_arrays,verify_controller_arrays,verify_release_arrays,verify_latency_arrays,verify_force_encoding_history,verify_force_history,verify_privileged_arrays,mapped,verify_offset_metadata
precision();torch.set_num_threads(2);load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan=load(D/'plan.json');index=[dict(folder=G+'_release_random')];source=L/'reliability_v2_current_recovery3/model_current_recovery.pt';post=L/'specialist_stage_20261015_postrnn1_fit/fitted_recurrent_post_actor.pt';bp=L/'reliability_v2_boundary_pilot1_random/boundary_last.pt'
def model(path,cls):
 kw=load(path.parent/'agent.json')['policy'].copy();kw.pop('class_name');obs=TensorDict({'policy':torch.zeros(1,33),'critic':torch.zeros(1,33)},batch_size=[1]);m=cls(obs,{'policy':['policy'],'critic':['critic']},6,**kw);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state_dict']);m.eval();return m
root=model(source,CurrentObsRecoveryActorCriticRecurrent);initial=copy.deepcopy(root);expose_joint5(initial);m=copy.deepcopy(initial);main=model(post,ActorCriticRecurrent);b=Residual();b.eval();original73=torch.load(bp,map_location='cpu',weights_only=False)['state_dict'];b.load_state_dict(original73);upgrade_critic(b);upgrade_history(b);original=copy.deepcopy(b.state_dict());errors=dict(actor=0.,root_anchor=0.,normalization_features=0.,original_boundary_GRU=0.,boundary_delta=0.,prior_value=0.,MC_return=0.,log_probability=0.);probe_steps=0;probe_selected=0;checked=[];domain_hashes=[];chunks=[]
for number,item in enumerate(index):
 F=L/item['folder'];r=load(F/'domain_collect_result.json');assert r['index']==number and r['seed']==1125;rows=[json.loads(s) for s in (F/'episode_outcomes.jsonl').read_text().splitlines()];assert len(rows)==len({q['env'] for q in rows})==128 and sum(q['success'] for q in rows)==r['successes'] and sum(q['peak_force_n']>100 for q in rows)==r['force_aborts'];assert r['actor_updates']==r['value_fit_attempts']==0 and r['conditioned'] and r['critic_dimensions']==75 and r['training_forbidden']
 snap=torch.load(F/'frozen_snapshot.pt',map_location='cpu',weights_only=False);assert snap['final_retry_release'] and r['final_retry_release'];assert all(torch.equal(v,initial.state_dict()[k]) for k,v in snap['deep_state_dict'].items());m.load_state_dict(snap['deep_state_dict']);assert model_hash(m)==r['deep_state_hash'];expected=torch.load(D/'boundary_value.pt',map_location='cpu',weights_only=False)['state_dict'] if r['blind'] else original;assert all(torch.equal(v,expected[k]) for k,v in snap['boundary_state_dict'].items());b.load_state_dict(expected);assert model_hash(b)==r['boundary_model_hash'];assert all(k in snap for k in ['torch_rng','cuda_rng','numpy_rng','python_rng']);assert r['new_joint_column_max_abs']==0
 domains=read_npz(F/'domain_parameters.npz');hs={n:hashlib.sha256(domains[n].tobytes()).hexdigest() for n in domains.files};assert hs==r['domain_hashes'];assert all(np.array_equal(snap['domain'][n].numpy(),domains[n]) for n in domains.files);domain_hashes.append(r['domain_joint_hash']);assert len(set(domain_hashes))==len(domain_hashes)
 X,Alive,Done=load_history(F/'complete_history.npz');FX,FA,FD=load_history(F/'complete_force_history.npz');assert np.array_equal(FA,Alive) and np.array_equal(FD,Done);
 with torch.no_grad():History=b.critic.sequence(torch.from_numpy(X),torch.from_numpy(FX))
 z=read_npz(F/'native_probes.npz');z={k:z[k] for k in z.files};alive=torch.from_numpy(z['alive']);selected=torch.from_numpy(z['selected']);assert not (selected&~alive).any();public=torch.from_numpy(z['public33']);deep_public=public if bool(z['public_shared']) else torch.from_numpy(z['deep_public33']);features=torch.from_numpy(z['deep_features']);old=torch.from_numpy(z['old_mean']);latent=torch.from_numpy(z['latent']);C=torch.from_numpy(z['critic_features']);mask=torch.from_numpy(z['boundary_mask']);delta=torch.from_numpy(z['delta']);base=torch.from_numpy(z['base']);composed=torch.from_numpy(z['composed']);probe_steps+=int(alive.sum());probe_selected+=int(selected.sum())
 with torch.no_grad():
  x=m.actor_obs_normalizer(deep_public);f=m.current_features(x);mu=m.actor(f);anchor=root.actor(root.current_features(root.actor_obs_normalizer(deep_public)));lp=torch.distributions.Normal(mu,torch.from_numpy(z['sigma'])).log_prob(latent).sum(-1);value=b.critic.combine_expected(C,History[:,z['env']],torch.from_numpy(z['critic_privileged6']),torch.from_numpy(z['critic_latency1']),torch.from_numpy(z['critic_release1']),torch.from_numpy(z['critic_control4']),torch.from_numpy(probe_queue(z)),torch.from_numpy(probe_joint_target(z)))[:,:,0]*100.;nx=main.actor_obs_normalizer(public);h,_=main.memory_a.rnn(nx,torch.zeros(1,public.shape[1],33));cf=torch.cat((nx.clamp(-8,8),h),-1);d=b.scale*torch.tanh(b.mean(C[:,:,:66]))
 for k,q in [('actor',abs(mu-old)[alive]),('root_anchor',abs(anchor-torch.from_numpy(z['root_mean']))[alive]),('normalization_features',abs(f-features)[alive]),('original_boundary_GRU',abs(cf-C[:,:,:66])[alive]),('boundary_delta',abs(d-delta)[alive]),('prior_value',abs(value-torch.from_numpy(z['prior_value']))[alive]),('log_probability',abs(lp-torch.from_numpy(z['old_logp']))[selected])]:
  if q.numel():errors[k]=max(errors[k],float(q.max()))
 before_brake=torch.from_numpy(z['before_brake']);br=torch.from_numpy(z['brake_state']);assert torch.equal(before_brake,base+torch.where(mask,delta,0.));assert torch.equal(before_brake,br[:,:,10:16]);guard=FinalRetryAxialBrake(public.shape[1],'cpu',release_on=True)
 for step in range(len(br)):
  q=br[step];projected,chosen=guard.apply(public[step],before_brake[step],q[:,0].long(),q[:,1].long(),q[:,2].long(),q[:,3].long(),q[:,4].bool()|~q[:,6].bool(),q[:,5].bool());assert torch.equal(chosen,q[:,7].bool()) and torch.equal(guard.latched,q[:,8].bool()) and torch.equal(projected,composed[step]) and torch.equal(projected,q[:,16:22]);guard.reset(torch.from_numpy(z['done'][step]))
 assert torch.equal(base[selected],latent[selected]) and torch.equal(composed[selected],before_brake[selected]);clip=np.clip(z['composed'],-1,1);clip[z['refresh']]=0;assert np.array_equal(clip,z['clipped'])
 for e in range(public.shape[1]):
  for step in np.flatnonzero(z['alive'][:,e]):
   delay=int(z['delay_steps'][step,e]);expected=z['clipped'][step-delay,e] if step>=delay else np.zeros(6,dtype=np.float32);assert np.array_equal(expected,z['delayed'][step,e])
 reward=torch.from_numpy(z['reward']);done=torch.from_numpy(z['done']);returns=torch.zeros_like(reward);future=torch.zeros(public.shape[1])
 for step in reversed(range(len(reward))):future=reward[step]+.99*future*(~done[step]);returns[step]=future
 ids=torch.from_numpy(z['env']);dc=torch.stack(((torch.from_numpy(domains['friction_scales'])[ids]-1)/.4,(torch.from_numpy(domains['stiffness_scales'])[ids]-1)/.2),-1);assert torch.equal(C[:,:,73:],dc[None].expand(len(C),-1,-1))
 q=read_npz(F/'selected_data.npz');verify_joint_target_arrays(q,z,r);verify_queue_arrays(q,z,r);verify_privileged_arrays(q,z);verify_controller_arrays(q,z);verify_release_arrays(q,z);verify_force_history(FX,FA,FD,q,z);verify_force_encoding_history(FX,FA,r);verify_latency_arrays(q,z,read_npz(F/'case_metadata.npz'));data={k:q[k] for k in q.files};eids=q['env'];held=((eids//4)%4)==0;assert ((eids//128)==number).all();assert r['heldout_steps']==int(held.sum()) and r['train_steps']==int((~held).sum()) and r['selected_steps']==len(eids);case_meta=read_npz(F/'case_metadata.npz');verify_offset_metadata(case_meta,r,5);cases=case_meta['case_ids'];expected_cases=np.arange(128)%4;assert np.array_equal(cases,expected_cases) and r['case_profile']=='all';assert np.array_equal(q['case_id'],cases[eids%128]);te=len(np.unique(eids[~held]));he=len(np.unique(eids[held]));assert te==r['deep_training_episodes'] and he==r['deep_heldout_episodes'];assert r['coverage_passed']==(int(held.sum())>200 and int((~held).sum())>500)
 with torch.no_grad():err=float(abs(b.critic.combine_expected(torch.from_numpy(q['critic_features']),History[q['tick'],q['env']%128],torch.from_numpy(q['critic_privileged6']),torch.from_numpy(q['critic_latency1']),torch.from_numpy(q['critic_release1']),torch.from_numpy(q['critic_control4']),torch.from_numpy(q['critic_queue12']),torch.from_numpy(q['critic_joint_target6']))[:,0]*100.-torch.from_numpy(q['prior_value'])).max()) if len(eids) else 0.
 errors['prior_value']=max(errors['prior_value'],err)
 assert np.array_equal(X[q['tick'],q['env']%128],mapped(q['critic_features'],q['public_force3']));assert np.array_equal(Alive[:,z['env']],z['alive']);assert np.array_equal(X[:,z['env']][z['alive']],mapped(z['critic_features'],z['public33'][...,24:27])[z['alive']]);cc=torch.from_numpy(q['critic_features']);envs=torch.from_numpy(q['env']%128);expected75=append_domain_context(cc[:,:73].contiguous(),torch.from_numpy(domains['friction_scales'])[envs],torch.from_numpy(domains['stiffness_scales'])[envs]);assert torch.equal(cc,expected75)
 for e,env in enumerate(z['env']):
  selected_env=eids==int(env)+128*number;s=z['selected'][:,e];assert int(selected_env.sum())==int(s.sum());assert np.allclose(q['critic_features'][selected_env],z['critic_features'][s,e],rtol=0,atol=1e-6);assert np.array_equal(q['phase'][selected_env],z['phase'][s,e])
  assert np.array_equal(q['public_force3'][selected_env],z['public33'][s,e,24:27])
  if selected_env.any():errors['MC_return']=max(errors['MC_return'],float(abs(torch.from_numpy(q['target'][selected_env])-returns[s,e]).max()))
 if held.any():
  score=metrics(torch.from_numpy(q['prior_value']),torch.from_numpy(q['target']),torch.from_numpy(q['phase']),torch.from_numpy(held));assert value_gate_passed(score)==r['value_gate_passed'];assert np.isclose(score['mse_ratio'],r['prior_metrics']['mse_ratio'],rtol=2e-5,atol=2e-5)
 else:score=None
 assert r['blind_gate_passed']==bool(r['blind'] and r['force_gate_passed'] and r['coverage_passed'] and score and value_gate_passed(score))
 if not r['blind']:chunks.append(data)
 checked.append(dict(actual_case_ids_verified=True,actual_reset_offsets_verified=True,deep_training_episodes=te,deep_heldout_episodes=he,seed=r['seed'],index=number,successes=r['successes'],force_aborts=r['force_aborts'],selected_steps=r['selected_steps'],heldout_steps=r['heldout_steps'],blind=r['blind'],blind_gate_passed=r['blind_gate_passed']))
assert max(errors.values())<1e-3,errors
report=dict(explicit_predecision_joint_target6_independently_verified=True,explicit_preaction_queue12_independently_verified=True,complete_all_env_trueforce3_independently_replayed=True,status="passed",diagnostic_only=True,not_independent_acceptance=True,training_forbidden=True,controller_policy_changed=True,actor_updates=0,value_fit_attempts=0,episodes=128,errors=errors,checked=checked)
(D/"saved_checkpoint_audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
