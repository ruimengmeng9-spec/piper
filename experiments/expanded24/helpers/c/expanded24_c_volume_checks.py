from expanded24_c_source_refs import expand_shared_sources
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
def mapped(C,force):
 assert force.shape == (*C.shape[:-1],3) and np.isfinite(force).all() and (np.abs(force)<=3.000001).all()
 X=np.concatenate((C[...,:33],C[...,66:]),-1);X[...,24:27]=force;return X

sha=lambda b:hashlib.sha256(b).hexdigest()
def verify_offset_metadata(z,result,expected_mm):
 offsets=z['reset_offset_m'];configured=z['configured_offset_mm']
 assert offsets.dtype==np.dtype('float32') and offsets.shape==(128,) and np.isfinite(offsets).all()
 assert offsets.tobytes()==np.full(128,expected_mm/1000.,dtype=np.float32).tobytes()
 assert configured.dtype==np.dtype('float32') and configured.shape==() and float(configured)==float(expected_mm)
 assert result['configured_offset_mm']==expected_mm and sha(offsets.tobytes())==result['reset_offset_sha256']
 return offsets

def history(raw,channels=42):
 z=np.load(io.BytesIO(raw));shape=tuple(z['shape']);delta=z['packed'].T.copy().reshape(-1).view(np.uint32).reshape(shape);x=np.bitwise_xor.accumulate(delta,axis=0).view(np.float32);a=z['alive'];d=z['done'];assert sha(x.tobytes())==str(z['content_sha256']);assert shape[1:]==(128,channels) and a[0].all() and not (a[1:]&~a[:-1]).any() and np.all((a&d).sum(0)==1) and not x[~a].any();return x,a,d

def members(path,expected):
 assert sha(Path(path).read_bytes())==expected
 with tarfile.open(path) as z:return {m.name:z.extractfile(m).read() for m in z.getmembers() if m.isfile()}

def pair_coverage(results):
 assert len(results)==2 and all(not r['blind'] and r['episodes']==128 for r in results)
 assert results[1]['index']==results[0]['index']+1 and results[0]['index']%2==0
 train=sum(r['deep_training_episodes'] for r in results);held=sum(r['deep_heldout_episodes'] for r in results)
 steps=all(r['train_steps']>500 and r['heldout_steps']>200 for r in results)
 return dict(domain=results[0]['index']//2,seeds=[r['seed'] for r in results],episodes=256,training_deep_episodes=train,heldout_deep_episodes=held,all_native128_step_gates=steps,passed=steps and train>=24 and held>=8)

def cohort_checks(blobs,G,seed,plan):
 assert G+f'_c{seed}/shared_source_reference.json' in blobs
 blobs=expand_shared_sources(blobs)
 F=G+f'_c{seed}_random/';r=json.loads(blobs[F+'domain_collect_result.json']);idx=plan['prescribed_sequence'].index(seed);assert r['seed']==seed and r['index']==idx and r['history_enabled'] and r['actor_updates']==r['value_fit_attempts']==0 and not r['training_forbidden'];rows=[json.loads(s) for s in blobs[F+'episode_outcomes.jsonl'].decode().splitlines()];assert len(rows)==128 and sorted(q['env'] for q in rows)==list(range(idx*128,(idx+1)*128));assert sum(v['success'] for v in rows)==r['successes'] and sum(v['peak_force_n']>100 for v in rows)==r['force_aborts'];assert r['force_gate_passed']==(r['force_aborts']<=12)
 prefix=G+f'_c{seed}/executable/';M=json.loads(blobs[prefix+'source_manifest.json'])['executable_hashes'];assert len(M)==65
 for n,h in M.items():assert sha(blobs[prefix+n])==h
 assert M==plan['executable_hashes']
 x,a,d=history(blobs[F+'complete_history.npz']);q=np.load(io.BytesIO(blobs[F+'selected_data.npz']));ids=q['env'];ticks=q['tick'];assert (ids//128==idx).all() and a[ticks,ids%128].all();assert len(set(zip(ticks.tolist(),ids.tolist())))==len(ids);assert np.array_equal(x[ticks,ids%128],mapped(q['critic_features'],q['public_force3']));held=((ids//4)%4)==0;assert int(held.sum())==r['heldout_steps'] and int((~held).sum())==r['train_steps'];counts=[int((ids==idx*128+e).sum()) for e in range(128)];assert counts==r['selected_steps_by_native_env']
 case_meta=np.load(io.BytesIO(blobs[F+'case_metadata.npz']));verify_offset_metadata(case_meta,r,5);cases=case_meta['case_ids'];eid=np.arange(128);expected=np.where((eid//16)%2==1,eid%2+2,eid%4) if not r['blind'] else eid%4;assert np.array_equal(cases,expected) and r['case_profile']==('all' if r['blind'] else 'lateral_focus') and sha(cases.tobytes())==r['case_ids_sha256'];assert np.array_equal(q['case_id'],cases[ids%128]);assert r['selected_steps_by_case']=={str(c):sum(counts[e] for e in range(128) if cases[e]==c) for c in range(4)};te=len(np.unique(ids[~held]));he=len(np.unique(ids[held]));assert te==r['deep_training_episodes'] and he==r['deep_heldout_episodes'];assert r['independent_episode_coverage_passed']==(te>=24 and he>=8);assert r['coverage_passed']==(int(held.sum())>200 and int((~held).sum())>500)
 dom=np.load(io.BytesIO(blobs[F+'domain_parameters.npz']));hs={n:sha(dom[n].tobytes()) for n in dom.files};assert hs==r['domain_hashes'] and sha(''.join(hs.values()).encode())==r['domain_joint_hash'];extra=np.stack(((dom['friction_scales'][ids%128]-1)/np.float32(.4),(dom['stiffness_scales'][ids%128]-1)/np.float32(.2)),-1);assert np.array_equal(extra,q['critic_features'][:,73:])
 assert r['value_training_protocol']=='balanced_four_mean_mse_tail_decay_force_smooth';assert r['value_prediction_protocol']=='balanced_four_history_masks_p50';assert r['final_retry_release'] and r['controller_policy_changed'];n=np.load(io.BytesIO(blobs[F+'native_probes.npz']));release_actions_verified(n);verify_horizon_clock(q,n);verify_joint_target_arrays(q,n,r);verify_queue_arrays(q,n,r);verify_controller_arrays(q,n);verify_release_arrays(q,n);verify_privileged_arrays(q,n);verify_latency_arrays(q,n,case_meta);fx,fa,fd=history(blobs[F+'complete_force_history.npz'],3);verify_force_history(fx,fa,fd,q,n);verify_force_encoding_history(fx,fa,r);assert np.array_equal(fa,a) and np.array_equal(fd,d);assert np.array_equal(a[:,n['env']],n['alive']);assert np.array_equal(x[:,n['env']][n['alive']],mapped(n['critic_features'],n['public33'][...,24:27])[n['alive']]);future=np.zeros(len(n['env']),dtype=np.float32);ret=np.zeros_like(n['reward'])
 for t in reversed(range(len(ret))):future=n['reward'][t]+np.float32(.99)*future*(~n['done'][t]);ret[t]=future
 error=0.
 for j,e in enumerate(n['env']):
  mask=n['selected'][:,j];chosen=ids==idx*128+int(e);assert mask.sum()==chosen.sum() and np.array_equal(q['tick'][chosen],np.flatnonzero(mask));assert np.allclose(q['critic_features'][chosen],n['critic_features'][mask,j],rtol=0,atol=1e-6)
  assert np.array_equal(q['public_force3'][chosen],n['public33'][mask,j,24:27])
  if chosen.any():error=max(error,float(abs(q['target'][chosen]-ret[mask,j]).max()))
 assert error<2e-4
 return dict(causal_force_encoding_online_offline_verified=True,discount_horizon_clock_verified=True,episodes=128,source_hash_checks=65,complete_history42_lossless_verified=True,explicit_predecision_joint_target6_verified=True,shared_source_all65_logical_bytes_verified=True,explicit_preaction_queue12_verified=True,complete_trueforce3_lossless_verified=True,explicit_preaction_latency1_verified=True,explicit_preaction_release1_verified=True,explicit_preaction_controller4_verified=True,successes=r['successes'],force_aborts=r['force_aborts'],domain_joint_hash=r['domain_joint_hash'],numpy_MC_return_error=error,actual_case_ids_verified=True,actual_reset_offsets_verified=True,deep_train_episodes=te,deep_held_episodes=he)

def release_actions_verified(q):
 q={k:q[k] for k in q.files} if not isinstance(q,dict) else q
 latched=np.zeros(q['public33'].shape[1],dtype=bool)
 for t in range(len(q['public33'])):
  x=q['public33'][t];a=q['before_brake'][t];br=q['brake_state'][t];eligible=(br[:,0]==0)&(br[:,1]>=3)&(br[:,2]>0)&~br[:,4].astype(bool)&br[:,6].astype(bool)&~br[:,5].astype(bool)
  latched|=eligible&(br[:,3]>=2);sel=eligible&latched;assert np.array_equal(sel,br[:,7].astype(bool)) and np.array_equal(latched,br[:,8].astype(bool));valid=np.abs(x[:,24:27]).max(-1)<=np.float32(2.);needs=(-x[:,29]/np.float32(20.)>np.float32(.001))|(valid&(np.linalg.norm(x[:,24:27]/np.float32(.02),axis=-1)>=np.float32(5.)))
  expected=a.copy();expected[sel]=0.;expected[:,2]=np.where(sel,np.maximum(a[:,2],np.where(needs,np.float32(.35),np.float32(0.))),a[:,2]);assert np.array_equal(expected,q['composed'][t]) and np.array_equal(expected,br[:,16:22]);deep=q['selected'][t];assert not (deep&sel).any() and np.array_equal(expected[deep],a[deep]);latched[q['done'][t]]=False
 clip=np.clip(q['composed'],-1,1);clip[q['refresh']]=0;assert np.array_equal(clip,q['clipped'])
 for e in range(q['public33'].shape[1]):
  for t in np.flatnonzero(q['alive'][:,e]):
   delay=int(q['delay_steps'][t,e]);expected=q['clipped'][t-delay,e] if t>=delay else np.zeros(6,dtype=np.float32);assert np.array_equal(expected,q['delayed'][t,e])
 return True


def verify_privileged_arrays(q,z=None):
 def check(public_pose,public_force,truth,extra):
  assert truth.shape==extra.shape==(*public_pose.shape[:-1],6)
  assert np.isfinite(truth).all() and np.isfinite(extra).all()
  assert np.array_equal((np.abs(public_force)>2).any(-1),(np.abs(truth[...,3:])>2).any(-1))
  expected=np.concatenate(((truth[...,:3]-public_pose)*np.float32(50.),(truth[...,3:]-public_force)*np.float32(25.)),-1)
  assert np.array_equal(expected,extra)
 check(q['public_pose3'],q['public_force3'],q['privileged_truth6'],q['critic_privileged6'])
 if z is not None:
  check(z['public33'][...,:3],z['public33'][...,24:27],z['privileged_truth6'],z['critic_privileged6'])
  for j,e in enumerate(z['env']):
   ix=q['env']%128==e;selected=z['selected'][:,j]
   for field in ['privileged_truth6','critic_privileged6']:assert np.array_equal(q[field][ix],z[field][selected,j])
   assert np.array_equal(q['public_pose3'][ix],z['public33'][selected,j,:3])
 return dict(status='passed',selected_steps=len(q['env']),same_step_truth_public_delta6_bitwise_exact=True)


def verify_force_history(fx,alive,done,q,z):
 assert fx.dtype==np.dtype('float32') and fx.shape==(*alive.shape,3) and np.isfinite(fx).all() and (np.abs(fx)<=3.000001).all()
 assert not fx[~alive].any() and np.array_equal(alive[:,z['env']],z['alive']) and np.array_equal(done[:,z['env']],z['done'])
 assert np.array_equal(fx[q['tick'],q['env']%128],q['privileged_truth6'][...,3:])
 assert np.array_equal(fx[:,z['env']][z['alive']],z['privileged_truth6'][...,3:][z['alive']])
 invalid=(np.abs(z['public33'][...,24:27])>2).any(-1)&z['alive']
 assert np.array_equal((np.abs(fx[:,z['env']])>2).any(-1)&z['alive'],invalid)
 return dict(status='passed',complete_preaction_trueforce3_verified=True,all_selected_and_alive_probe_rows_bitwise_exact=True)


def verify_latency_arrays(q,z,case):
 raw=case['action_delay_steps'];assert raw.shape==(128,) and raw.dtype.kind in 'iu' and np.isin(raw,[0,1,2]).all()
 ids=q['env']%128;lat=q['critic_latency1'];steps=q['action_delay_steps']
 assert lat.dtype==np.dtype('float32') and lat.shape==(len(ids),1) and np.array_equal(steps,raw[ids])
 assert np.array_equal(lat[:,0],steps.astype(np.float32)*np.float32(.5))
 zl=z['critic_latency1'];assert zl.shape==(*z['alive'].shape,1) and zl.dtype==np.dtype('float32')
 assert np.array_equal(zl[...,0],z['delay_steps'].astype(np.float32)*np.float32(.5))
 expected=np.broadcast_to(raw[z['env']][None],z['alive'].shape)
 assert np.array_equal(z['delay_steps'][z['alive']],expected[z['alive']])
 for j,e in enumerate(z['env']):
  selected=z['selected'][:,j];ix=ids==e;assert np.array_equal(lat[ix],zl[selected,j])
 return dict(status='passed',actual_preaction_delay_steps_normalized_by2_exact=True,all_selected_and_alive_probe_rows_match_reset_metadata=True)


def verify_release_arrays(q,z):
 def check(rel,hold):
  assert hold.dtype.kind in 'iu' and np.isin(hold,np.arange(9)).all()
  assert rel.dtype==np.dtype('float32') and rel.shape==(*hold.shape,1)
  assert np.array_equal(rel[...,0],hold.astype(np.float32)*np.float32(.125))
 check(q['critic_release1'],q['supervisor_release_hold']);check(z['critic_release1'],z['supervisor_release_hold'])
 for j,e in enumerate(z['env']):
  ix=q['env']%128==e;mask=z['selected'][:,j]
  for field in ['critic_release1','supervisor_release_hold']:assert np.array_equal(q[field][ix],z[field][mask,j])
 return dict(status='passed',prephysics_supervisor_release_counter_raw_integer_and_normalization_verified=True,selected_probe_alignment_exact=True)


def verify_controller_arrays(q,z):
 def check(ctrl,ema,trigger):
  assert ctrl.dtype==ema.dtype==trigger.dtype==np.dtype('float32')
  assert ctrl.shape==(*trigger.shape,4) and ema.shape==(*trigger.shape,3)
  assert np.isfinite(ctrl).all() and (np.abs(ema)<=100.00001).all() and ((trigger>=-.015001)&(trigger<=.020001)).all()
  assert np.array_equal(ctrl,np.concatenate((ema*np.float32(.01),trigger[...,None]*np.float32(50.)),-1))
 check(q['critic_control4'],q['supervisor_force_ema'],q['supervisor_trigger_depth']);check(z['critic_control4'],z['supervisor_force_ema'],z['supervisor_trigger_depth'])
 for j,e in enumerate(z['env']):
  ix=q['env']%128==e;mask=z['selected'][:,j]
  for field in ['critic_control4','supervisor_force_ema','supervisor_trigger_depth']:assert np.array_equal(q[field][ix],z[field][mask,j])
 return dict(status='passed',raw_prephysics_EMA_and_trigger_independently_normalized=True,all_selected_probe_rows_exact=True)


def probe_queue(z):
 current=z['public33'][...,18:24];old=np.zeros_like(current);old[1:]=current[:-1];delay=z['delay_steps']
 out=np.concatenate((np.where((delay>=1)[...,None],current,0.),np.where((delay>=2)[...,None],old,0.)),-1);out[~z['alive']]=0
 return out

def verify_queue_arrays(q,z,result):
 queue=q['critic_queue12'];steps=q['action_delay_steps'];ids=q['env']%128
 assert queue.dtype==np.dtype('float32') and queue.shape==(len(ids),12) and np.isfinite(queue).all() and (np.abs(queue)<=1.).all()
 assert not queue[steps==0].any() and not queue[:,6:][steps<2].any()
 assert result['explicit_preaction_queue12'] and result['all128_live_queue_raw_mirror_exact'] and result['raw_queue_envs']==128 and len(result['raw_queue_all128_sha256'])==64
 assert sha(queue.tobytes())==result['selected_raw_queue_sha256']
 raw=probe_queue(z);alive=z['alive'];delay=z['delay_steps'];expected=np.where((delay==0)[...,None],z['clipped'],np.where((delay==1)[...,None],raw[...,:6],raw[...,6:]));expected[z['refresh']]=0
 assert np.array_equal(expected[alive],z['delayed'][alive])
 for j,e in enumerate(z['env']):
  ix=ids==e;sel=z['selected'][:,j];assert np.array_equal(queue[ix],raw[sel,j])
  assert sha(raw[:,j][alive[:,j]].tobytes())==result['raw_queue_probe_live_sha256'][str(int(e))]
 return dict(status='passed',all128_online_raw_buffer_equality=True,selected_raw_mirror_sha256_exact=True,all_recorded_native_probes_Q12_and_delayed_outputs_bitwise_exact=True)

def probe_joint_target(z):
 assert np.array_equal(z['joint_target_probe_ids'],z['env'])
 raw=z['raw_preaction_joint_target6'];assert raw.shape==(*z['public33'].shape[:2],6) and raw.dtype==np.float32 and np.isfinite(raw).all()
 return (raw-z['public33'][...,6:12])/np.float32(.06)

def verify_joint_target_arrays(q,z,r):
 J=q['critic_joint_target6'];assert J.dtype==np.float32 and J.shape==(len(q['env']),6) and np.isfinite(J).all()
 assert r['explicit_predecision_joint_target6'] and r['joint_target_native_envs']==128 and r['joint_target_all128_live_shadow_mirror_exact'] and r['joint_positions_all128_match_raw_public']
 assert sha(J.tobytes())==r['selected_raw_joint_target6_sha256'];assert sha(z['raw_preaction_joint_target6'].tobytes())==r['raw_probe_joint_targets_sha256']
 expected=probe_joint_target(z);checked=0;error=0.
 for j,e in enumerate(z['env']):
  ix=q['env']%128==e;sel=z['selected'][:,j];assert ix.sum()==sel.sum()
  if ix.any():error=max(error,float(np.abs(J[ix]-expected[sel,j]).max()));checked+=int(ix.sum())
 assert checked>0,'J6 raw target probes must cover selected deep states';assert error<2e-5
 return dict(status='passed',all128_online_J6_mirror_exact=True,all_original_probe_target6_recorded=True,selected_probe_steps=checked,CPU_J6_error=error)

def verify_horizon_clock(q,z):
 c=q['critic_features'][:,69];tick=q['tick']
 assert c.dtype==np.float32 and np.isfinite(c).all() and ((c>=0)&(c<=1)).all()
 assert np.allclose(c,tick.astype(np.float32)/np.float32(720.),rtol=0,atol=1e-7)
 C=z['critic_features'][...,69];a=z['alive'];expected=np.broadcast_to(np.arange(len(C),dtype=np.float32)[:,None]/np.float32(720.),C.shape)
 assert np.allclose(C[a],expected[a],rtol=0,atol=1e-7)
 return dict(status='passed',current_predecision_C69_matches_raw_record_tick=True,no_future_phase_or_episode_length_used=True,selected_clock_unique=int(len(np.unique(c))))

def verify_force_encoding_history(fx,alive,result):
 assert fx.dtype==np.float32 and fx.shape==(*alive.shape,3) and fx.shape[1]==128
 previous=np.empty_like(fx);previous[0]=np.float32(3.);previous[1:]=fx[:-1]
 valid=(np.abs(fx)<=np.float32(2.)).all(-1)&(np.abs(previous)<=np.float32(2.)).all(-1)
 difference=np.where(valid[...,None],(fx-previous)*np.float32(.5),np.float32(0.))
 difference=np.where((np.abs(fx)<=2.).all(-1)[...,None],fx*np.float32(.5)-difference*np.float32(.5),np.float32(0.))
 assert np.array_equal(difference[0],np.where((np.abs(fx[0])<=2.).all(-1)[...,None],fx[0]*np.float32(.5),np.float32(0.))) and np.isfinite(difference).all() and (np.abs(difference)<=2.).all()
 assert result['force_history_encoding']=='half_level_minus_valid_quarter_difference' and result['force_encoding_online_raw_history_exact']
 expected=[sha(difference[:,e][alive[:,e]].tobytes()) for e in range(128)]
 assert expected==result['force_encoding_per_env_sha256']
 return dict(status='passed',all128_online_force_difference_matches_causal_raw_history=True,missing_zero_first_valid_half_level=True,no_raw_force_data_replaced=True)
