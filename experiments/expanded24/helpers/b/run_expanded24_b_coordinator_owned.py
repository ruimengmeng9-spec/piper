from expanded24_resource_meter import check_budget
import shutil as _storage_shutil
def _namespace_free_bytes(target):
    assert _storage_shutil.disk_usage('/data/shijinsheng/piper-insertion').free>=5045649408,'Original data filesystem reserve must remain intact'
    return _storage_shutil.disk_usage(target).free
import datetime,hashlib,json,shutil,subprocess,tarfile,time,traceback
from pathlib import Path
def pair_coverage(results):
 assert len(results)==2 and all(not r['blind'] and r['episodes']==128 for r in results)
 assert results[1]['index']==results[0]['index']+1 and results[0]['index']%2==0
 train=sum(r['deep_training_episodes'] for r in results);held=sum(r['deep_heldout_episodes'] for r in results)
 steps=all(r['train_steps']>500 and r['heldout_steps']>200 for r in results)
 return dict(domain=results[0]['index']//2,seeds=[r['seed'] for r in results],episodes=256,training_deep_episodes=train,heldout_deep_episodes=held,all_native128_step_gates=steps,passed=steps and train>=24 and held>=8)


L=Path('/home/shijinsheng/piper-training-workspace-20261019/logs');R=L.parent;G='reliability_v2_expanded24_b_20261020';D=L/G;S=L/(G+'_source');I='sha256:0c1687cdae03921a28307bb3c0471174b69d37741c3cbb9a68cfd4e1640fbba8';load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=load(D/'plan.json');roles=load(L/'reliability_v2_guarded_recurrent_20261017/plan.json')['roles'];parents=load(L/'reliability_v2_brake_recovery_20261017/plan.json')['checkpoint_hashes'];collected=load(D/'collection_index.json') if (D/'collection_index.json').exists() else [];volumes=[v for item in collected for v in item['archives']]
def status(phase,**kw):
 v=dict(server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,**kw);(D/'status.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
def size(folder):return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
def verify():
 assert sha(L/'accepted_piper_bundle_20261016_signal_ema.tar.gz')==p['parent_accepted_bundle_sha256'];assert all(sha(L/n)==h for n,h in parents.items());free=_namespace_free_bytes(L);assert free>=5045649408 and check_budget(p)['bank_owned_net_bytes']<=201326592
 retained=(L/'expanded24_shared_source_20261020.tar.gz').stat().st_size+size(D)+size(S)+sum(size(f) for f in L.glob(G+'_c*') if f.is_dir())+sum(f.stat().st_size for f in L.glob(G+'_c*.tar.gz'))+sum((L/n).stat().st_size for n in p['helper_hashes']);assert retained<=167772160
 return retained
def container(script,gpu=False):
 cmd=['docker','run','--rm','--read-only']+(['--gpus','device=2'] if gpu else [])+['--tmpfs','/tmp:rw,size=128m','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','--cpus=2','--memory=3g','--network','none','--entrypoint','/isaac-sim/python.sh','-v',str(R)+':/workspace/piper-insertion','-v',str(R)+':'+str(R)+':ro','-v','/data/shijinsheng/piper-insertion:/data/shijinsheng/piper-insertion:ro',I,'/workspace/piper-insertion/logs/'+script];subprocess.run(cmd,check=True)
try:
 assert p['stage']=='collect_bank_B_only' and p['collection_seeds']==p['training_seeds'][8:16] and p['prescribed_sequence']==p['training_seeds'] and not (D/'value_fit_result.json').exists();assert p['training_seeds']==[2811,2813,2815,2817,2819,2821,2823,2825,2827,2829,2831,2833,2835,2837,2839,2841,2843,2845,2847,2849,2851,2853,2855,2857] and p['blind_gate_seeds']==[2873,2889];proof=load(L/'reliability_v2_expanded24_a_native_20261020/local_verification.json');assert proof['status']=='passed' and proof['training_authorized'] and proof['source_archive_sha256']==p['source_archive_sha256'];assert sha(Path(proof['server_archive_path']))==proof['archive_sha256']==p['native_equivalence_archive_sha256'];(D/'native_equivalence_reference.json').write_text(json.dumps(proof,indent=2));assert load(D/'seed_usage_audit.json')['status']=='passed'
 for n,h in p['helper_hashes'].items():assert sha(L/n)==h
 M=load(S/'executable/source_manifest.json')['executable_hashes'];assert len(M)==65 and all(sha(S/'executable'/n)==h for n,h in M.items())
 p.setdefault('initial_free_bytes',_namespace_free_bytes(L));assert p['initial_free_bytes']>=5246976000;(D/'plan.json').write_text(json.dumps(p,indent=2));stop=None
 assert p['predecessor_group']=='reliability_v2_expanded24_a_20261020' and p['fit_forbidden_for_this_bank'] and p['maximum_real_data_fit_attempts']==0
 for n,h in p['predecessor_file_hashes'].items():assert sha(L/p['predecessor_group']/n)==h
 assert load(L/p['predecessor_group']/'result.json')['status']=='training_bank_A_ready'
 for local_index,seed in enumerate(p['collection_seeds']):
  index=8+local_index
  if local_index<len(collected):continue
  verify()
  if size(D)+size(S)+(L/'expanded24_shared_source_20261020.tar.gz').stat().st_size>4194304:stop='stopped_group_metadata_forecast_before_next_collection';break
  assert not subprocess.check_output(['docker','ps','--filter','name=shijinsheng-piper','--format','{{.Names}}'],text=True).strip();assert _namespace_free_bytes(L)>=5045649408+33554432;status('collecting_training_domain',seed=seed,index=index,actor_updates=0)
  job=G+f'_c{seed}';cmd=['python3',str(L/'run_expanded24_b_job.py'),'--group',job,'--source',G+'_source','--seed',str(seed-1),'--profiles','random','--episodes','128','--boundary-mode','eval','--boundary-checkpoint','reliability_v2_boundary_pilot1_random/boundary_last.pt','--filter','ema','--initial',roles['initial'],'--late-initial',roles['late_initial'],'--post',roles['post'],'--align',roles['alignment'],'--deep-recovery',roles['deep_recovery'],'--near-align',roles['near_alignment'],'--release-guard','--handoff-contact','--conditioning','on','--history','--case-profile','lateral_focus','--offset-mm','5'];(D/f'command_{seed}.json').write_text(json.dumps(cmd,indent=2));F=L/(job+'_random')
  if not (F/'domain_collect_result.json').exists():subprocess.run(cmd,check=True)
  else:
   assert (D/f'c{seed}_p0_local_verified.json').exists() and (D/f'c{seed}_p1_local_verified.json').exists(),'Resume only fully sampled,dual-verified cohort'
   assert not (D/'value_fit_result.json').exists(),'Do not rerun or refit a completed value fit'
  r=load(F/'domain_collect_result.json');assert r['final_retry_release'];assert r['seed']==seed and r['index']==index and r['history_enabled'] and not r['training_forbidden'] and r['actor_updates']==r['value_fit_attempts']==0;assert r['domain_joint_hash'] not in [v['result']['domain_joint_hash'] for v in collected]
  raw_size=size(F)+size(L/job)
  for folder in [F,L/job]:
   for ref in folder.rglob('*.archive_ref.json'):raw_size+=load(ref)['bytes']-ref.stat().st_size
  parts=[];refs=[];storage_stop=raw_size>14680064
  for part in range(2):
   A=L/(job+f'_p{part}.tar.gz');verify()
   if not A.exists():
    assert _namespace_free_bytes(L)>=5045649408+min(raw_size,8388608)*1.02
    with tarfile.open(A,'w:gz',compresslevel=9) as z:
     for folder in [L/job,F]:
      for f in folder.rglob('*'):
       if not f.is_file():continue
       if (part==0)!=(f.name=='complete_history.npz'):continue
       if f.suffix in ['.json','.jsonl','.py','.npz','.pt'] or f.name=='console.log':z.add(f,arcname=str(f.relative_to(L)))
   else:
    prior=load(D/f'c{seed}_p{part}_local_verified.json');assert prior['archive_sha256']==sha(A)
    for folder in [L/job,F]:
     for ref in folder.rglob('*.archive_ref.json'):
      v=load(ref)
      if v.get('archive')==str(A):refs.append(v)
   meta=dict(path=str(A),sha256=sha(A),bytes=A.stat().st_size,part=part);parts.append(meta);storage_stop=storage_stop or A.stat().st_size>8388608;verify();status('awaiting_local_cohort_verification',seed=seed,index=index,part=part,archive=meta,result_folder=F.name);deadline=time.monotonic()+1800;P=D/f'c{seed}_p{part}_local_verified.json'
   while True:
    assert time.monotonic()<deadline,'Leave archived data intact;local verification not yet received'
    try:proof=load(P);break
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(5)
   assert proof['status']=='passed' and proof['archive_sha256']==meta['sha256'] and proof['complete_history42_lossless_verified'] and proof['episodes']==128 and (part==0 or (proof['discount_horizon_clock_verified'] and proof['explicit_predecision_joint_target6_verified'] and proof['shared_source_all65_logical_bytes_verified'] and proof['explicit_preaction_queue12_verified'] and proof['complete_trueforce3_lossless_verified'] and proof['explicit_preaction_latency1_verified']))
   for folder in [L/job,F]:
    for f in list(folder.rglob('*')):
     if f.is_file() and (f.suffix in ['.npz','.py'] or f.name=='all_failure_terminal_actor.jsonl') and str(f.relative_to(L)) in proof['all_archive_member_hashes']:
      member=str(f.relative_to(L));assert sha(f)==proof['all_archive_member_hashes'][member];v=dict(archive=str(A),archive_sha256=meta['sha256'],member=member,sha256=sha(f),bytes=f.stat().st_size);f.with_suffix(f.suffix+'.archive_ref.json').write_text(json.dumps(v,indent=2));f.unlink();refs.append(v)
   verify()
  item=dict(seed=seed,folder=F.name,job_folder=job,result=r,archives=parts,compacted_verified_duplicates=refs);collected.append(item);volumes.extend(parts);(D/'collection_index.json').write_text(json.dumps(collected,indent=2));closed=sum(v['bytes'] for v in parts)+size(F)+size(L/job);verify()
  if storage_stop or closed>14680064:stop=f'stopped_sequence_storage_forecast_seed{seed}_before_further_gradients';break
  if not r['force_gate_passed']:stop=f'force_gate_failed_seed{seed}_before_fit';break
  if not r['coverage_passed']:stop=f'native128_step_coverage_failed_seed{seed}_before_fit';break
  if index%2==1:
   domain=pair_coverage([v['result'] for v in collected[-2:]]);prior=load(D/'paired_domain_coverage.json') if (D/'paired_domain_coverage.json').exists() else [];prior.append(domain);(D/'paired_domain_coverage.json').write_text(json.dumps(prior,indent=2))
   if not domain['passed']:stop=f'paired256_independent_episode_coverage_failed_domain{index//2}_before_fit';break
 assert collected;status('independent_CPU_complete_history_audit',stop_reason=stop);container('audit_expanded24_b.py');audit=load(D/'saved_checkpoint_audit.json');assert audit['status']=='passed';fit=load(D/'value_fit_result.json') if (D/'value_fit_result.json').exists() else None;assert not (D/'value_fit_result.json').exists();passed=not stop and len(collected)==8;result=dict(status='training_bank_B_ready' if passed else 'history_stopped',stop_reason=stop,episodes=128*len(collected),successes=sum(v['result']['successes'] for v in collected),force_aborts=sum(v['result']['force_aborts'] for v in collected),actor_updates=0,value_fits=fit['value_fits'] if fit else 0,value_fit_attempts=fit['value_fit_attempts'] if fit else 0,fit=fit,collection_index=collected,cohort_volumes=volumes,CPU_audit=audit,archive_layout='complete_multi_volume;all cohort volumes plus final review plus indexed shared source required',candidate_accepted=False,accepted_parent_preserved=True);(D/'result.json').write_text(json.dumps(result,indent=2));A=L/(G+'_review.tar.gz');assert not A.exists();verify()
 with tarfile.open(A,'w:gz',compresslevel=9) as z:
  for folder in [D,S]:
   for f in folder.rglob('*'):
    if f.is_file() and (f.suffix in ['.json','.jsonl','.npz','.pt','.py'] or f.name=='console.log'):z.add(f,arcname=str(f.relative_to(L)))
  for n in p['helper_hashes']:z.add(L/n,arcname=n)
 meta=dict(path=str(A),sha256=sha(A),bytes=A.stat().st_size);assert verify()+A.stat().st_size<=167772160;(D/'archive.json').write_text(json.dumps(meta,indent=2));status(result['status'],episodes=result['episodes'],actor_updates=0,value_fits=result['value_fits'],stop_reason=stop,archive=meta)
 old=load(L/'piper_status_latest.json');old.update(server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=_namespace_free_bytes(L),active_Piper_containers=[]);old['last_experiment']=dict(group=G,status=result['status'],reason=stop,training_updates=0,value_fits=result['value_fits'],value_fit_attempts=result['value_fit_attempts'],complete_training_episodes=result['episodes'],archive=meta,cohort_volumes=volumes,candidate_accepted=False,development_seeds_consumed=[],formal_seeds_consumed=[]);(L/'piper_status_latest.json').write_text(json.dumps(old,indent=2))
except BaseException as e:status('failed',error=str(e),accepted_parent_preserved=True);traceback.print_exc();raise
