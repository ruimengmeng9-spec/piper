import shutil as _storage_shutil
def _namespace_free_bytes(target):
    assert _storage_shutil.disk_usage('/data/shijinsheng/piper-insertion').free>=5045649408,'Original data filesystem reserve must remain intact'
    return _storage_shutil.disk_usage(target).free
import datetime,hashlib,json,shutil,subprocess,tarfile,traceback
from pathlib import Path
L=Path('/home/shijinsheng/piper-training-workspace-20261019/logs');R=L.parent;E='reliability_v2_expanded24_a_native_20261020';D=L/E;S=L/(E+'_source');I='sha256:0c1687cdae03921a28307bb3c0471174b69d37741c3cbb9a68cfd4e1640fbba8';load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=load(D/'plan.json');roles=load(L/'reliability_v2_guarded_recurrent_20261017/plan.json')['roles'];parents=load(L/'reliability_v2_brake_recovery_20261017/plan.json')['checkpoint_hashes']
def status(phase,**kw):
 v=dict(server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,**kw);(D/'status.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
def verify():
 assert sha(L/'accepted_piper_bundle_20261016_signal_ema.tar.gz')==p['parent_accepted_bundle_sha256'];assert all(sha(L/n)==h for n,h in parents.items());assert _namespace_free_bytes(L)>=5045649408;assert p['initial_free_bytes']-_namespace_free_bytes(L)<=201326592
def docker(script,gpu=False):
 c=['docker','run','--rm','--read-only']+(['--gpus','device=2'] if gpu else [])+['--tmpfs','/tmp:rw,size=128m','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','--cpus=2','--memory=3g','--network','none','--entrypoint','/isaac-sim/python.sh','-v',str(R)+':/workspace/piper-insertion','-v',str(R)+':'+str(R)+':ro','-v','/data/shijinsheng/piper-insertion:/data/shijinsheng/piper-insertion:ro',I,'/workspace/piper-insertion/logs/'+script];subprocess.run(c,check=True)
try:
 import time
 stable=0;deadline=time.monotonic()+60
 while time.monotonic()<deadline:
  active=subprocess.check_output(['docker','ps','--filter','name=shijinsheng-piper','--format','{{.Names}}'],text=True).strip();mem,util=map(int,subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','));stable=stable+1 if mem>=14000 and util<=5 and not active else 0
  if stable>=3:break
  time.sleep(2)
 assert stable>=3,'GPU admission gate unchanged'
 assert not (D/'result.json').exists() and not (D/'attempt_started.json').exists();(D/'attempt_started.json').write_text(json.dumps(dict(status='one_shot_diagnostic_started')));assert p['training_seeds']==[2811,2813,2815,2817,2819,2821,2823,2825,2827,2829,2831,2833,2835,2837,2839,2841,2843,2845,2847,2849,2851,2853,2855,2857] and p['blind_gate_seeds']==[2873,2889]
 for name,h in p['equivalence_helper_hashes'].items():assert sha(L/name)==h
 assert _namespace_free_bytes(L)>=p['campaign_budget']['admit_all_banks_before_first_sample_required'];p.setdefault('initial_free_bytes',_namespace_free_bytes(L));assert p['initial_free_bytes']>=5045649408+201326592;(D/'plan.json').write_text(json.dumps(p,indent=2));verify();status('verifying_cached_CPU_CUDA_candidate_and_collector_registration')
 cert=L/'reliability_v2_expanded24_integration_preflight_20261020/verification.json';c=load(cert);assert sha(cert)==p['candidate_source_integration_report_sha256'] and c['status']=='expanded24_candidate_CPU_CUDA_equivalence_passed'
 assert set(c['synthetic_reports'])=={'cpu','cuda:0'} and all(x['all37_guard_groups_and_failed_native23_late_verified'] and x['TRAIN72_numpy_weight_max_error']==0. and x['synthetic4epoch_parameter_max_error']==0. and x['synthetic4epoch_optimizer_max_error']==0. for x in c['synthetic_reports'].values())
 assert all(p['executable_hashes'][n]==h for n,h in c['candidate_source_hashes'].items() if n!='critic_conditioned_collect.py')
 assert p['collector_registration_AST_whitelist_verified'];import ast,copy
 original=ast.parse((L/'reliability_v2_force_smooth_b_20261020_source/executable/critic_conditioned_collect.py').read_text());registered=ast.parse((S/'executable/critic_conditioned_collect.py').read_text());oldlist=next(ast.literal_eval(x.value) for x in original.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SEEDS' for t in x.targets))
 for node in ast.walk(registered):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SEEDS' for t in node.targets):assert ast.literal_eval(node.value)==p['training_seeds']+p['blind_seeds'];node.value=ast.parse(repr(oldlist),mode='eval').body
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='blind' for t in node.targets):
   comp=next(x for x in ast.walk(node.value) if isinstance(x,ast.Compare) and isinstance(x.left,ast.Name) and x.left.id=='index');assert comp.comparators[0].value==24;comp.comparators[0].value=16
 assert ast.dump(registered)==ast.dump(original)
 pre=dict(status='passed',candidate_source_report_sha256=sha(cert),all64_tested_source_files_identical=True,collector_only_seed_list_and_blind_boundary_changes=True,new_source_sha256=p['source_archive_sha256'],no_new_optimizer_or_gradients=True);(D/'preflight.json').write_text(json.dumps(pre,indent=2))
 for condition in ['release']:
  status('native_zero_update_equivalence',condition=condition,seed=1125,actor_updates=0);cmd=['python3',str(L/'run_expanded24_a_job.py'),'--group',E+'_'+condition,'--source',E+'_source','--seed','1124','--profiles','random','--episodes','128','--boundary-mode','eval','--boundary-checkpoint','reliability_v2_boundary_pilot1_random/boundary_last.pt','--filter','ema','--initial',roles['initial'],'--late-initial',roles['late_initial'],'--post',roles['post'],'--align',roles['alignment'],'--deep-recovery',roles['deep_recovery'],'--near-align',roles['near_alignment'],'--handoff-contact','--conditioning','on','--equivalence','--history','--offset-mm','5'];cmd+=['--release-guard'] if condition=='release' else [];(D/(condition+'_command.json')).write_text(json.dumps(cmd,indent=2));subprocess.run(cmd,check=True);verify()
 status('exact_native_and_CPU_value_audit');docker('compare_expanded24_a_equivalence.py');r=load(D/'equivalence_result.json');assert r['status']=='passed';docker('audit_expanded24_a_diagnostic.py');assert load(D/'saved_checkpoint_audit.json')['status']=='passed';(D/'result.json').write_text(json.dumps(dict(status='native_equivalence_passed_pending_local_verification',equivalence=r,actor_updates=0,value_fits=0,candidate_accepted=False,accepted_parent_preserved=True),indent=2));folders=[D,S]+[L/(E+'_'+label) for label in ['release']]+[L/(E+'_'+label+'_random') for label in ['release']];size=sum(f.stat().st_size for folder in folders for f in folder.rglob('*') if f.is_file());assert p['initial_free_bytes']-_namespace_free_bytes(L)+size*1.02<=100663296;A=L/(E+'_review.tar.gz');assert not A.exists()
 with tarfile.open(A,'w:gz',compresslevel=9) as z:
  for folder in folders:
   for f in folder.rglob('*'):
    if f.is_file() and (f.suffix in ['.py','.json','.jsonl','.npz','.pt'] or f.name=='console.log'):z.add(f,arcname=str(f.relative_to(L)))
  for name in p['equivalence_helper_hashes']:z.add(L/name,arcname=name)
 meta=dict(path=str(A),sha256=sha(A),bytes=A.stat().st_size);(D/'archive.json').write_text(json.dumps(meta,indent=2));verify();status('native_equivalence_passed_pending_local_verification',archive=meta,actor_updates=0,value_fits=0,episodes=128)
except BaseException as e:status('failed',error=str(e),accepted_parent_preserved=True);traceback.print_exc();raise
