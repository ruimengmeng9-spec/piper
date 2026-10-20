"""Resume audits/archive of completed known128. Never starts a native process or optimizer."""
from pathlib import Path
import datetime,hashlib,json,subprocess,tarfile,shutil,sys
L=Path('/home/shijinsheng/piper-training-workspace-20261019/logs');sys.path.insert(0,str(L))
from expanded24_resource_meter import check_budget
R=L.parent;E='reliability_v2_expanded24_a_native_20261020';D=L/E;S=L/(E+'_source');F=L/(E+'_release_random');B=L/'reliability_v2_expanded24_budget_revision_20261020';load=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();p=load(D/'plan.json');review=load(B/'measurement_review.json');q=dict(p,resource_accounting_revision=review['config'],bank_initial_owned_allocated_bytes=0)
def status(phase,**kw):
 v=dict(server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,**kw);(D/'status.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
def verify():
 assert all(sha(L/n)==h for n,h in p['equivalence_helper_hashes'].items())
 assert all(sha(S/'executable'/n)==h for n,h in p['executable_hashes'].items())
 assert sha(L/'accepted_piper_bundle_20261016_signal_ema.tar.gz')==p['parent_accepted_bundle_sha256']
 parents=load(L/'reliability_v2_brake_recovery_20261017/plan.json')['checkpoint_hashes'];assert all(sha(L/n)==h for n,h in parents.items())
 assert p['initial_free_bytes']==load(B/'before_plan.json')['initial_free_bytes']
 return check_budget(q)
def cpu(script):
 cmd=['docker','run','--rm','--read-only','--tmpfs','/tmp:rw,size=128m','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','--cpus=2','--memory=3g','--network','none','--entrypoint','/isaac-sim/python.sh','-v',str(R)+':/workspace/piper-insertion','-v',str(R)+':'+str(R)+':ro','-v','/data/shijinsheng/piper-insertion:/data/shijinsheng/piper-insertion:ro','sha256:0c1687cdae03921a28307bb3c0471174b69d37741c3cbb9a68cfd4e1640fbba8','/workspace/piper-insertion/logs/'+script];subprocess.run(cmd,check=True)
try:
 assert load(D/'status.json')['phase']=='failed' and load(F/'status.json')['status']=='completed' and load(F/'status.json')['process_exit_code']==0
 result=load(F/'domain_collect_result.json');assert result['episodes']==128 and result['seed']==1125 and result['training_forbidden'] and result['actor_updates']==result['value_fit_attempts']==0
 assert not (D/'result.json').exists() and not (L/(E+'_review.tar.gz')).exists()
 with (B/'audit_only_resume_started.json').open('x') as f:json.dump(dict(no_native_launch=True,no_optimizer=True),f)
 allocation=verify();revision=dict(status='project_owned_resource_metric_corrected_before_fresh_samples',reason='Shared filesystem free-space delta includes allocations outside this project. Project-owned workspace/cache/log blocks plus conservative own-container layer allocation now enforce the same201326592/402653184 caps; actual free floors and original free baselines are unchanged.',original_failure_sha256=sha(B/'before_status.json'),original_initial_free_bytes=p['initial_free_bytes'],configuration=review['config'],measurement=allocation,additional_helper_hashes={n:sha(L/n) for n in ['expanded24_resource_meter.py','run_expanded24_known_budget_resume_20261020.py']},known_physical_episodes_repeated=0,actor_updates=0,real_data_gradient_steps=0);(D/'resource_accounting_revision.json').write_text(json.dumps(revision,indent=2))
 status('audit_existing_known128_after_resource_accounting_review');cpu('compare_expanded24_a_equivalence.py');r=load(D/'equivalence_result.json');assert r['status']=='passed';cpu('audit_expanded24_a_diagnostic.py');assert load(D/'saved_checkpoint_audit.json')['status']=='passed'
 (D/'result.json').write_text(json.dumps(dict(status='native_equivalence_passed_pending_local_verification',equivalence=r,actor_updates=0,value_fits=0,candidate_accepted=False,accepted_parent_preserved=True,physical_native_reruns=0),indent=2))
 folders=[D,S,L/(E+'_release'),F];size=sum(f.stat().st_size for folder in folders for f in folder.rglob('*') if f.is_file());assert verify()['project_owned_allocated_bytes']+size*1.02<=100663296
 A=L/(E+'_review.tar.gz');assert not A.exists()
 with tarfile.open(A,'w:gz',compresslevel=9) as z:
  for folder in folders:
   for f in folder.rglob('*'):
    if f.is_file() and (f.suffix in ['.py','.json','.jsonl','.npz','.pt'] or f.name=='console.log'):z.add(f,arcname=str(f.relative_to(L)))
  for name in dict.fromkeys(list(p['equivalence_helper_hashes'])+list(revision['additional_helper_hashes'])):z.add(L/name,arcname=name)
 meta=dict(path=str(A),sha256=sha(A),bytes=A.stat().st_size);(D/'archive.json').write_text(json.dumps(meta,indent=2));verify();status('native_equivalence_passed_pending_local_verification',archive=meta,actor_updates=0,value_fits=0,episodes=128,physical_reruns=0)
except BaseException as exc:status('failed',error=repr(exc),accepted_parent_preserved=True,physical_reruns=0);raise
