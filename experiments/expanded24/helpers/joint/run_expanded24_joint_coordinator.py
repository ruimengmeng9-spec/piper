"""One full3072 cold critic fit. No native collection, blind tests or actor updates."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,tarfile,time,traceback
from expanded24_resource_meter import check_budget
from expanded24_archive_index_io import read_index
L=Path('/home/shijinsheng/piper-training-workspace-20261019/logs');R=L.parent
G='reliability_v2_expanded24_joint_20261020';D=L/G
I='sha256:0c1687cdae03921a28307bb3c0471174b69d37741c3cbb9a68cfd4e1640fbba8'
load=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
p=load(D/'plan.json');S=L/(p['source_group']+'_source')
def status(phase,**kw):
    v=dict(server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,**kw)
    (D/'status.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
def size(folder):return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
def verify():
    assert all(sha(S/'executable'/n)==h for n,h in p['executable_hashes'].items())
    assert sha(L/'expanded24_shared_source_20261020.tar.gz')==p['source_archive_sha256']
    assert sha(L/'accepted_piper_bundle_20261016_signal_ema.tar.gz')==p['parent_accepted_bundle_sha256']
    parents=load(L/'reliability_v2_brake_recovery_20261017/plan.json')['checkpoint_hashes']
    assert all(sha(L/n)==h for n,h in parents.items())
    assert sha(R/'piper_env.py')=='933d4a4ccdbef7fe3f999402e31286cb02b92e84854626befceae773673bee8e'
    assert sha(R/'cad/challenge_cup/challenge_cup_robot.usdc')=='5f70fc422a92027a1bb6197a02a4c0befc907dc0619d0a90df19d4dd82d8b044'
    for n,h in p['closed_bank_file_hashes'].items():assert sha(L/n)==h
    for n,h in p['helper_hashes'].items():assert sha(L/n)==h
    allocation=check_budget(p)
    metadata=size(D)+size(S)+(L/'expanded24_shared_source_20261020.tar.gz').stat().st_size
    assert metadata<=4194304,'Preserve metadata/source/fit cap'
    retained=size(D)+sum((L/n).stat().st_size for n in p['helper_hashes'])+(L/'expanded24_shared_source_20261020.tar.gz').stat().st_size
    review=L/(G+'_review.tar.gz')
    if review.exists():retained+=review.stat().st_size
    assert retained<=167772160
    return dict(resource=allocation,global_metadata_source_fit_bytes=metadata,retained_bytes=retained)
def container(script,gpu=False):
    if gpu:
        stable=0;deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            memory,util=map(int,subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            idle=not subprocess.check_output(['docker','ps','--filter','name=shijinsheng-piper','--format','{{.Names}}'],text=True).strip()
            stable=stable+1 if memory>=14000 and util<=5 and idle else 0
            if stable>=3:break
            time.sleep(2)
        assert stable>=3,'Unchanged GPU admission'
    name='shijinsheng-piper-reliability-v2-expanded24-c-20261020-joint-'+script.removesuffix('.py').replace('_','-')
    cmd=['docker','run','--rm','--read-only','--name',name]+(['--gpus','device=2'] if gpu else [])+['--tmpfs','/tmp:rw,size=128m','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','--cpus=2','--memory=3g','--network','none','--entrypoint','/isaac-sim/python.sh','-v',str(R)+':/workspace/piper-insertion','-v',str(R)+':'+str(R)+':ro','-v','/data/shijinsheng/piper-insertion:/data/shijinsheng/piper-insertion:ro',I,'/workspace/piper-insertion/logs/'+script]
    subprocess.run(cmd,check=True)
try:
    assert p['stage']=='sole_joint3072_value_fit_after_all_three_closed_banks' and p['maximum_real_data_fit_attempts']==1
    assert p['actor_updates_permitted']==p['blind_episodes_permitted_here']==0
    assert not (D/'fit_attempt_started.json').exists() and not (D/'value_fit_result.json').exists()
    idx=load(D/'collection_index.json');assert len(idx)==24 and [r['seed'] for r in idx]==p['training_seeds']
    assert [r['result']['index'] for r in idx]==list(range(24)) and all(not r['result']['blind'] and r['result']['force_gate_passed'] and r['result']['coverage_passed'] for r in idx)
    assert len(read_index(D/'volume_index.json')['volumes'])==1759
    for bank in ['a','b','c']:
        F=L/('reliability_v2_expanded24_'+bank+'_20261020')
        assert load(F/'result.json')['status']=='training_bank_'+bank.upper()+'_ready'
        assert load(F/'final_state.json')['status']=='closed_verified'
        assert load(F/'post_closeout_budget.json')['status']=='passed'
    verify();status('full3072_CPU_CUDA_no_gradient_resource_admission',actor_updates=0,real_fit_attempts=0)
    container('preflight_expanded24_joint_full_data.py',True)
    pre=load(D/'full_data_preflight.json')
    assert pre['status']=='passed' and pre['no_fit_or_optimizer_constructed'] and pre['peak_RAM_bytes']<3221225472
    assert pre['packed_complete_episodes']<=1536 and pre['admission']['all24_native_and12_pair_phase_step_and_independent_episode_gates_before_fit_marker']
    assert set(pre['reports'])=={'cpu','cuda:0'} and all(r['complete_episodes']==3072 and r['no_gradients'] for r in pre['reports'].values())
    verify()
    with (D/'fit_attempt_started.json').open('x') as marker:json.dump(dict(attempt=1,initial_boundary_sha256=p['initial_boundary_sha256'],full_data_preflight_sha256=sha(D/'full_data_preflight.json'),source_sha256=p['source_archive_sha256'],cold_original_parent=True,epochs=160,no_second_attempt=True),marker)
    status('sole_complete3072_value_fit',actor_updates=0,real_fit_attempts=1)
    container('fit_expanded24_joint.py',True);fit=load(D/'value_fit_result.json')
    assert fit['actor_updates']==0 and fit['value_fit_attempts']==1 and fit['fit']['epochs']==160
    assert fit['fit']['weights']['loss_group_count']==72 and fit['fit']['after_assessment']['assessed_groups']==37
    verify();status('independent_CPU_complete_history_model_optimizer_RNG_audit',actor_updates=0,real_fit_attempts=1)
    container('audit_expanded24_joint.py');audit=load(D/'saved_checkpoint_audit.json');assert audit['status']=='passed' and audit['actor_updates']==0 and audit['episodes']==3072
    passed=fit['fit']['after_assessment']['all_value_gates_passed']
    result=dict(status='value_passed_requires_separate_blind_tests' if passed else 'history_stopped',stop_reason=None if passed else '37_training_value_gates_failed_no_blind_no_actor_update',episodes=3072,successes=sum(v['result']['successes'] for v in idx),force_aborts=sum(v['result']['force_aborts'] for v in idx),actor_updates=0,value_fit_attempts=1,value_fits=fit['value_fits'],fit=fit,collection_index=[{k:v for k,v in item.items() if k!='compacted_verified_duplicates'} for item in idx],cohort_volumes=[v for item in idx for v in item['archives']],CPU_audit=audit,archive_layout='all1759 inherited complete archives plus current fit review; source package is the unchanged shared A package',candidate_accepted=False,accepted_parent_preserved=True,blind_episodes=0)
    (D/'result.json').write_text(json.dumps(result,separators=(',',':')));verify()
    archive=L/(G+'_review.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=9) as z:
        for f in D.rglob('*'):
            if f.is_file() and f.suffix in ['.json','.jsonl','.py','.npz','.pt']:z.add(f,arcname=str(f.relative_to(L)))
        # Portable logical source prefix, without another physical65-file server copy.
        for f in (S/'executable').iterdir():
            if f.is_file() and f.suffix in ['.py','.json']:z.add(f,arcname=G+'_source/executable/'+f.name)
        for n in p['helper_hashes']:z.add(L/n,arcname=n)
    meta=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size);verify()
    (D/'archive.json').write_text(json.dumps(meta,indent=2));status(result['status'],actor_updates=0,real_fit_attempts=1,blind_episodes=0,archive=meta)
    latest=load(L/'piper_status_latest.json');latest.update(server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=shutil.disk_usage(L).free,active_Piper_containers=[])
    latest['last_experiment']=dict(group=G,status=result['status'],reason=result['stop_reason'],training_updates=0,value_fits=result['value_fits'],value_fit_attempts=1,complete_training_episodes=3072,blind_episodes=0,actor_updates=0,archive=meta,candidate_accepted=False,development_seeds_consumed=[],formal_seeds_consumed=[])
    (L/'piper_status_latest.json').write_text(json.dumps(latest,indent=2))
except BaseException as e:
    status('failed',error=repr(e),no_refit_or_resampling=True,accepted_parent_preserved=True);traceback.print_exc();raise
