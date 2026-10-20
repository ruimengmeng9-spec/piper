import shutil as _storage_shutil
def _namespace_free_bytes(target):
    assert _storage_shutil.disk_usage('/data/shijinsheng/piper-insertion').free>=5045649408,'Original data filesystem reserve must remain intact'
    return _storage_shutil.disk_usage(target).free
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,tarfile
L=Path('/home/shijinsheng/piper-training-workspace-20261019/logs');E='reliability_v2_expanded24_a_native_20261020';G='reliability_v2_expanded24_a_20261020';D=L/G;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((L/E/'local_verification.json').read_text());assert proof['status']=='passed' and proof['training_authorized'];A=Path(proof['server_archive_path']);assert sha(A)==proof['archive_sha256'];refs=[]
with tarfile.open(A) as z:
 for folder in [L/(E+s) for s in ['_release','_release_random']]:
  for f in list(folder.rglob('*')):
   if not f.is_file() or f.suffix not in ['.npz','.py']:continue
   member=str(f.relative_to(L));assert member in proof['all_archive_member_hashes'];h=proof['all_archive_member_hashes'][member];assert sha(f)==h and hashlib.sha256(z.extractfile(member).read()).hexdigest()==h
   r=dict(path=str(f),archive=str(A),archive_sha256=proof['archive_sha256'],member=member,sha256=h,bytes=f.stat().st_size);f.with_suffix(f.suffix+'.archive_ref.json').write_text(json.dumps(r,indent=2));f.unlink();refs.append(r)
CP=L/E/'completed_equivalence_compaction.json'
if CP.exists():refs=json.loads(CP.read_text())['records']+refs
CP.write_text(json.dumps(dict(status='passed',records=refs,all_unique_bytes_preserved=True),indent=2))
p=json.loads((D/'plan.json').read_text());p['initial_free_bytes']=json.loads((L/E/'plan.json').read_text())['initial_free_bytes'];p['campaign_initial_free_bytes']=p['initial_free_bytes'];p['native_equivalence_archive_sha256']=proof['archive_sha256'];p['storage_admission_after_dual_verified_compaction']=dict(server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=_namespace_free_bytes(L),required=5246976000)
assert _namespace_free_bytes(L)>=5246976000;(D/'plan.json').write_text(json.dumps(p,indent=2));assert not (D/'status.json').exists()
f=(D/'coordinator.log').open('ab');job=subprocess.Popen(['python3',str(L/'run_expanded24_a_coordinator.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps(dict(pid=job.pid,status='training_collections_launched',free_bytes=_namespace_free_bytes(L),compacted_equivalence_duplicates=len(refs))))
