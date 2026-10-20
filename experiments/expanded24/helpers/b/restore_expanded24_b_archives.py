import shutil as _storage_shutil
def _namespace_free_bytes(target):
    assert _storage_shutil.disk_usage('/data/shijinsheng/piper-insertion').free>=5045649408,'Original data filesystem reserve must remain intact'
    return _storage_shutil.disk_usage(target).free
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');a=p.parse_args();L=Path('/home/shijinsheng/piper-training-workspace-20261019/logs');G='reliability_v2_expanded24_b_20261020';E='reliability_v2_expanded24_a_native_20261020';D=L/G;load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda b:hashlib.sha256(b).hexdigest();records=load(L/E/'completed_equivalence_compaction.json')['records']
source_plan=load(D/'plan.json');package=L/'expanded24_shared_source_20261020.tar.gz';assert sha(package.read_bytes())==source_plan['source_archive_sha256']
with tarfile.open(package) as z:
 for name,h in source_plan['executable_hashes'].items():assert sha(z.extractfile('executable/'+name).read())==h
if (D/'collection_index.json').exists():records += [r for v in load(D/'collection_index.json') for r in v['compacted_verified_duplicates']]
for archive in sorted({r['archive'] for r in records}):
 rows=[r for r in records if r['archive']==archive];A=Path(archive);assert A.parent==L and sha(A.read_bytes())==rows[0]['archive_sha256']
 with tarfile.open(A) as z:
  for r in rows:
   f=Path(r.get('path',str(L/r['member'])));assert f.resolve().is_relative_to(L.resolve()) and (f.suffix in ['.py','.npz'] or f.name=='all_failure_terminal_actor.jsonl') and '_source/' not in str(f);b=z.extractfile(r['member']).read();assert sha(b)==r['sha256'] and len(b)==r['bytes']
   if f.exists():assert sha(f.read_bytes())==r['sha256']
   elif a.restore:
    assert _namespace_free_bytes(L)>=5045649408+len(b);f.write_bytes(b);assert sha(f.read_bytes())==r['sha256']
print(json.dumps(dict(status='passed',archived_duplicate_entries=len(records),restored=a.restore,all_unique_bytes_preserved=True)))
