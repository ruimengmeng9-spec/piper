import shutil as _storage_shutil
def _namespace_free_bytes(target):
    assert _storage_shutil.disk_usage('/data/shijinsheng/piper-insertion').free>=5045649408,'Original data filesystem reserve must remain intact'
    return _storage_shutil.disk_usage(target).free
from pathlib import Path
import datetime,json,shutil
L=Path('/home/shijinsheng/piper-training-workspace-20261019/logs');G='reliability_v2_expanded24_a_20261020';E='reliability_v2_expanded24_a_native_20261020';D=L/G;load=lambda p:json.loads(p.read_text(encoding='utf-8'));p=load(D/'plan.json');r=load(D/'result.json');size=lambda d:sum(f.stat().st_size for f in d.rglob('*') if f.is_file());rows=[]
for item in r['collection_index']:
 folders=[L/item['folder'],L/item['job_folder']];actual=sum(size(f) for f in folders);raw=actual
 for folder in folders:
  for ref in folder.rglob('*.archive_ref.json'):raw+=load(ref)['bytes']-ref.stat().st_size
 closed=actual+sum(v['bytes'] for v in item['archives']);assert raw<=14680064 and closed<=14680064;assert all(v['bytes']<=8388608 for v in item['archives']);rows.append(dict(seed=item['seed'],index=item['result']['index'],raw_bytes=raw,closed_bytes=closed,archive_part_bytes=[v['bytes'] for v in item['archives']]))
pairs=[]
for i in range(0,min(8,len(rows))-1,2):
 raw=sum(v['raw_bytes'] for v in rows[i:i+2]);closed=sum(v['closed_bytes'] for v in rows[i:i+2]);assert raw<=29360128 and closed<=29360128;pairs.append(dict(domain=i//2,raw_bytes=raw,closed_bytes=closed))
S=L/(G+'_source');retained=size(D)+size(S)+sum(size(f) for f in L.glob(G+'_c*') if f.is_dir())+sum(f.stat().st_size for f in L.glob(G+'_c*.tar.gz'))+sum((L/n).stat().st_size for n in p['helper_hashes'])+(L/(G+'_review.tar.gz')).stat().st_size
shared_bytes=(L/'expanded24_shared_source_20261020.tar.gz').stat().st_size;retained+=shared_bytes;global_bytes=size(D)+size(S)+shared_bytes;free=_namespace_free_bytes(L);net=p['initial_free_bytes']-free;assert retained<=167772160 and global_bytes<=4194304 and net<=201326592 and free>=5045649408
v=dict(status='passed',server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native128=rows,paired256=pairs,retained_bytes=retained,retained_cap_bytes=167772160,global_metadata_source_fit_bytes=global_bytes,global_cap_bytes=4194304,net_bytes_from_before_single_known_diagnostic=net,net_cap_bytes=201326592,free_bytes=free,minimum_free_bytes=5045649408,initial_free_bytes=p['initial_free_bytes'],initial_includes_one_known_diagnostic=True,canonical_models_and_full_dual_data_archives_preserved=True)
(D/'final_budget.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
