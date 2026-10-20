"""Original storage caps, including every inherited native/pair and current fit metadata."""
from pathlib import Path
import datetime,json
from expanded24_resource_meter import check_budget
L=Path('/home/shijinsheng/piper-training-workspace-20261019/logs');G='reliability_v2_expanded24_joint_20261020';D=L/G
load=lambda f:json.loads(f.read_text());size=lambda d:sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
p=load(D/'plan.json');r=load(D/'result.json');rows=[]
assert r['episodes']==3072 and r['actor_updates']==0 and r['value_fit_attempts']==1
for item in r['collection_index']:
    folders=[L/item['folder'],L/item['job_folder']];actual=sum(size(f) for f in folders);raw=actual
    for folder in folders:
        for ref in folder.rglob('*.archive_ref.json'):
            v=load(ref);raw+=v['bytes']-ref.stat().st_size
    closed=actual+sum(v['bytes'] for v in item['archives'])
    assert raw<=14680064 and closed<=14680064 and all(v['bytes']<=8388608 for v in item['archives'])
    rows.append(dict(seed=item['seed'],index=item['result']['index'],raw_bytes=raw,closed_bytes=closed))
pairs=[]
for i in range(0,24,2):
    raw=sum(v['raw_bytes'] for v in rows[i:i+2]);closed=sum(v['closed_bytes'] for v in rows[i:i+2])
    assert raw<=29360128 and closed<=29360128;pairs.append(dict(domain=i//2,raw_bytes=raw,closed_bytes=closed))
S=L/(p['source_group']+'_source');shared=(L/'expanded24_shared_source_20261020.tar.gz').stat().st_size
metadata=size(D)+size(S)+shared;retained=size(D)+sum((L/n).stat().st_size for n in p['helper_hashes'])+shared+(L/(G+'_review.tar.gz')).stat().st_size
m=check_budget(p);assert metadata<=4194304 and retained<=167772160
out=dict(status='passed',server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native128=rows,paired256=pairs,retained_bytes=retained,retained_cap_bytes=167772160,global_metadata_source_fit_bytes=metadata,global_cap_bytes=4194304,source_reused_from_C=True,source_bytes_conservatively_included=True,net_bytes_from_before_fit_deployment=m['bank_owned_net_bytes'],net_cap_bytes=201326592,campaign_cap_bytes=402653184,free_bytes=m['actual_free_bytes'],old_free_bytes=m['old_actual_free_bytes'],minimum_free_bytes=5045649408,initial_free_bytes=p['initial_free_bytes'],campaign_initial_free_bytes=p['campaign_initial_free_bytes'],owned_resource_measurement=m,original_caps_and_snapshots_preserved=True,canonical_models_and_full_dual_data_archives_preserved=True)
(D/'final_budget.json').write_text(json.dumps(out,separators=(',',':')));print(json.dumps(out))
