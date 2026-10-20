"""Project-owned allocated bytes, including exact own Docker writable-layer allocation."""
from pathlib import Path
import json,os,re,shutil,subprocess
IMAGE='sha256:0c1687cdae03921a28307bb3c0471174b69d37741c3cbb9a68cfd4e1640fbba8'
ROOT=Path('/home/shijinsheng/piper-training-workspace-20261019')
def allocated(paths):
    seen=set();total=0;count=0
    def add(f):
        nonlocal total,count
        st=f.lstat();key=(st.st_dev,st.st_ino)
        if key not in seen:seen.add(key);total+=st.st_blocks*512;count+=1
    for path in paths:
        if not path.exists() and not path.is_symlink():continue
        add(path)
        if path.is_dir() and not path.is_symlink():
            for parent,dirs,files in os.walk(path,followlinks=False):
                for name in dirs+files:add(Path(parent)/name)
                dirs[:]=[name for name in dirs if not (Path(parent)/name).is_symlink()]
    return dict(allocated_bytes=total,inodes=count)

def project_allocation(config):
    L=ROOT/'logs';paths=[f for f in L.iterdir() if 'expanded24' in f.name]
    paths += [ROOT/'cache',L/'__pycache__',ROOT/'runtime_cache_policy_20261019.json',ROOT/'record_process_exit.py',ROOT/'record_process_exit_before_expanded24_20261020.py',L/'reliability_v2_runtime_cache_20261019/expanded24_scope_fixture',L/'reliability_v2_runtime_cache_20261019/extension_expanded24_20261020.json']
    # Existing project cache is counted in full (conservative), not treated as zero growth.
    paths += [f for f in (L/'kit').rglob('*') if f.is_file() and f.stat().st_mtime>=config['kit_log_cutoff_epoch']]
    workspace=allocated(paths)
    ids=subprocess.check_output(['docker','ps','-aq','--filter','name=shijinsheng-piper-reliability-v2-expanded24-'],text=True).split()
    mounts=[];containers=[]
    for identifier in ids:
        assert re.fullmatch('[a-f0-9]{12,64}',identifier)
        info=json.loads(subprocess.check_output(['docker','inspect','--size',identifier],text=True))[0]
        name=info['Name'].lstrip('/');assert re.match(r'^shijinsheng-piper-reliability-v2-expanded24-[abc]-(?:native-)?20261020-',name)
        assert not info['State']['Running'],'Only measure after native exit; never interrupt a live job'
        full=info['Id'];assert re.fullmatch('[a-f0-9]{64}',full)
        assert all(x['Type']=='bind' for x in info['Mounts']),'Unaccounted anonymous/named volume; stop'
        engine=Path(info['LogPath']).parent;assert engine==Path('/var/lib/docker/containers')/full
        mounts.append(engine)
        graph=info.get('GraphDriver',{}).get('Data',{})
        upper_bound=0
        if graph.get('UpperDir'):
            upper=Path(graph['UpperDir']);assert str(upper).startswith('/var/lib/docker/overlay2/') and upper.name=='diff';mounts.append(upper)
            work=graph.get('WorkDir')
            if work:
                work=Path(work);assert work.parent==upper.parent and work.name=='work';mounts.append(work)
            method='exact_overlay_allocated_blocks_plus_metadata_reserve'
        else:
            # Docker's containerd image store exposes SizeRw, not the host snapshot path.
            # Add two filesystem blocks per reported changed path (including directories,
            # links, mount placeholders and driver injections) instead of omitting them.
            assert info['Driver']=='overlayfs' and isinstance(info.get('SizeRw'),int) and info['SizeRw']>=0
            changes=subprocess.check_output(['docker','diff',full],text=True).splitlines();assert all(x[:2] in ['A ','C ','D '] for x in changes)
            block=os.statvfs(ROOT).f_frsize;assert block<=4096
            upper_bound=info['SizeRw']+2*4096*(len(changes)+16)
            method='SizeRw_plus_conservative_two_blocks_per_changed_path'
        containers.append(dict(id=full,name=name,SizeRw_reported_bytes=info.get('SizeRw'),writable_layer_allocated_bytes_upper_bound=upper_bound,writable_layer_accounting=method,overlay_parent_metadata_reserve=65536))
    layer=dict(allocated_bytes=0,inodes=0)
    if mounts:
        # Mount ONLY these registered, stopped trial containers read-only. Read stat data only.
        cmd=['docker','run','--rm','--read-only','--network','none','--cpus=2','--memory=3g','--tmpfs','/tmp:rw,size=16m','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','/isaac-sim/python.sh']
        for i,path in enumerate(mounts):cmd += ['-v',str(path)+':/measure'+str(i)+':ro']
        code='from pathlib import Path\nimport os,json\nseen=set();total=count=0\nfor root in '+repr(['/measure'+str(i) for i in range(len(mounts))])+':\n for parent,dirs,files in os.walk(root,followlinks=False):\n  for name in [None]+files+dirs:\n   p=Path(parent) if name is None else Path(parent)/name;s=p.lstat();key=(s.st_dev,s.st_ino)\n   if key not in seen:seen.add(key);total+=s.st_blocks*512;count+=1\n  dirs[:]=[n for n in dirs if not (Path(parent)/n).is_symlink()]\nprint(json.dumps(dict(allocated_bytes=total,inodes=count)))'
        cmd += [IMAGE,'-c',code];layer=json.loads(subprocess.check_output(cmd,text=True,timeout=90))
    total=workspace['allocated_bytes']+layer['allocated_bytes']+sum(x['overlay_parent_metadata_reserve']+x['writable_layer_allocated_bytes_upper_bound'] for x in containers)+65536
    return dict(project_owned_allocated_bytes=total,workspace=workspace,own_container_layers_and_engine_logs=layer,containers=containers,measurement_transient_reserve=65536,scope='New expanded24 names plus complete own cache, new kit logs, own stopped Docker writable layers and exact engine/log directory blocks; containerd SizeRw receives conservative per-path overhead; symlinks not followed; shared immutable image is not a new allocation',baseline_owned_bytes=0,conservative_preexisting_preflight_and_cache_included=True)

def check_budget(plan,measurement=None):
    m=project_allocation(plan['resource_accounting_revision']) if measurement is None else measurement
    free=shutil.disk_usage(ROOT).free;old_free=shutil.disk_usage('/data/shijinsheng/piper-insertion').free
    bank=m['project_owned_allocated_bytes']-plan.get('bank_initial_owned_allocated_bytes',0)
    assert free>=5045649408 and old_free>=5045649408
    assert bank<=201326592 and m['project_owned_allocated_bytes']<=402653184
    return dict(**m,bank_owned_net_bytes=bank,bank_net_cap_bytes=201326592,campaign_cap_bytes=402653184,actual_free_bytes=free,old_actual_free_bytes=old_free,minimum_free_bytes=5045649408,original_initial_free_bytes=plan['initial_free_bytes'],observed_shared_filesystem_delta=plan['initial_free_bytes']-free,original_free_baseline_preserved=True)
