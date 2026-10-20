"""Portable, hash-bound compressed index shared by new archive workflow readers."""
from pathlib import Path
import base64,copy,gzip,hashlib,json,tarfile

def sha(raw):return hashlib.sha256(raw).hexdigest()

def read_index(path):
    value=json.loads(Path(path).read_bytes())
    if value.get('schema')=='archive-index-inline-gzip-v1':
        packed=base64.b64decode(value['gzip_base64'],validate=True);assert sha(packed)==value['gzip_sha256']
        raw=gzip.decompress(packed);assert len(raw)==value['original_bytes'] and sha(raw)==value['original_sha256']
        decoded=json.loads(raw);assert len(decoded['volumes'])==value['volume_count'];value=decoded
    assert isinstance(value['volumes'],list)
    paths=[v['server_path'] for v in value['volumes']];assert len(paths)==len(set(paths))
    return value

def write_raw_index(raw,path):
    path=Path(path);assert not path.exists();value=json.loads(raw)
    packed=gzip.compress(raw,compresslevel=9,mtime=0)
    wrapper=dict(schema='archive-index-inline-gzip-v1',gzip_base64=base64.b64encode(packed).decode('ascii'),gzip_sha256=sha(packed),original_sha256=sha(raw),original_bytes=len(raw),volume_count=len(value['volumes']))
    with path.open('x',encoding='utf-8') as f:json.dump(wrapper,f,separators=(',',':'))
    assert read_index(path)==value
    return wrapper

def stage_for_deployment(source,destination):
    # Preserve exact original bytes when staging a previously plain index.
    raw=Path(source).read_bytes();value=json.loads(raw)
    if value.get('schema')=='archive-index-inline-gzip-v1':
        read_index(source)
        with Path(destination).open('xb') as f:f.write(raw)
        assert read_index(destination)==read_index(source)
    else:write_raw_index(raw,destination)

def append_verified_volume(source,destination,volume):
    data=read_index(source);archive=Path(volume['local_path']);raw=archive.read_bytes()
    assert len(raw)==volume['bytes'] and sha(raw)==volume['sha256']
    prior=[x for x in data['volumes'] if x['server_path']==volume['server_path']]
    if prior:
        assert len(prior)==1 and prior[0]['sha256']==volume['sha256'] and prior[0]['bytes']==volume['bytes']
        return False
    data=copy.deepcopy(data);data['volumes'].append(volume)
    write_raw_index(json.dumps(data,separators=(',',':')).encode(),destination)
    assert read_index(destination)['volumes'][:-1]==read_index(source)['volumes']
    return True

def restore_member_bytes(index,server_path,member,expected_sha256):
    rows=[v for v in read_index(index)['volumes'] if v['server_path']==server_path];assert len(rows)==1
    row=rows[0];archive=Path(row['local_path']);raw=archive.read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
    with tarfile.open(archive) as z:
        item=z.getmember(member);assert item.isfile();restored=z.extractfile(item).read()
    assert sha(restored)==expected_sha256
    return restored

def verify_required_archives(index,server_paths):
    rows={x['server_path']:x for x in read_index(index)['volumes']};verified=[]
    for path in server_paths:
        row=rows[path];archive=Path(row['local_path']);assert archive.stat().st_size==row['bytes'] and sha(archive.read_bytes())==row['sha256'];verified.append(path)
    return verified


def replace_index(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.staged');assert not temporary.exists()
    write_raw_index(json.dumps(value,separators=(',',':')).encode(),temporary)
    assert read_index(temporary)==value;temporary.replace(path)
