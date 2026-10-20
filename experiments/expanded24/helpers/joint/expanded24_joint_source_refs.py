import hashlib,io,json,tarfile
from pathlib import Path
def expand_shared_sources(blobs,package_path=None):
 out=dict(blobs)
 for name,data in blobs.items():
  if not name.endswith('/shared_source_reference.json'):continue
  ref=json.loads(data);base=name[:-len('shared_source_reference.json')];prefix=base+'executable/';assert ref['schema']=='shared-source-v1' and ref['logical_executable_prefix']==prefix
  manifest=blobs[prefix+'source_manifest.json'];assert hashlib.sha256(manifest).hexdigest()==ref['source_manifest_sha256'];M=json.loads(manifest)['executable_hashes'];assert len(M)==65
  candidates=[Path(package_path)] if package_path is not None else [Path(__file__).resolve().parent/n for n in ['expanded24_shared_source_20261020.tar.gz','expanded24_a_source.tar.gz']]
  paths=[q for q in candidates if q.is_file() and hashlib.sha256(q.read_bytes()).hexdigest()==ref['source_archive_sha256']];assert paths,'Required SHA-bound shared source package missing'
  with tarfile.open(paths[0]) as z:
   members=z.getmembers();assert all(m.isfile() for m in members);p={m.name:z.extractfile(m).read() for m in members}
  assert set(p)=={'executable/'+n for n in M}|{'executable/source_manifest.json'} and p['executable/source_manifest.json']==manifest
  for n,h in M.items():
   assert '/' not in n and '\\' not in n and n.endswith('.py') and hashlib.sha256(p['executable/'+n]).hexdigest()==h
   assert prefix+n not in out;out[prefix+n]=p['executable/'+n]
 return out
