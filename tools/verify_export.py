"""Verify the public source export without importing simulator dependencies."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'PUBLIC_EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
for item in m['files']:
 p=root/item['path'];data=p.read_bytes()
 assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],item['path']
s=root/'experiments/expanded24/source'
source=json.loads((s/'source_manifest.json').read_text(encoding='utf-8'))['executable_hashes']
assert len(source)==65
for name,expected in source.items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==expected,name
print('Verified',len(m['files']),'exported files and all 65 scientific source files.')
