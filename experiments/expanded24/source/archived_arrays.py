import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
class ArchivedArrays:
 def __init__(self,F):self.F=Path(F);p=self.F/'recovery_tensor_references.json';self.refs=json.loads(p.read_text()) if p.exists() else {};self.original_load=np.load
 def exists(self,p):return Path(p).exists() or (Path(p).parent==self.F and Path(p).name in self.refs)
 def load(self,p,*args,**kw):
  if isinstance(p,(str,Path)) and not Path(p).exists() and Path(p).parent==self.F and Path(p).name in self.refs:
   r=self.refs[Path(p).name];assert hashlib.sha256(Path(r['archive']).read_bytes()).hexdigest()==r['archive_sha256']
   with tarfile.open(r['archive']) as z:data=z.extractfile(r['member']).read()
   assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'];return self.original_load(io.BytesIO(data),*args,**kw)
  return self.original_load(p,*args,**kw)
