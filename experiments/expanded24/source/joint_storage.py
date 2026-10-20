"""Atomic end-of-batch learner snapshots, with full-round storage admission."""
import copy,hashlib,io,json,os,random,shutil
from pathlib import Path
import numpy as np
import torch
from boundary_rl import model_hash
from guarded_value import optimizer_digest
MINFREE=5045649408
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return copy.deepcopy(x)
def durable(D,model,boundary,actor_opt,value_opt,history,gate,completed,fits,attempts,raw):
 state=dict(deep_state_dict=cpu(model.state_dict()),boundary_state_dict=cpu(boundary.model.state_dict()),actor_optimizer=cpu(actor_opt.state_dict()),value_optimizer=cpu(value_opt.state_dict()),history=copy.deepcopy(history),value_gate=copy.deepcopy(gate),completed_updates=completed,value_fits=fits,fit_attempts=attempts,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),source_deep_sha256='055571a29e42fc6e36b347823cb5aa57e6ea1505af6fb59226ab5108d33eabdf',fixed_domain_hashes={n:hashlib.sha256(getattr(raw,n).detach().cpu().numpy().tobytes()).hexdigest() for n in ['friction_scales','stiffness_scales','peg_masses','grip_axial_offsets','force_noise_scales']})
 state['fixed_final_retry_brake_latched']=cpu(raw.fixed_final_retry_hook.brake.latched)
 b=io.BytesIO();torch.save(state,b);data=b.getvalue();assert shutil.disk_usage(D).free>=MINFREE+len(data)+2*1024**2;p=Path(D)/'learner_latest.pt';tmp=p.with_suffix('.pt.tmp');tmp.write_bytes(data);os.replace(tmp,p)
 (Path(D)/'learner_latest_manifest.json').write_text(json.dumps(dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),batches=len(history),completed_updates=completed,actor_model_hash=model_hash(model),actor_optimizer_digest=optimizer_digest(actor_opt),value_optimizer_digest=optimizer_digest(value_opt),RNG_and_optimizer_saved=True),indent=2))
