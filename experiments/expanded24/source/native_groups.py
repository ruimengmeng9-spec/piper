"""Combine complete native128 cohorts without changing physical vector width."""
import copy,hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
import torch
def zero_record(x):
 if isinstance(x,torch.Tensor):return torch.zeros_like(x)
 if isinstance(x,dict):return {k:zero_record(v) for k,v in x.items()}
 if isinstance(x,tuple):return tuple(zero_record(v) for v in x)
 raise TypeError(type(x))
def concatenate(records):
 first=records[0]
 if isinstance(first,torch.Tensor):return torch.cat(records,0)
 if isinstance(first,dict):return {k:concatenate([r[k] for r in records]) for k in first}
 if isinstance(first,tuple):return tuple(concatenate([r[i] for r in records]) for i in range(len(first)))
 raise TypeError(type(first))
def combine(units):
 assert 1<=len(units)<=2 and all(u for u in units)
 padding=[zero_record(u[0]) for u in units]
 return [concatenate([u[t] if t<len(u) else pad for u,pad in zip(units,padding)]) for t in range(max(map(len,units)))]
def arrays(records,probe,hook,brake):
 K={k:torch.stack([r[0][k] for r in records]) for k in records[0][0]};C=torch.stack([r[1] for r in records]);V=torch.stack([r[2] for r in records]);alive=torch.stack([r[5] for r in records]);M=K['selected']&alive;phase=torch.zeros_like(V)
 for e in range(M.shape[1]):
  q=M[:,e].nonzero().flatten();phase[q,e]=torch.arange(len(q),device=M.device)/max(len(q)-1,1)
 public=torch.stack([r[6] for r in records]);shared=torch.equal(public,K['public'])
 out=dict(public33=public[:,probe].cpu().numpy(),deep_features=K['features'][:,probe].cpu().numpy(),old_mean=K['mean'][:,probe].cpu().numpy(),root_mean=K['root'][:,probe].cpu().numpy(),latent=K['latent'][:,probe].cpu().numpy(),old_logp=K['logp'][:,probe].cpu().numpy(),selected=M[:,probe].cpu().numpy(),native=K['native'][:,probe].cpu().numpy(),alive=alive[:,probe].cpu().numpy(),critic_features=C[:,probe].cpu().numpy(),prior_value=V[:,probe].cpu().numpy(),reward=torch.stack([r[3] for r in records])[:,probe].cpu().numpy(),done=torch.stack([r[4] for r in records])[:,probe].cpu().numpy(),phase=phase[:,probe].cpu().numpy(),before_brake=torch.stack([r[7]['before_brake'] for r in records])[:,probe].cpu().numpy(),brake_state=torch.stack([r[7]['brake_state'] for r in records])[:,probe].cpu().numpy(),brake_columns=np.array(brake.columns),composed=torch.stack([r[7]['out'] for r in records])[:,probe].cpu().numpy(),base=torch.stack([r[7]['base'] for r in records])[:,probe].cpu().numpy(),delta=torch.stack([r[7]['delta'] for r in records])[:,probe].cpu().numpy(),boundary_mask=torch.stack([r[7]['mask'] for r in records])[:,probe].cpu().numpy(),clipped=torch.stack([r[8] for r in records])[:,probe].cpu().numpy(),delayed=torch.stack([r[9] for r in records])[:,probe].cpu().numpy(),delay_steps=torch.stack([r[7]['delay'] for r in records])[:,probe].cpu().numpy(),refresh=torch.stack([r[7]['refresh'] for r in records])[:,probe].cpu().numpy(),sigma=hook.sigma.cpu().numpy(),env=probe.cpu().numpy(),public_shared=np.array(shared))
 if not shared:out['deep_public33']=K['public'][:,probe].cpu().numpy()
 return out
def compare_first(records,rows,hook,brake,D,actor_hash,opt_hash):
 archive=Path('/workspace/piper-insertion/logs/reliability_v2_joint_braked_deep_20261017_review.tar.gz');expected='27eb53e93e270e19cb6b86b5efa47a5698cd8c13152a37d2f6eb27fa253e162b';assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
 G='reliability_v2_joint_braked_deep_20261017';prefix=G+'_train_random/';probe=torch.tensor([0,1,2,3,32,33,34,35],device=hook.active.device);current=arrays(records,probe,hook,brake)
 with tarfile.open(archive) as z:
  oldrows=[json.loads(x) for x in z.extractfile(prefix+'deep_episode_outcomes.jsonl').read().decode().splitlines()];old=np.load(io.BytesIO(z.extractfile(prefix+'native_probes_batch1.npz').read()));history=json.load(z.extractfile(prefix+'deep_train_result.json'))['batches'][0]
  exact_rows=all({k:r[k] for k in o}==o for r,o in zip(rows,oldrows));array_checks={k:bool(np.array_equal(current[k],old[k])) for k in old.files};same_actor=actor_hash==history['actor_sha256_after'];same_opt=opt_hash==history['actor_optimizer_sha256_after']
 report=dict(status='passed' if exact_rows and all(array_checks.values()) and same_actor and same_opt else 'failed',replica_not_new_independent_data=True,old_archive_sha256=expected,outcomes_exact=exact_rows,probe_arrays_exact=array_checks,actor_exact=same_actor,actor_optimizer_exact=same_opt,no_old_arrays_used_as_training_input=True)
 (Path(D)/'first_cohort_replication.json').write_text(json.dumps(report,indent=2));return report['status']=='passed'
