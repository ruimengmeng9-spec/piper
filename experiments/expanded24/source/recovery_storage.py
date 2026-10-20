"""Durable learner state and byte-exact duplicate storage for an interrupted replica."""
import copy,hashlib,io,json,os,random,shutil,tarfile,time
from pathlib import Path
import numpy as np
import torch
from boundary_rl import model_hash
from guarded_value import optimizer_digest
class RecoveryStorage:
 def __init__(self,D):
  self.D=Path(D);self.archive=Path('/workspace/piper-insertion/logs/reliability_v2_brake_ppo_20261017_partial.tar.gz');self.archive_sha256='15ce1118b54a930d418095bc45282e07049f2cddc820ae4c74b34f5cf0a03989';assert self.sha(self.archive)==self.archive_sha256
  with tarfile.open(self.archive) as z:
   self.reference=json.load(z.extractfile('reliability_v2_brake_ppo_20261017/partial_failure_result.json'));raw=json.load(z.extractfile('reliability_v2_brake_ppo_20261017/partial_raw_manifest.json'))['files'];self.items={x['name']:x for x in raw};rows=[json.loads(s) for s in z.extractfile('reliability_v2_brake_ppo_20261017_train_random/post_episode_outcomes.jsonl').read().decode().splitlines()];self.outcomes={(x['batch'],x['env']):x for x in rows}
  self.refs={};self.comparisons=[]
 @staticmethod
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 def wait_space(self,batch,progress):
  if shutil.disk_usage(self.D).free>5045649408+32*1024**2:return True
  progress('waiting_storage_reserve',next_batch=batch+1,latest_learner_checkpoint=str(self.D/'learner_latest.pt'));deadline=time.monotonic()+600
  while shutil.disk_usage(self.D).free<=5045649408+32*1024**2:
   if time.monotonic()>deadline:return False
   time.sleep(5)
  return True
 def checkpoint(self,model,boundary,actor_opt,value_opt,completed,value_fits,fit_attempts,history,gate,raw):
  def cpu(x):
   if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
   if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
   if isinstance(x,list):return [cpu(v) for v in x]
   if isinstance(x,tuple):return tuple(cpu(v) for v in x)
   return copy.deepcopy(x)
  fingerprints={}
  for name in ['friction_scales','stiffness_scales','peg_masses','grip_axial_offsets','force_noise_scales']:
   x=getattr(raw,name);fingerprints[name]=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
  state=dict(model_state_dict=cpu(model.state_dict()),boundary_state_dict=cpu(boundary.model.state_dict()),actor_optimizer=cpu(actor_opt.state_dict()),value_optimizer=cpu(value_opt.state_dict()),completed_updates=completed,value_fits=value_fits,fit_attempts=fit_attempts,history=copy.deepcopy(history),value_gate=copy.deepcopy(gate),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),fixed_domain_hashes=fingerprints,source_qualified_sha256='d092d8c3dbb5a440d1f90f0e0c005bf1b754a5d3d7107234f7a50a13812abf1f');buf=io.BytesIO();torch.save(state,buf);data=buf.getvalue();assert shutil.disk_usage(self.D).free>=5045649408+len(data)+2*1024**2;p=self.D/'learner_latest.pt';temp=self.D/'learner_latest.pt.tmp';temp.write_bytes(data);os.replace(temp,p);(self.D/'learner_latest_manifest.json').write_text(json.dumps(dict(sha256=self.sha(p),bytes=len(data),completed_actor_updates=completed,collected_batches=len(history),actor_model_hash=model_hash(model),actor_optimizer_digest=optimizer_digest(actor_opt),value_optimizer_digest=optimizer_digest(value_opt),RNG_and_optimizer_saved=True),indent=2))
 def compare_and_deduplicate(self,row,rows):
  i=row['batch'];comparison=dict(batch=i,replica_of_interrupted_training=i<=5)
  if i<=5:
   old=self.reference['history'][i-1];comparison.update(actor_state_exact=row['actor_sha256_after']==old['actor_sha256_after'],actor_optimizer_exact=row['actor_optimizer_sha256_after']==old['actor_optimizer_sha256_after'],native_outcomes_exact=all(self.outcomes[(i,x['env'])]==x for x in rows),value_optimizer_exact=(row.get('value_fit',{}).get('retained_optimizer_sha256')==old.get('value_fit',{}).get('retained_optimizer_sha256')))
  for p in sorted(self.D.glob('*.npz')):
   if p.name not in self.items:continue
   item=self.items[p.name]
   if self.sha(p)!=item['sha256']:continue
   assert self.sha(self.archive)==self.archive_sha256
   with tarfile.open(self.archive) as z:data=z.extractfile(item['relative_path']).read()
   assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'];assert p.resolve().parent==self.D.resolve()
   self.refs[p.name]=dict(archive=str(self.archive),archive_sha256=self.archive_sha256,member=item['relative_path'],sha256=item['sha256'],bytes=item['bytes']);(self.D/'recovery_tensor_references.json').write_text(json.dumps(self.refs,indent=2));p.unlink()
  comparison['byte_exact_duplicate_tensor_files_referenced']=len(self.refs);self.comparisons.append(comparison);(self.D/'recovery_replay_comparison.json').write_text(json.dumps(dict(repeated_seed=1099,not_independent_validation=True,comparisons=self.comparisons),indent=2))
