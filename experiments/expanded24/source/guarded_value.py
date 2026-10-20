"""Training-only value selection with an epoch-zero incumbent and fresh gates."""
import copy,hashlib,json
import torch
from torch import nn
from context_ppo import metrics,predict

def value_gate_passed(p):
 return p['mse_ratio']<=.9 and p['residual_variance_ratio']<=.9 and all(v['mse_ratio']<=1.05 for v in p['phases'].values())

def actor_permitted(batch,prior):return batch>=2 and value_gate_passed(prior)

def optimizer_digest(opt):
 h=hashlib.sha256()
 def add(x):
  if isinstance(x,torch.Tensor):h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):h.update(str(k).encode());add(x[k])
  elif isinstance(x,(list,tuple)):
   for y in x:add(y)
  else:h.update(json.dumps(x,sort_keys=True).encode())
 add(opt.state_dict());return h.hexdigest()

def fit_value(model,opt,C,T,eid,phase,epochs=40):
 held=((eid//4)%4)==0;train=~held;assert int(held.sum())>200 and int(train.sum())>500
 with torch.no_grad():before=metrics(predict(model,C),T,phase,held)
 before_weights=copy.deepcopy(model.critic.state_dict());before_opt=optimizer_digest(opt)
 best=before['mse'];state=copy.deepcopy(before_weights);moment=copy.deepcopy(opt.state_dict());best_epoch=0;steps=0
 for epoch in range(epochs):
  ids=train.nonzero().flatten();ids=ids[torch.randperm(len(ids),device=C.device)]
  for q in ids.split(1024):
   prediction=predict(model,C[q]);loss=((prediction-T[q])/100.).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.critic.parameters(),1.);opt.step();steps+=1
  # Same full-batch prediction shape as the epoch-zero score, avoiding tie drift.
  with torch.no_grad():score=float((predict(model,C)[held]-T[held]).square().mean())
  if score<best:best=score;state=copy.deepcopy(model.critic.state_dict());moment=copy.deepcopy(opt.state_dict());best_epoch=epoch+1
 model.critic.load_state_dict(state);opt.load_state_dict(moment)
 with torch.no_grad():after=metrics(predict(model,C),T,phase,held)
 assert after['mse']<=before['mse']+1e-5
 retained=best_epoch>0
 if not retained:
  assert all(torch.equal(v,before_weights[k]) for k,v in model.critic.state_dict().items());assert optimizer_digest(opt)==before_opt
 return dict(before=before,after=after,best_epoch=best_epoch,optimizer_steps=steps,epochs=epochs,epoch0_candidate=True,retained_fit=retained,epoch0_optimizer_sha256=before_opt,retained_optimizer_sha256=optimizer_digest(opt),heldout_rule='complete env//4 modulo4 equals0;no same-batch fit can authorize actor updates')
