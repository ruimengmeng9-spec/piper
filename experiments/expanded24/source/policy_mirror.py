"""Public kinematic reflection, training/diagnostic supervision only."""
import torch
from tensordict import TensorDict
ACTION_SIGNS=[1.,-1.,1.,-1.,1.,-1.]
JOINT_SIGNS=[-1.,1.,1.,-1.,1.,-1.]
OBS_SIGNS=[1.,-1.,1.]+[-1.,1.,-1.]+JOINT_SIGNS+JOINT_SIGNS+ACTION_SIGNS+[1.,-1.,1.]+[1.,-1.,1.]+[-1.,1.,-1.]
assert len(OBS_SIGNS)==33
def observations(obs):
 result=obs.clone();x=obs['policy'];y=x*torch.tensor(OBS_SIGNS,device=x.device,dtype=x.dtype)
 unavailable=x[:,24:27].abs().amax(-1)>2.
 y[unavailable,24:27]=3. # Sentinel is a missing-data symbol, not a force vector.
 result['policy']=y
 return result
def actions(value):return value*torch.tensor(ACTION_SIGNS,device=value.device,dtype=value.dtype)
