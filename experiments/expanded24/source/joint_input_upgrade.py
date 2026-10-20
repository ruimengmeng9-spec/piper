"""Future preparation only: expose public joint5 without changing initial actions."""
import torch
def expose_joint5(model):
 assert model.current_actor_mask.shape==(1,33)
 assert model.current_actor_mask[0].nonzero().flatten().tolist()==[2,3,4,5,27,28,29,30,31,32]
 assert model.actor[0].in_features==33 and model.actor[-1].out_features==6
 # Index10 is the existing public joint5 encoder. Its previous masked input was0.
 with torch.no_grad():
  model.actor[0].weight[:,10].zero_()
  model.current_actor_mask[0,10]=1
