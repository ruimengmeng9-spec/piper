"""Sample only the existing native deep-withdrawal role, before composition."""
import types
import torch
from torch.distributions import Normal
class DeepHook:
 def __init__(self,raw,boundary,root):
  self.active=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device);self.sigma=torch.tensor([.01,.01,.01,.0075,.0075,.0075],device=raw.device);model=raw.deep_recovery_model;original=raw.deep_recovery_policy
  def inference(obs):
   mean=original(obs);public=model.get_actor_obs(obs).detach().clone();x=model.actor_obs_normalizer(public);feature=model.current_features(x);s=raw.supervisor
   native=((raw.deep_recovery_latch|(s.attempts>=2)|(-public[:,2]/20>.020)|(public[:,3:6].norm(dim=-1)>.08))&(s.mode==1))&raw.recovery_enabled&~raw.preparing
   selected=native&self.active;latent=mean.detach().clone()
   if selected.any():latent[selected]=Normal(mean[selected],self.sigma).sample()
   out=torch.where(selected[:,None],latent,mean);assert torch.equal(out[~selected],mean[~selected])
   anchor=root.actor(root.current_features(x));self.cache=dict(public=public,features=feature.detach().clone(),mean=mean.detach().clone(),latent=latent.detach().clone(),logp=Normal(mean,self.sigma).log_prob(latent).sum(-1).detach().clone(),selected=selected.clone(),native=native.clone(),root=anchor.detach().clone())
   return out
  raw.deep_recovery_policy=inference;compose=boundary.compose
  def composed(r,base,first):
   c=self.cache;assert torch.equal(c['native'],r.deep_recovery_latch&r.recovery_enabled&~r.preparing)
   assert not (c['selected']&((r.supervisor.mode!=1)|first|r.preparing)).any()
   assert torch.equal(base[c['selected']],c['latent'][c['selected']])
   out=compose(r,base,first);expected=base+torch.where(boundary.mask,boundary.delta,0.)
   assert torch.equal(out,expected)
   self.compose_cache=dict(base=base.detach().clone(),out=out.detach().clone(),delta=boundary.delta.detach().clone(),mask=boundary.mask.detach().clone())
   return out
  boundary.compose=composed

  if hasattr(raw,'_pre_physics_step'):
   native_pre=raw._pre_physics_step
   def pre(this,actions):
    target_before=this.targets[:,this.arm_ids].detach().clone()
    joint_before=this.robot.data.joint_pos[:,this.arm_ids].detach().clone()
    joint_error_before=(target_before-joint_before)/.06
    buffer=this.action_delay.buffer.detach();steps=this.action_delay.steps.detach()
    pending=torch.cat((torch.where((steps>=1)[:,None],buffer[0],0.),torch.where((steps>=2)[:,None],buffer[1],0.)),-1).clone()
    previous=this.actions.detach().clone()
    native_pre(actions)
    assert torch.equal(this.previous_actions,previous)
    self.compose_cache['preaction_joint_target6']=joint_error_before
    self.compose_cache['raw_preaction_target6']=target_before
    self.compose_cache['raw_preaction_joint6']=joint_before
    self.compose_cache['preaction_queue12']=pending
    self.compose_cache['before_brake']=self.compose_cache['out'].detach().clone()
    self.compose_cache['out']=this.r2_last_action.detach().clone()
    self.compose_cache['brake_state']=this.fixed_final_retry_hook.cache.detach().clone()
    self.compose_cache.update(supervisor_force_ema=this.supervisor.force_ema.detach().clone(),supervisor_trigger_depth=this.supervisor.trigger_depth.detach().clone(),release_hold=this.supervisor.release_hold.detach().clone(),clipped=this.actions.detach().clone(),delayed=this.action_delay.buffer[this.action_delay.steps,this.action_delay.ids].detach().clone(),delay=this.action_delay.steps.detach().clone(),refresh=this.contact_refresh_active.detach().clone())
   raw._pre_physics_step=types.MethodType(pre,raw)
