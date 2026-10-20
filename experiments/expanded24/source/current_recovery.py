"""Current-observation neural specialist for identified withdrawal OOD bouts."""
import torch
from torch.distributions import Normal
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.utils import unpad_trajectories

class CurrentObsRecoveryActorCriticRecurrent(ActorCriticRecurrent):
    def __init__(self,obs,obs_groups,num_actions,**kwargs):
        super().__init__(obs,obs_groups,num_actions,**kwargs)
        assert num_actions==6 and self.get_actor_obs(obs).shape[-1]==33 and obs_groups['policy']==['policy'] and not self.state_dependent_std
        for parameter in self.parameters():parameter.requires_grad_(False)
        for parameter in self.actor.parameters():parameter.requires_grad_(True)
        self.register_buffer('current_actor_mean',torch.zeros(1,33));self.register_buffer('current_actor_std',torch.ones(1,33))
        mask=torch.zeros(1,33);mask[:,[2,3,4,5,27,28,29,30,31,32]]=1
        self.register_buffer('current_actor_mask',mask)

    def current_features(self,x):
        return ((x-self.current_actor_mean)/self.current_actor_std)*self.current_actor_mask

    def act_inference(self,obs):
        x=self.actor_obs_normalizer(self.get_actor_obs(obs));self.memory_a(x)
        return self.actor(self.current_features(x))

    def act(self,obs,masks=None,hidden_state=None):
        x=self.actor_obs_normalizer(self.get_actor_obs(obs));self.memory_a(x,masks,hidden_state)
        if masks is not None:x=unpad_trajectories(x,masks).squeeze(0)
        mean=self.actor(self.current_features(x));std=self.std.expand_as(mean) if self.noise_std_type=='scalar' else torch.exp(self.log_std).expand_as(mean);self.distribution=Normal(mean,std)
        return self.distribution.sample()

    def update_normalization(self,obs):pass
