import torch
from domain_value_upgrade import append_domain_context
from history_value import force_history_inputs,history_inputs,privileged_delta6,controller_state_inputs,queue_inputs

class HistoryShadow:
    def __init__(self, raw, boundary):
        self.model = boundary.model
        self.model.critic.memory.eval()
        self.feature = torch.zeros(raw.num_envs,75,device=raw.device)
        self.value = torch.zeros(raw.num_envs,device=raw.device)
        self.hidden = torch.zeros(1,raw.num_envs,48,device=raw.device)
        self.active = torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
        friction = raw.friction_scales.detach().clone()
        stiffness = raw.stiffness_scales.detach().clone()
        assert raw.cfg.privileged_critic
        self.truth6 = torch.zeros(raw.num_envs,6,device=raw.device)
        self.previous_truth_force = torch.full_like(self.truth6[:,3:],3.)
        self.force_encoding = torch.zeros_like(self.truth6[:,3:])
        self.extra6 = torch.zeros_like(self.truth6)
        self.latency1 = torch.zeros(raw.num_envs,1,device=raw.device)
        self.release1 = torch.zeros(raw.num_envs,1,device=raw.device)
        self.control4 = torch.zeros(raw.num_envs,4,device=raw.device)
        self.queue12 = torch.zeros(raw.num_envs,12,device=raw.device)
        self.joint_target6 = torch.zeros(raw.num_envs,6,device=raw.device)
        inference = boundary.act_inference
        def act(obs):
            # Capture the input before deep action sampling and physics, without another observation call.
            applied_target=raw.targets[:,raw.arm_ids].detach().clone()
            measured_joint=raw.robot.data.joint_pos[:,raw.arm_ids].detach().clone()
            self.joint_target6=((applied_target-measured_joint)/.06).detach().clone()
            self.queue12 = queue_inputs(raw.action_delay.buffer.detach(),raw.action_delay.steps.detach()).detach().clone()
            truth = obs['critic'].detach()
            self.truth6 = torch.cat((truth[:,:3],truth[:,24:27]),-1).clone()
            return inference(obs)
        boundary.act_inference = act
        original = boundary.compose
        def compose(r,base,first):
            action = original(r,base,first)
            self.feature = append_domain_context(boundary.context.detach(),friction,stiffness).detach().clone()
            with torch.no_grad():
                self.control4 = controller_state_inputs(r.supervisor.force_ema.detach(),r.supervisor.trigger_depth.detach()).detach().clone()
                hold = r.supervisor.release_hold.detach()
                assert hold.dtype == torch.int64 and ((hold>=0)&(hold<=8)).all()
                self.release1 = (hold.float()[:,None]*.125).clone()
                delay = r.action_delay.steps.detach()
                assert ((delay>=0)&(delay<=2)).all()
                self.latency1 = (delay.float()[:,None]*.5).clone()
                self.extra6 = privileged_delta6(boundary.public,self.truth6).detach().clone()
                self.force_encoding=force_history_inputs(self.truth6[:,3:],self.previous_truth_force).detach().clone()
                self.previous_truth_force.copy_(self.truth6[:,3:])
                H,self.hidden = self.model.critic.step(history_inputs(self.feature,boundary.public),self.hidden,self.active,self.truth6[:,3:],self.force_encoding)
                self.value = (self.model.critic.combine_expected(self.feature,H,self.extra6,self.latency1,self.release1,self.control4,self.queue12,self.joint_target6)[:,0]*100.).detach().clone()
            return action
        boundary.compose = compose
        raw.critic_shadow = self

    def reset(self, done):
        self.hidden[:,done.bool().flatten()] = 0
        self.previous_truth_force[done.bool().flatten()] = 3.
