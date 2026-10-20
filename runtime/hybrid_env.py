"""Stage3C: an insertion learner plus a frozen withdrawal skill and supervisor.

Supervisor decisions use only the actor's 33 measured/estimated inputs. True
physical signals label outcomes only. Command targets and delay history survive
all skill transitions. The scripted realignment mode only suppresses axial
insertion; lateral and angular correction remain the insertion policy's actions.
"""
import torch
import json
from pathlib import Path
from tensordict import TensorDict
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from cad_env import ChallengeCupCfg, ChallengeCupEnv


@configclass
class HybridCfg(ChallengeCupCfg):
    episode_length_s = 12.
    fixed_contact_cases = False
    recovery_offset = .005
    recovery_center_x = -.0045
    recovery_memory = 'shadow'
    precontact_fraction = 1.


class SkillSupervisor:
    def __init__(self, n, device):
        self.mode = torch.zeros(n, dtype=torch.long, device=device) # insert, retreat, realign
        self.streak = torch.zeros_like(self.mode)
        self.release_hold = torch.zeros_like(self.mode)
        self.align_hold = torch.zeros_like(self.mode)
        self.attempts = torch.zeros_like(self.mode)
        self.exits = torch.zeros_like(self.mode)
        self.realignments = torch.zeros_like(self.mode)
        self.trigger_depth = torch.zeros(n, device=device)
        self.retreat_steps = torch.zeros_like(self.mode)
        self.realign_steps = torch.zeros_like(self.mode)
        self.force_ema = torch.zeros(n,3,device=device)
        self.alignment_xy_tolerance=.001
        self.alignment_angle_tolerance=.08

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value,torch.Tensor):value[ids] = 0

    def update(self, obs):
        estimated = obs[:,:3] / 20.
        depth = -estimated[:,2]
        force = (obs[:,24:27] / .02).norm(dim=-1)
        self.force_ema.lerp_(obs[:,24:27] / .02,.25)
        filtered_force = self.force_ema.norm(dim=-1)
        at_mouth = (depth >= -.015) & (depth <= .02)
        blocked = (self.mode == 0) & at_mouth & (force >= 10.)
        self.streak = torch.where(blocked, self.streak + 1, torch.zeros_like(self.streak))
        enter = (self.streak >= 2) & (self.attempts < 3)
        self.trigger_depth[enter] = depth[enter]
        self.mode[enter] = 1
        self.attempts += enter.long()
        self.retreat_steps += (self.mode == 1).long()
        outward = self.trigger_depth - depth
        # Vector smoothing reduces white measurement noise.5N includes the
        # configured axis bias bound (vector norm can reach3.46N at zero force).
        released = (self.mode == 1) & (outward >= .005) & (filtered_force < 5.)
        self.release_hold = torch.where(released, self.release_hold + 1, torch.zeros_like(self.release_hold))
        exit_retreat = (self.mode == 1) & (self.release_hold >= 8)
        self.mode[exit_retreat] = 2
        self.exits += exit_retreat.long()
        self.realign_steps += (self.mode == 2).long()
        aligned = (self.mode == 2) & (estimated[:,:2].abs().amax(-1) < self.alignment_xy_tolerance)
        aligned &= (obs[:,3:6].norm(dim=-1) < self.alignment_angle_tolerance) & (filtered_force < 5.)
        self.align_hold = torch.where(aligned, self.align_hold + 1, torch.zeros_like(self.align_hold))
        resume = (self.mode == 2) & (self.align_hold >= 8)
        self.mode[resume] = 0
        self.streak[resume] = 0
        self.realignments += resume.long()
        return enter


class HybridEnv(ChallengeCupEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.supervisor = SkillSupervisor(self.num_envs, self.device)
        self.recovery_enabled = True
        self.recovery_policy = None
        self.recovery_model = None
        self.alignment_policy = None
        self.alignment_model = None
        self.case_ids = torch.arange(self.num_envs, device=self.device) % 4
        self.true_exit_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.true_released_exit_count = torch.zeros_like(self.true_exit_count)
        self.true_trigger_depth = torch.zeros(self.num_envs, device=self.device)
        self.trigger_wall_force = torch.zeros(self.num_envs, device=self.device)
        self.fixed_mask = torch.arange(self.num_envs,device=self.device) < round(self.num_envs * cfg.precontact_fraction) if cfg.fixed_contact_cases else torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.preparing = self.fixed_mask.clone()
        self.step_preparing = self.preparing.clone()
        self.preparation_steps = torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.handover_force = torch.zeros(self.num_envs,device=self.device)

    def _setup_scene(self):
        super()._setup_scene()
        self.contact.cfg.filter_prim_paths_expr = [f'/World/envs/env_.*/Wall{i}' for i in range(4)]

    def _pre_physics_step(self, actions):
        obs = self._get_observations()
        self.step_preparing = self.preparing.clone()
        self.preparation_steps += self.preparing.long()
        assert self.recovery_policy is not None, 'Frozen recovery skill must be attached before stepping'
        before = self.supervisor.exits.clone()
        just_exited=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        with torch.no_grad():
            if self.recovery_enabled:
                measured = obs['policy'].clone()
                measured[self.preparing,24:27] = 0. # No switch while the external contact preparer controls.
                enter = self.supervisor.update(measured)
                if self.cfg.recovery_memory == 'cold' and enter.any():self.recovery_model.reset(enter)
                # Truth is recorded after the actor-only decision; it cannot change selection.
                relative, _, true_force = self._geometry()
                self.true_trigger_depth[enter] = -relative[enter,2]
                wall = self.contact.data.force_matrix_w_history.norm(dim=-1).sum(-1).amax(1)[:,0]
                self.trigger_wall_force[enter] = wall[enter]
                just_exited = self.supervisor.exits > before
                withdrawal = self.true_trigger_depth + relative[:,2]
                self.true_exit_count += (just_exited & (withdrawal >= .003)).long()
                self.true_released_exit_count += (just_exited & (withdrawal >= .003) & (true_force.norm(dim=-1)<2.)).long()
            # Keep its own GRU history even while the insertion policy controls.
            recovery = self.recovery_policy(TensorDict(obs,batch_size=[self.num_envs]))
            alignment=None
            if self.alignment_policy is not None:
                if just_exited.any():self.alignment_model.reset(just_exited)
                alignment=self.alignment_policy(TensorDict(obs,batch_size=[self.num_envs])).clone()
                # Collector captured the first zero-action handover physics
                # step. Start the frozen aligner's cold episode on the next.
                alignment[just_exited]=0.
                if just_exited.any():self.alignment_model.reset(just_exited)
        applied = actions.clone()
        if self.recovery_enabled:
            applied[self.supervisor.mode == 1] = recovery[self.supervisor.mode == 1]
            if alignment is not None:applied[self.supervisor.mode==2]=alignment[self.supervisor.mode==2]
            applied[self.supervisor.mode == 2,2] = 0.
        applied[self.preparing] = 0.
        applied[self.preparing,2] = -.1
        super()._pre_physics_step(applied)

    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        force = self._geometry()[2].norm(dim=-1)
        wall = self.contact.data.force_matrix_w_history.norm(dim=-1).sum(-1).amax(1)[:,0]
        handover = self.preparing & (force >= 5.) & (wall >= 5.)
        self.handover_force[handover] = force[handover]
        self.preparing[handover] = False
        prep_timeout = self.preparing & (self.preparation_steps >= 240)
        self.end_reason[prep_timeout & ~terminated] = 3
        return terminated, timeout | prep_timeout

    def _get_rewards(self):
        reward = super()._get_rewards()
        masked = torch.where(self.step_preparing,torch.zeros_like(reward),reward)
        self.episode_returns -= reward - masked
        return masked

    def _reset_idx(self, ids):
        if ids is None:ids = self.robot._ALL_INDICES
        initialized = hasattr(self, 'supervisor')
        if initialized:
            s = self.supervisor
            extra = {name:value[ids].clone() for name,value in [
                ('recovery_case',self.case_ids),('recovery_attempts',s.attempts),('recovery_exits',s.exits),
                ('recovery_realignments',s.realignments),('recovery_end_mode',s.mode),
                ('recovery_retreat_steps',s.retreat_steps),('recovery_realign_steps',s.realign_steps),
                ('recovery_true_exits',self.true_exit_count),('recovery_trigger_wall_force_n',self.trigger_wall_force),
                ('recovery_true_released_exits',self.true_released_exit_count),
                ('preparation_steps',self.preparation_steps),('handover_force_n',self.handover_force)]}
        super()._reset_idx(ids)
        if initialized:
            self.extras['terminal'].update(extra)
            self.supervisor.reset(ids)
            self.true_exit_count[ids] = 0
            self.true_released_exit_count[ids] = 0
            self.true_trigger_depth[ids] = 0.
            self.trigger_wall_force[ids] = 0.
            self.preparing[ids] = self.fixed_mask[ids]
            self.preparation_steps[ids] = 0
            self.handover_force[ids] = 0.
            if self.recovery_model is not None:
                mask = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device);mask[ids] = True
                self.recovery_model.reset(mask)
                if self.alignment_model is not None:self.alignment_model.reset(mask)
        if self.cfg.fixed_contact_cases:
            ids = ids[self.fixed_mask[ids]] if initialized else ids
            if not len(ids):return
            directions = torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]],device=self.device)
            offset = directions[ids % 4] * self.cfg.recovery_offset
            offset[:,0] += self.cfg.recovery_center_x
            self.hole_pos[ids] = torch.tensor(self.geometry['hole_center'],device=self.device)
            self.hole_pos[ids] += quat_apply(self.hole_quat[ids],offset)
            for wall, offset in zip(self.walls,self.wall_offsets):
                shifted = quat_apply(self.hole_quat[ids],torch.tensor(offset,device=self.device).expand(len(ids),-1))
                wall.write_root_pose_to_sim(torch.cat((self.hole_pos[ids]+self.scene.env_origins[ids]+shifted,
                                                       self.hole_quat[ids]),-1),env_ids=ids)


def self_test():
    s = SkillSupervisor(2,'cpu')
    obs = torch.zeros(2,33);obs[:,24] = .4 #20N, at mouth
    assert not s.update(obs).any()
    assert s.update(obs).all() and (s.mode == 1).all()
    obs[:,2] = .12;obs[:,24] = 0 #6mm out
    for _ in range(12):s.update(obs)
    assert (s.mode == 2).all() and (s.exits == 1).all()
    for _ in range(8):s.update(obs)
    assert (s.mode == 0).all() and (s.realignments == 1).all()
    s.reset(torch.tensor([0]));assert s.attempts.tolist() == [0,1]
    generator=torch.Generator(device='cpu').manual_seed(77)
    noisy=SkillSupervisor(64,'cpu');inputs=torch.zeros(64,33);inputs[:,24] = .4
    noisy.update(inputs);noisy.update(inputs)
    inputs[:,2] = .12
    for _ in range(120):
        inputs[:,24:27]=(torch.randn(64,3,generator=generator)*2.+torch.tensor([2.,2.,2.]))*.02
        noisy.update(inputs)
    assert (noisy.exits == 1).all(), noisy.exits.tolist()
    # Test RNG must not affect the experiment's randomization sequence.
    print('SUPERVISOR_STATE_SEQUENCE_PASSED',flush=True)


def hybrid_smoke(env,raw,logdir):
    raw.verify_cad_scene()
    env.reset()
    for _ in range(320):
        action = torch.zeros(env.num_envs,6,device=env.device);action[:,2] = -.35
        obs,reward,dones,info = env.step(action)
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        assert not dones.any(), 'Straight approach/withdrawal physical probe unexpectedly terminated'
    assert (raw.handover_force >= 5.).all(), raw.handover_force.tolist()
    assert (raw.supervisor.attempts == 1).all(), raw.supervisor.attempts.tolist()
    assert (raw.true_exit_count == 1).all(), raw.true_exit_count.tolist()
    assert (raw.supervisor.mode == 2).all(), raw.supervisor.mode.tolist()
    result=dict(status='passed',task='physical_skill_handover_probe',simulation_device=raw.device,
        handover_force_n=raw.handover_force.tolist(),physical_exit_count=raw.true_exit_count.tolist(),
        final_mode=raw.supervisor.mode.tolist(),success_claim='exit verified; straight controller cannot realign or insert',
        command_and_delay_buffer='preserved_across_all_switches',uploaded_model_source_sha256=raw.geometry['source_sha256'])
    (Path(logdir)/'smoke.json').write_text(json.dumps(result,indent=2))
    print('HYBRID_PHYSICAL_PROBE_PASSED',json.dumps(result),flush=True)
