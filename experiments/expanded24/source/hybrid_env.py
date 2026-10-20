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


def coverage_case_ids(env_ids):
    return torch.where((env_ids // 16) % 2 == 1, env_ids % 2 + 2, env_ids % 4)

@configclass
class HybridCfg(ChallengeCupCfg):
    episode_length_s = 12.
    fixed_contact_cases = False
    recovery_offset = .005
    recovery_center_x = -.0045
    recovery_memory = 'shadow'
    precontact_fraction = 1.
    train_stop_failed_retry = False
    initial_prealign = False
    lateral_cases_only = False
    negative_lateral_case_only = False
    replay_original_axial_cases = False
    replay_original_positive_lateral = False
    training_joint_margin_cost = 0.
    recovery_offset_min = -1.
    block_force_filter = 'instant'
    coverage_lateral_focus = False


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
        self.block_force_filter='instant'
        self.alignment_xy_tolerance=.001
        self.alignment_angle_tolerance=.08

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value,torch.Tensor):value[ids] = 0

    def update(self, obs):
        estimated = obs[:,:3] / 20.
        depth = -estimated[:,2]
        force = (obs[:,24:27] / .02).norm(dim=-1)
        force_valid=obs[:,24:27].abs().amax(-1)<=2.
        previous_ema=self.force_ema.clone()
        self.force_ema.lerp_(obs[:,24:27] / .02,.25)
        self.force_ema[~force_valid]=previous_ema[~force_valid]
        filtered_force = self.force_ema.norm(dim=-1)
        at_mouth = (depth >= -.015) & (depth <= .02)
        trigger_force=force if self.block_force_filter=='instant' else torch.where(force>=40.,force,filtered_force)
        blocked = (self.mode == 0) & at_mouth & (trigger_force >= 10.) & force_valid
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
        self.supervisor.block_force_filter=cfg.block_force_filter
        self.recovery_enabled = True
        self.recovery_policy = None
        self.recovery_model = None
        self.alignment_policy = None
        self.alignment_model = None
        self.initial_policy = None
        self.initial_model = None
        self.near_alignment_model=None
        self.near_alignment_policy=None
        self.near_alignment_control_steps=torch.zeros(self.num_envs,device=self.device)
        self.deep_recovery_model=None
        self.deep_recovery_policy=None
        self.deep_recovery_latch=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.deep_recovery_control_steps=torch.zeros(self.num_envs,device=self.device)
        self.late_initial_model=None
        self.late_initial_policy=None
        self.mirror_model = None
        self.mirror_policy = None
        self.training_teacher_model = None
        self.training_teacher_policy = None
        self.case_ids = torch.full((self.num_envs,),3,dtype=torch.long,device=self.device) if cfg.negative_lateral_case_only else torch.arange(self.num_envs, device=self.device) % (2 if cfg.lateral_cases_only else 4) + (2 if cfg.lateral_cases_only else 0)
        if cfg.coverage_lateral_focus: self.case_ids = coverage_case_ids(torch.arange(self.num_envs, device=self.device))
        self.true_exit_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.true_released_exit_count = torch.zeros_like(self.true_exit_count)
        self.true_trigger_depth = torch.zeros(self.num_envs, device=self.device)
        self.trigger_wall_force = torch.zeros(self.num_envs, device=self.device)
        self.fixed_mask = torch.arange(self.num_envs,device=self.device) < round(self.num_envs * cfg.precontact_fraction) if cfg.fixed_contact_cases else torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.preparing = self.fixed_mask.clone()
        self.step_preparing = self.preparing.clone()
        self.preparation_steps = torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.handover_force = torch.zeros(self.num_envs,device=self.device)
        self.failed_retry=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.training_episode_offset=torch.full((self.num_envs,),cfg.recovery_offset,device=self.device)
        self.main_control_steps=torch.zeros(self.num_envs,device=self.device)
        self.main_z_sum=torch.zeros_like(self.main_control_steps)
        self.main_z_clipped=torch.zeros_like(self.main_control_steps)
        self.main_xy_clipped=torch.zeros_like(self.main_control_steps)
        self.main_rot_clipped=torch.zeros_like(self.main_control_steps)
        self.late_initial_control_steps=torch.zeros(self.num_envs,device=self.device)
        self.r2_last_actor=torch.zeros(self.num_envs,33,device=self.device)
        self.r2_last_action=torch.zeros(self.num_envs,6,device=self.device)

    def _get_observations(self):
        observations=super()._get_observations()
        observations['training_control_mask']=torch.ones(self.num_envs,1,device=self.device)
        observations['training_bc_target']=torch.zeros(self.num_envs,6,device=self.device)
        observations['training_bc_weight']=torch.zeros(self.num_envs,1,device=self.device)
        return observations

    def _setup_scene(self):
        super()._setup_scene()
        self.contact.cfg.filter_prim_paths_expr = [f'/World/envs/env_.*/Wall{i}' for i in range(4)]

    def _pre_physics_step(self, actions):
        obs = self._get_observations()
        self.step_preparing = self.preparing.clone()
        self.preparation_steps += self.preparing.long()
        assert self.recovery_policy is not None, 'Frozen recovery skill must be attached before stepping'
        before = self.supervisor.exits.clone()
        before_realign=self.supervisor.realignments.clone()
        just_exited=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        with torch.no_grad():
            if self.recovery_enabled:
                measured = obs['policy'].clone()
                measured[self.preparing,24:27] = 0. # No switch while the external contact preparer controls.
                if self.cfg.initial_prealign:
                    valid=measured[:,24:27].abs().amax(-1)<=2.
                    free=valid&(measured[:,24:27].norm(dim=-1)/.02<5.)
                    outside=(measured[:,2]/20.>=.003)&(measured[:,2]/20.<=.035)
                    select=(self.supervisor.mode==0)&(self.supervisor.exits==0)&(self.supervisor.realignments==0)&free&outside&~self.preparing
                    self.supervisor.mode[select]=2
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
            if self.deep_recovery_policy is not None:
                expert=self.deep_recovery_policy(TensorDict(obs,batch_size=[self.num_envs]))
                x=obs['policy'];retreat=self.supervisor.mode==1
                observed_ood=(self.supervisor.attempts>=2)|(-x[:,2]/20>.020)|(x[:,3:6].norm(dim=-1)>.08)
                self.deep_recovery_latch=(self.deep_recovery_latch|observed_ood)&retreat
                select=self.deep_recovery_latch&self.recovery_enabled&~self.preparing
                recovery=torch.where(select[:,None],expert,recovery)
                self.deep_recovery_control_steps+=select.float()
            alignment=None
            if self.alignment_policy is not None:
                if just_exited.any():self.alignment_model.reset(just_exited)
                alignment=self.alignment_policy(TensorDict(obs,batch_size=[self.num_envs])).clone()
                # Collector captured the first zero-action handover physics
                # step. Start the frozen aligner's cold episode on the next.
                alignment[just_exited]=0.
                if just_exited.any():self.alignment_model.reset(just_exited)
            if self.near_alignment_policy is not None:
                assert alignment is not None
                if just_exited.any():self.near_alignment_model.reset(just_exited)
                expert=self.near_alignment_policy(TensorDict(obs,batch_size=[self.num_envs]));assert (expert[:,2]==0).all()
                x=obs['policy'];f=x[:,24:27]
                close=(x[:,2]/20>=.003)&(x[:,2]/20<=.050)&(x[:,:2].norm(dim=-1)/20<.002)&(x[:,3:6].norm(dim=-1)<.04)&(f.abs().amax(-1)<=2)&(f.norm(dim=-1)/.02<8)
                selected=(self.supervisor.mode==2)&~self.preparing&~just_exited&close
                alignment=torch.where(selected[:,None],expert,alignment)
                self.near_alignment_control_steps+=selected.float()
                if just_exited.any():self.near_alignment_model.reset(just_exited)
        self.stage_tracking_mask=(self.supervisor.mode==2)&~self.preparing&(self.alignment_policy is not None)
        initial=None
        if self.initial_policy is not None:
            reset_initial=(self.supervisor.realignments>before_realign)&(self.supervisor.exits==0)
            if self.cfg.initial_prealign:reset_initial|=self.episode_controller_steps==1
            if reset_initial.any():self.initial_model.reset(reset_initial)
            initial=self.initial_policy(TensorDict(obs,batch_size=[self.num_envs]))
        initial_select=(self.supervisor.mode==0)&(self.supervisor.exits==0)&~self.preparing
        if self.late_initial_policy is not None:
            assert initial is not None and not self.cfg.initial_prealign
            # Same observed history, frozen GRU/body, and unchanged Z/rotation
            # decoder: the specialist is a neural XY head for this late domain.
            late=self.late_initial_policy(TensorDict(obs,batch_size=[self.num_envs]))
            unchanged_from=3 if hasattr(self.late_initial_model,'position_head') else 2
            assert float((late[:,unchanged_from:]-initial[:,unchanged_from:]).abs().max())<=1e-6,'Late specialist frozen output drift'
            x=obs['policy'];depth=-x[:,2]/20
            late_select=initial_select&(depth>=.020)&(depth<=.050)&(x[:,3:6].norm(dim=-1)<.04)&(x[:,24:27].abs().amax(-1)<=2)&(x[:,24:27].norm(dim=-1)/.02<8)
            initial=torch.where(late_select[:,None],late,initial)
            self.late_initial_control_steps+=late_select.float()
        if initial is not None:actions=torch.where(initial_select[:,None],initial,actions)
        self.training_student_control=(self.supervisor.mode==0)&(self.supervisor.exits>0)&~self.preparing if initial is not None else (self.supervisor.mode==0)&~self.preparing
        self.failed_retry=enter&(self.supervisor.exits>=1)
        mirrored=None
        mirror_select=self.training_student_control&(self.case_ids==2)
        if self.mirror_policy is not None:
            from policy_mirror import observations as reflect_obs,actions as reflect_action
            with torch.no_grad():mirrored=reflect_action(self.mirror_policy(reflect_obs(TensorDict(obs,batch_size=[self.num_envs])))).clone()
            if self.training_teacher_policy is None:actions=torch.where(mirror_select[:,None],mirrored,actions)
        if self.training_teacher_policy is not None:
            with torch.no_grad():plain=self.training_teacher_policy(TensorDict(obs,batch_size=[self.num_envs])).clone()
            assert mirrored is not None
            self.training_teacher_target=torch.where((self.case_ids==2)[:,None],mirrored,plain).clamp(-1,1)
            mix=self.training_teacher_mix
            actions=torch.where(self.training_student_control[:,None],(1-mix)*actions+mix*self.training_teacher_target,actions)



        metrics_actions=getattr(self,'unclipped_policy_actions',actions)
        if initial is not None:metrics_actions=torch.where(initial_select[:,None],initial,metrics_actions)
        if mirrored is not None and self.training_teacher_policy is None:metrics_actions=torch.where(mirror_select[:,None],mirrored,metrics_actions)
        if self.training_teacher_policy is not None:metrics_actions=torch.where(self.training_student_control[:,None],actions,metrics_actions)
        controlled=(self.supervisor.mode==0)&~self.preparing
        self.main_control_steps+=controlled.float()
        self.main_z_sum+=torch.where(controlled,metrics_actions[:,2],0.)
        self.main_z_clipped+=(controlled&(metrics_actions[:,2].abs()>1.)).float()
        self.main_xy_clipped+=(controlled&(metrics_actions[:,:2].abs().amax(-1)>1.)).float()
        self.main_rot_clipped+=(controlled&(metrics_actions[:,3:].abs().amax(-1)>1.)).float()
        self.training_control_mode=self.supervisor.mode.clone()
        self.training_control_preparing=self.preparing.clone()
        self.training_control_offset=self.training_episode_offset.clone()
        applied = actions.clone()
        if self.recovery_enabled:
            applied[self.supervisor.mode == 1] = recovery[self.supervisor.mode == 1]
            if alignment is not None:applied[self.supervisor.mode==2]=alignment[self.supervisor.mode==2]
            applied[self.supervisor.mode == 2,2] = 0.
        applied[self.preparing] = 0.
        applied[self.preparing,2] = -.1
        if hasattr(self,'boundary_controller'):applied=self.boundary_controller.compose(self,applied,just_exited)
        self.r2_last_actor.copy_(obs['policy'])
        self.r2_last_action.copy_(applied)
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
        if self.cfg.train_stop_failed_retry:
            # Stricter TRAINING termination: a new failed retry after alignment.
            # Never claims success, never used by full-chain evaluation.
            early=self.failed_retry&~terminated
            self.end_reason[early]=5
            terminated|=early
        return terminated, timeout | prep_timeout

    def _get_rewards(self):
        reward = super()._get_rewards()
        masked = torch.where(self.step_preparing,torch.zeros_like(reward),reward)
        if self.cfg.training_joint_margin_cost>0:
            # Public encoder joint5 is already in actor33; training cost only.
            pitch=self.robot.data.joint_pos[:,self.arm_ids[4]].abs()
            risk=((pitch-.65)/.511).clamp(0,2).square()
            masked=masked-self.cfg.training_joint_margin_cost*risk*self.training_student_control.float()
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
                ('preparation_steps',self.preparation_steps),('handover_force_n',self.handover_force),
                ('near_alignment_control_steps',self.near_alignment_control_steps),('deep_recovery_control_steps',self.deep_recovery_control_steps),('late_initial_control_steps',self.late_initial_control_steps),('main_control_steps',self.main_control_steps),
                ('main_z_action_mean',self.main_z_sum/self.main_control_steps.clamp_min(1)),
                ('main_z_clipped_fraction',self.main_z_clipped/self.main_control_steps.clamp_min(1)),
                ('main_xy_clipped_fraction',self.main_xy_clipped/self.main_control_steps.clamp_min(1)),
                ('main_rot_clipped_fraction',self.main_rot_clipped/self.main_control_steps.clamp_min(1))]}
        if initialized:
            rel,ori,force=self._geometry()
            r2_truth=torch.cat((rel,ori,force),dim=-1)[ids].clone()
            r2_actor=self.r2_last_actor[ids].clone()
            r2_action=self.r2_last_action[ids].clone()
        super()._reset_idx(ids)
        if initialized:
            self.extras['terminal'].update(extra)
            terminal=self.extras['terminal']
            rows=[]
            for j,i in enumerate(ids.cpu().tolist()):
                if int(terminal['end_reason'][j])!=4:
                    row=dict(env=i,actor33=r2_actor[j].cpu().tolist(),action=r2_action[j].cpu().tolist(),true_relative_orientation_force=r2_truth[j].cpu().tolist())
                    for k,v in terminal.items():
                        if isinstance(v,torch.Tensor) and v.ndim==1 and len(v)==len(ids):row[k]=float(v[j])
                    rows.append(row)
            if rows:
                with Path(self.cfg.log_dir,'all_failure_terminal_actor.jsonl').open('a') as f:
                    for row in rows:f.write(json.dumps(row)+'\n')

            self.supervisor.reset(ids)
            self.near_alignment_control_steps[ids]=0
            self.deep_recovery_latch[ids]=False
            self.deep_recovery_control_steps[ids]=0
            self.late_initial_control_steps[ids]=0
            self.true_exit_count[ids] = 0
            self.true_released_exit_count[ids] = 0
            self.true_trigger_depth[ids] = 0.
            self.trigger_wall_force[ids] = 0.
            self.preparing[ids] = self.fixed_mask[ids]
            self.preparation_steps[ids] = 0
            self.handover_force[ids] = 0.
            for value in [self.failed_retry,self.main_control_steps,self.main_z_sum,self.main_z_clipped,self.main_xy_clipped,self.main_rot_clipped]:value[ids]=0
            if self.recovery_model is not None:
                mask = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device);mask[ids] = True
                self.recovery_model.reset(mask)
                if self.alignment_model is not None:self.alignment_model.reset(mask)
                if self.initial_model is not None:self.initial_model.reset(mask)
                if self.near_alignment_model is not None:self.near_alignment_model.reset(mask)
                if self.deep_recovery_model is not None:self.deep_recovery_model.reset(mask)
                if self.late_initial_model is not None:self.late_initial_model.reset(mask)
                if self.mirror_model is not None:self.mirror_model.reset(mask)
                if self.training_teacher_model is not None:self.training_teacher_model.reset(mask)
        if self.cfg.fixed_contact_cases:
            ids = ids[self.fixed_mask[ids]] if initialized else ids
            if not len(ids):return
            directions = torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]],device=self.device)
            case = torch.full_like(ids,3) if self.cfg.negative_lateral_case_only else (ids % 2 + 2 if self.cfg.lateral_cases_only else ids % 4)
            if self.cfg.coverage_lateral_focus: case = coverage_case_ids(ids)
            if self.cfg.recovery_offset_min>=0:
                scales=torch.empty(len(ids),1,device=self.device).uniform_(self.cfg.recovery_offset_min,self.cfg.recovery_offset)
            else:scales=torch.full((len(ids),1),self.cfg.recovery_offset,device=self.device)
            offset = directions[case] * scales
            if initialized:self.training_episode_offset[ids]=scales.squeeze(-1)
            offset[:,0] += self.cfg.recovery_center_x
            if self.cfg.replay_original_axial_cases:
                axial=case<2
                offset[axial]=directions[case[axial]]*.005
                offset[axial,0]-=.0045
            if self.cfg.replay_original_positive_lateral:
                positive=case==2
                offset[positive]=directions[case[positive]]*.005
                offset[positive,0]-=.0045
                if initialized:self.training_episode_offset[ids[positive]]=.005

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
    unavailable=SkillSupervisor(4,'cpu');u=torch.zeros(4,33);u[:,24:27]=3.
    unavailable.update(u);unavailable.update(u)
    assert (unavailable.mode==0).all() and (unavailable.attempts==0).all() and (unavailable.force_ema==0).all()
    u[:,24:27]=0.;u[:,24]=.4
    unavailable.update(u);assert unavailable.update(u).all()
    print('UNAVAILABLE_FORCE_SUPERVISOR_PASSED',flush=True)


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
