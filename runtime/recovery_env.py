"""Stage3B: physical contact preparation followed by learned withdrawal only.

The preparer is part of the environment. It never retreats, teleports the peg
into contact, resets the accumulated command at handover, or advances global
physics inside an asynchronous reset. Success here is NOT insertion success.
"""
import json
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply
from cad_env import ChallengeCupEnv, ChallengeCupCfg
from isaaclab.utils import configclass
from contact_reward import contact_cost


@configclass
class ContactRecoveryCfg(ChallengeCupCfg):
    episode_length_s = 8.
    recovery_offset = .004
    recovery_center_x = -.0045
    task_profile = 'contact_withdrawal_only'
    success_profile = 'withdrawal3mm_release2N_hold8_after_contact5N_v1'


class ContactRecoveryEnv(ChallengeCupEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.preparing = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_preparing = self.preparing.clone()
        self.preparation_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.handover_force = torch.zeros(self.num_envs, device=self.device)
        self.handover_depth = torch.zeros_like(self.handover_force)
        self.handover_wall_force = torch.zeros_like(self.handover_force)
        self.previous_withdrawal_potential = torch.zeros_like(self.handover_force)
        self.case_ids = torch.arange(self.num_envs, device=self.device) % 4

    def _setup_scene(self):
        super()._setup_scene()
        # Diagnostic only: actor retains the same whole-assembly net force.
        self.contact.cfg.filter_prim_paths_expr = [f'/World/envs/env_.*/Wall{i}' for i in range(4)]

    def wall_contact_force(self):
        history = self.contact.data.force_matrix_w_history
        return history.norm(dim=-1).sum(dim=-1).amax(dim=1)[:,0]

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        initialized = hasattr(self, 'preparing')
        if initialized:
            extra = {name: tensor[env_ids].clone() for name, tensor in [
                ('recovery_case', self.case_ids), ('preparation_steps', self.preparation_steps),
                ('handover_force_n', self.handover_force), ('handover_depth_m', self.handover_depth),
                ('handover_wall_force_n', self.handover_wall_force)]}
        super()._reset_idx(env_ids)
        if initialized:
            self.extras['terminal'].update(extra)
            self.preparing[env_ids] = True
            self.preparation_steps[env_ids] = 0
            self.handover_force[env_ids] = 0.
            self.handover_depth[env_ids] = 0.
            self.handover_wall_force[env_ids] = 0.
        # Same legal joint pose and four fixed rim offsets on every reset.
        directions = torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]], device=self.device)
        cases = env_ids % 4
        offset = directions[cases] * self.cfg.recovery_offset
        # Fixed fixture placement around the observed unloaded approach path.
        # v2 CPU/GPU: hole-X=-5mm gave true XY<0.5mm with zero contact.
        offset[:,0] += self.cfg.recovery_center_x
        self.hole_pos[env_ids] = torch.tensor(self.geometry['hole_center'], device=self.device)
        self.hole_pos[env_ids] += quat_apply(self.hole_base_quat.expand(len(env_ids), -1), offset)
        for wall, offset in zip(self.walls, self.wall_offsets):
            shifted = quat_apply(self.hole_quat[env_ids], torch.tensor(offset, device=self.device).expand(len(env_ids), -1))
            wall.write_root_pose_to_sim(torch.cat((self.hole_pos[env_ids] + self.scene.env_origins[env_ids] + shifted,
                                                   self.hole_quat[env_ids]), -1), env_ids=env_ids)

    def _pre_physics_step(self, actions):
        self.step_preparing = self.preparing.clone()
        relative, _, _ = self._geometry()
        withdrawal = (self.handover_depth + relative[:,2]).clamp(0., .003)
        self.previous_withdrawal_potential = 20. * withdrawal / .003
        applied = actions.clone()
        applied[self.preparing] = 0.
        applied[self.preparing,2] = -.1  # 0.2mm commanded approach per policy step
        self.preparation_steps += self.preparing.long()
        super()._pre_physics_step(applied)

    def _get_dones(self):
        old_hold = self.hold.clone()
        old_success = self.ever_success.clone()
        super()._get_dones()  # Shared physical failure and contact diagnostics.
        relative, _, force = self._geometry()
        norm = force.norm(dim=-1)
        wall_force = self.wall_contact_force()
        handover = self.preparing & (norm >= 5.) & (wall_force >= 5.)
        self.handover_force[handover] = norm[handover]
        self.handover_wall_force[handover] = wall_force[handover]
        self.handover_depth[handover] = -relative[handover,2]
        self.preparing[handover] = False
        withdrawal = self.handover_depth + relative[:,2]
        valid = ~self.step_preparing & (self.handover_force >= 5.)
        valid &= (withdrawal >= .003) & (norm < 2.) & self.retreat_after_contact
        self.hold = torch.where(valid, old_hold + 1, torch.zeros_like(old_hold))
        success = self.hold >= 8
        self.ever_success = old_success | success
        self.end_hold_steps.copy_(self.hold.float())
        failure = (norm > 100.) | (relative[:,:2].norm(dim=-1) > .12)
        failure |= (relative[:,2] < -.06) | (relative[:,2] > .15)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        timeout |= self.preparing & (self.preparation_steps >= 240)
        self.end_reason[:] = 0
        self.end_reason[timeout] = 3
        self.end_reason[failure] = 2
        self.end_reason[norm > 100.] = 1
        self.end_reason[success] = 4
        return success | failure, timeout

    def _get_rewards(self):
        relative, _, force = self._geometry()
        withdrawal = (self.handover_depth + relative[:,2]).clamp(0., .003)
        potential = 20. * withdrawal / .003
        potential = torch.where(self.reset_buf, 0., potential)
        reward = .99 * potential - self.previous_withdrawal_potential - .02
        reward -= .002 * self.actions.square().sum(-1)
        reward -= contact_cost(force.norm(dim=-1), self.cfg.reward_profile)
        reward += 100. * (self.hold >= 8).float()
        reward -= 10. * (self.reset_terminated & (self.hold < 8)).float()
        reward -= 5. * self.reset_time_outs.float()
        # Actor actions were ignored during preparation: no immediate credit.
        reward = torch.where(self.step_preparing, torch.zeros_like(reward), reward)
        self.episode_returns += reward
        self.extras['log'] = {'recovery/preparing':self.preparing.float().mean(),
                             'recovery/withdrawal_m':withdrawal.mean(),
                             'recovery/contact_force_n':force.norm(dim=-1).mean(),
                             'recovery/success_step':(self.hold >= 8).float().mean()}
        return reward


def recovery_smoke(env, raw, logdir):
    """Verify actual contact, handover and outward control, separately on CPU/GPU."""
    raw.verify_cad_scene()
    records = []
    for direction in [0., .5]:
        env.reset()
        completed = set()
        for _ in range(500):
            action = torch.zeros(env.num_envs, 6, device=env.device)
            action[:,2] = direction
            obs, reward, dones, info = env.step(action)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            if dones.any():
                terminal = info['terminal']
                for j, index in enumerate(terminal['env_ids'].tolist()):
                    if index in completed:
                        continue
                    completed.add(index)
                    records.append({key:terminal[key][j].item() for key in [
                        'recovery_case','success','peak_force_n','handover_force_n','preparation_steps',
                        'retreat_after_contact','release_after_contact','end_reason', 'handover_wall_force_n',
                        'true_depth_m','xy_error_m','orientation_error_rad','command_lead_m']}
                        | {'probe_outward_action':direction, 'env_id':index})
            if len(completed) == env.num_envs:
                break
        assert len(completed) == env.num_envs, records
    outward = [r for r in records if r['probe_outward_action'] > 0.]
    stationary = [r for r in records if r['probe_outward_action'] == 0.]
    (Path(logdir)/'probe_records.json').write_text(json.dumps(records,indent=2))
    assert all(5. <= r['handover_force_n'] < 40. and r['handover_wall_force_n']>=5. for r in records), records
    assert all(r['success'] and r['retreat_after_contact'] and r['release_after_contact'] for r in outward), records
    assert not any(r['success'] for r in stationary), records
    assert max(r['peak_force_n'] for r in outward) < 100., records
    result = {'status':'passed','task':'contact_withdrawal_only','simulation_device':raw.device,
              'preparation':'fixed four rim directions; physical slow approach; no scripted retreat',
              'fixture_center_shift_x_m':raw.cfg.recovery_center_x,
              'success':'>=3mm actual withdrawal, outward command, <2N for8steps after >=5N contact',
              'uploaded_model_source_sha256':raw.geometry['source_sha256'], 'records':records}
    (Path(logdir)/'smoke.json').write_text(json.dumps(result,indent=2))
    print('RECOVERY_PHYSICAL_SMOKE_PASSED', json.dumps(result), flush=True)
