"""Convert Piper, check contact simulation, train PPO, and evaluate held-out resets."""
import argparse
import json
from pathlib import Path
import time
import traceback
import os
import threading
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['smoke', 'train', 'eval','collect_alignment'], default='smoke')
parser.add_argument('--history_proof',choices=['root','bridge'])
parser.add_argument('--independent_boundary_history',action='store_true')
parser.add_argument('--fixed_final_retry_brake',action='store_true')
parser.add_argument('--joint5_input',action='store_true')
parser.add_argument('--critic_phase_pilot',action='store_true')
parser.add_argument('--critic_domain_collect',action='store_true')
parser.add_argument('--critic_history',action='store_true')
parser.add_argument('--final_retry_release',action='store_true')
parser.add_argument('--critic_conditioning',choices=['off','on'],default='on')
parser.add_argument('--critic_equivalence',action='store_true')
parser.add_argument('--conditioned_critic_checkpoint')
parser.add_argument('--deep_ppo_updates',type=int,default=0)
parser.add_argument('--withdrawal_probe',action='store_true')
parser.add_argument('--withdrawal_detail',action='store_true')
parser.add_argument('--paired_diagnostic',action='store_true')
parser.add_argument('--pair_scale',type=float,default=1.)
parser.add_argument('--critic_shadow',action='store_true')
parser.add_argument('--post_ppo_updates',type=int,default=0)
parser.add_argument('--recurrent_alignment_updates',type=int,default=0)
parser.add_argument('--initial_case_replay')
parser.add_argument('--alignment_axial_cap',action='store_true')
parser.add_argument('--handoff_trace',action='store_true')
parser.add_argument('--handoff_ppo_updates',type=int,default=0)
parser.add_argument('--initial_trace',action='store_true')
parser.add_argument('--initial_ppo_updates',type=int,default=0)
parser.add_argument('--boundary_mode',choices=['zero','train','eval'],default=None)
parser.add_argument('--boundary_checkpoint',default=None)
parser.add_argument('--block_force_filter',choices=['instant','ema'],default='instant')
parser.add_argument('--boundary_updates',type=int,default=6)
parser.add_argument('--scene_profile',choices=['simplified','challenge_cup'],default='simplified')
parser.add_argument('--task_profile',choices=['insertion','contact_withdrawal_only','insertion_with_recovery','post_exit_alignment'],default='insertion')
parser.add_argument('--alignment_bank',action='append',default=[])
parser.add_argument('--alignment_reward_scale_mm',type=float,default=5.)
parser.add_argument('--alignment_angle_weight',type=float,default=2.)
parser.add_argument('--alignment_insertion_depth_mm',type=float,default=0.)
parser.add_argument('--alignment_checkpoint',default=None)
parser.add_argument('--initial_insertion_checkpoint',default=None)
parser.add_argument('--late_initial_checkpoint',default=None)
parser.add_argument('--deep_recovery_checkpoint',default=None)
parser.add_argument('--near_alignment_checkpoint',default=None)
parser.add_argument('--initial_prealign',action='store_true')
parser.add_argument('--training_bc',action='store_true')
parser.add_argument('--mirror_teacher_probe',action='store_true')
parser.add_argument('--mirror_dagger_fit',action='store_true')
parser.add_argument('--mirror_teacher_checkpoint',default=None)
parser.add_argument('--training_joint_margin_cost',type=float,default=0.)
parser.add_argument('--training_bc_lateral_checkpoint',default='specialist_stage_20261015_bc1_train/model_399.pt')
parser.add_argument('--training_bc_trusted_offset_mm',type=float,default=1.625)
parser.add_argument('--training_bc_trust_case2',action='store_true')
parser.add_argument('--evaluate_without_recovery',action='store_true')
parser.add_argument('--skip_straight_baseline',action='store_true')
parser.add_argument('--recovery_checkpoint',default=None)
parser.add_argument('--recovery_memory',choices=['shadow','cold'],default='shadow')
parser.add_argument('--fixed_contact_cases',action='store_true')
parser.add_argument('--precontact_fraction',type=float,default=1.)
parser.add_argument('--recovery_offset_mm',type=float,default=4.)
parser.add_argument('--recovery_center_x_mm',type=float,default=-4.5)
parser.add_argument('--recovery_offset_min_mm',type=float,default=None)
parser.add_argument('--case_profile',choices=['all','lateral','replay','negative_curriculum','negative_lateral','lateral_focus'],default='all')
parser.add_argument('--cad_asset_dir',default=None)
parser.add_argument('--adapt_cad_normalization',action='store_true')
parser.add_argument('--init_axial_bias',type=float,default=None)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--run_name', default='ppo_initial')
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--eval_episodes', type=int, default=256)
parser.add_argument('--hole_width_mm', type=float, default=16.)
parser.add_argument('--hole_xy_noise_mm', type=float, default=4.)
parser.add_argument('--joint_noise_rad',type=float,default=.01)
parser.add_argument('--hole_yaw_noise_rad',type=float,default=.04)
parser.add_argument('--observation_bias_mm', type=float, default=0.)
parser.add_argument('--action_delay_min', type=int, default=0)
parser.add_argument('--action_delay_max', type=int, default=0)
parser.add_argument('--friction_scale_min', type=float, default=1.)
parser.add_argument('--friction_scale_max', type=float, default=1.)
parser.add_argument('--force_noise_std_n', type=float, default=0.)
parser.add_argument('--force_bias_bound_n', type=float, default=0.)
parser.add_argument('--stiffness_scale_min', type=float, default=1.)
parser.add_argument('--stiffness_scale_max', type=float, default=1.)
parser.add_argument('--peg_mass_min_g', type=float, default=20.)
parser.add_argument('--peg_mass_max_g', type=float, default=20.)
parser.add_argument('--grip_axial_min_mm', type=float, default=0.)
parser.add_argument('--grip_axial_max_mm', type=float, default=0.)
parser.add_argument('--grip_endpoint_fraction', type=float, default=0.)
parser.add_argument('--grip_positive_endpoint_probability',type=float,default=.5)
parser.add_argument('--nominal_replay_fraction',type=float,default=0.)
parser.add_argument('--nominal_dynamics_fraction',type=float,default=0.)
parser.add_argument('--command_position_bound_mm',type=float,default=0.)
parser.add_argument('--command_rotation_bound_rad',type=float,default=0.)
parser.add_argument('--command_axial_lead_bound_mm',type=float,default=0.)
parser.add_argument('--near_contact_position_step_mm',type=float,default=0.)
parser.add_argument('--near_contact_rotation_step_rad',type=float,default=0.)
parser.add_argument('--near_contact_band_mm',type=float,default=30.)
parser.add_argument('--reward_profile',choices=['potential_v2','contact_v9','alignment_v11','contact_v12'],default='potential_v2')
parser.add_argument('--entropy_coef',type=float,default=.005)
parser.add_argument('--warm_start', action='store_true', help='Transfer weights; reset optimizer and update count')
parser.add_argument('--exploration_std', type=float, default=None)
parser.add_argument('--rebuild_robot_asset', action='store_true')
parser.add_argument('--compare_checkpoint', default=None)
parser.add_argument('--recurrent', action='store_true')
parser.add_argument('--privileged_critic', action='store_true')
parser.add_argument('--reference_checkpoint', action='append', default=[])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0<=args.entropy_coef<=.05:
    parser.error('Entropy coefficient must be within 0..0.05')
launcher = AppLauncher(args)
app = launcher.app
print('SIMULATOR_APP_READY', flush=True)

import torch
from rsl_rl.runners import OnPolicyRunner
from current_alignment import CurrentObsAlignmentActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['CurrentObsAlignmentActorCriticRecurrent']=CurrentObsAlignmentActorCriticRecurrent
from current_recovery import CurrentObsRecoveryActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['CurrentObsRecoveryActorCriticRecurrent']=CurrentObsRecoveryActorCriticRecurrent
from risk_guided import RiskGuidedActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['RiskGuidedActorCriticRecurrent']=RiskGuidedActorCriticRecurrent
from causal_episodic import CausalEpisodicActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['CausalEpisodicActorCriticRecurrent']=CausalEpisodicActorCriticRecurrent
from post_smooth_residual import PostSmoothResidualActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['PostSmoothResidualActorCriticRecurrent']=PostSmoothResidualActorCriticRecurrent
from post_residual import PostResidualActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['PostResidualActorCriticRecurrent']=PostResidualActorCriticRecurrent
from gated_post_xyz import GatedPostXYZActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['GatedPostXYZActorCriticRecurrent']=GatedPostXYZActorCriticRecurrent
from late_xyz2 import LateXYZBandActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['LateXYZBandActorCriticRecurrent']=LateXYZBandActorCriticRecurrent
from late_xyz import LateXYZActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['LateXYZActorCriticRecurrent']=LateXYZActorCriticRecurrent
from late_xy_adapter import LateXYActorCriticRecurrent
OnPolicyRunner._construct_algorithm.__globals__['LateXYActorCriticRecurrent']=LateXYActorCriticRecurrent
from contact_reward import alignment_length_scale,orientation_length_scale
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,
)

ROOT = Path("/workspace/piper-insertion")
STATUS_PATH = ROOT/'logs'/args.run_name/'status.json'
STATUS_PATH.parent.mkdir(parents=True,exist_ok=True)
STATUS_PATH.write_text(json.dumps(dict(status='running',mode=args.mode,seed=args.seed,
    num_envs=args.num_envs,iterations=args.iterations,start_time=time.time()),indent=2))

class InstalledImporterConverter(UrdfConverter):
    """Support the installed 2.4.19 importer without changing shared Isaac Lab."""
    def _get_urdf_import_config(self):
        import omni.kit.commands
        _, config = omni.kit.commands.execute('URDFCreateImportConfig')
        config.set_distance_scale(1.)
        config.set_make_default_prim(True)
        config.set_create_physics_scene(False)
        config.set_density(self.cfg.link_density)
        config.set_convex_decomp(self.cfg.collider_type == 'convex_decomposition')
        config.set_collision_from_visuals(self.cfg.collision_from_visuals)
        config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)
        if hasattr(config, 'set_merge_fixed_ignore_inertia'):
            config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)
        config.set_fix_base(self.cfg.fix_base)
        config.set_self_collision(self.cfg.self_collision)
        config.set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)
        config.set_replace_cylinders_with_capsules(self.cfg.replace_cylinders_with_capsules)
        return config

@configclass
class AgentCfg(RslRlOnPolicyRunnerCfg):
    seed = args.seed
    device = 'cuda:0'
    num_steps_per_env = 32
    max_iterations = args.iterations
    save_interval = 50
    experiment_name = 'piper_insertion'
    obs_groups = {'policy': ['policy'], 'critic': ['policy']}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=.6, actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[128,128,64], critic_hidden_dims=[128,128,64], activation='elu')
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2,
        entropy_coef=args.entropy_coef, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3e-4, schedule='adaptive', gamma=.99, lam=.95,
        desired_kl=.01, max_grad_norm=1.)

@torch.inference_mode()
def evaluate(env, policy=None, label='policy', episodes=256):
    if args.task_profile=='post_exit_alignment':env.unwrapped.bank_cursor.zero_()
    obs, _ = env.reset()
    policy_model=getattr(policy,'__self__',None)
    if policy_model is not None:
        policy_model.reset(torch.ones(env.num_envs,dtype=torch.bool,device=env.device))
    successes, peaks, lengths, returns, delays = [], [], [], [], []
    reasons, friction_scales = [], []
    stiffness_scales = []
    peg_masses, grip_offsets = [], []
    contact_seen,retreats,releases=[],[],[]
    diagnostic_keys=['true_depth_m','nominal_depth_m','command_lead_m','peak_command_lead_m','orientation_error_rad',
                     'xy_error_m','shaft_fits','contact_force_n','hold_steps','down_step_m']
    if args.task_profile=='contact_withdrawal_only':
        diagnostic_keys += ['recovery_case','preparation_steps','handover_force_n','handover_depth_m','handover_wall_force_n']
    if args.task_profile=='insertion_with_recovery':
        diagnostic_keys += ['recovery_case','recovery_attempts','recovery_exits','recovery_realignments',
                           'recovery_end_mode','recovery_retreat_steps','recovery_realign_steps',
                           'recovery_true_exits','recovery_true_released_exits','recovery_trigger_wall_force_n','preparation_steps','handover_force_n']
    if args.late_initial_checkpoint:diagnostic_keys+=['late_initial_control_steps']
    if args.deep_recovery_checkpoint:diagnostic_keys+=['deep_recovery_control_steps']
    if args.near_alignment_checkpoint:diagnostic_keys+=['near_alignment_control_steps']
    if args.task_profile=='insertion_with_recovery':diagnostic_keys+=['main_control_steps','main_z_action_mean','main_z_clipped_fraction','main_xy_clipped_fraction','main_rot_clipped_fraction']
    if args.task_profile=='post_exit_alignment':diagnostic_keys+=['alignment_bank_index','alignment_case','alignment_projected_fit']
    diagnostics={key:[] for key in diagnostic_keys}
    cap_fractions=[]
    axial_cap_fractions=[]
    # Assign an equal quota to each environment so short successful episodes
    # cannot be overrepresented while longer failed episodes are still running.
    remaining = torch.full((env.num_envs,), episodes//env.num_envs,
                           dtype=torch.long, device=env.device)
    remaining[:episodes%env.num_envs] += 1
    started = time.time()
    for _ in range(10000):
        with torch.inference_mode():
            if policy is not None:
                actions = policy(obs)
            else:
                # Fixed world-frame straight insertion; no lateral/orientation feedback.
                actions = torch.zeros(env.num_envs,6,device=env.device)
                actions[:,2] = -.35
            env.unwrapped.unclipped_policy_actions=actions.detach().clone()
            obs, reward, dones, info = env.step(actions)
            if policy_model is not None:
                policy_model.reset(dones.bool())
            count = int(dones.sum())
            if count:
                terminal = info['terminal']
                ids = terminal['env_ids']
                if not torch.equal(ids, dones.nonzero().flatten()):
                    raise RuntimeError('Terminal records do not match finished environments')
                accepted = remaining[ids] > 0
                selected = ids[accepted]
                remaining[selected] -= 1
                successes.extend(terminal['success'][accepted].cpu().int().tolist())
                peaks.extend(terminal['peak_force_n'][accepted].cpu().tolist())
                returns.extend(terminal['return'][accepted].cpu().tolist())
                lengths.extend(terminal['seconds'][accepted].cpu().tolist())
                delays.extend(terminal['delay_steps'][accepted].cpu().tolist())
                reasons.extend(terminal['end_reason'][accepted].cpu().tolist())
                friction_scales.extend(terminal['friction_scale'][accepted].cpu().tolist())
                stiffness_scales.extend(terminal['stiffness_scale'][accepted].cpu().tolist())
                peg_masses.extend(terminal['peg_mass_kg'][accepted].cpu().tolist())
                grip_offsets.extend(terminal['grip_axial_m'][accepted].cpu().tolist())
                contact_seen.extend(terminal['contact_seen'][accepted].cpu().tolist())
                retreats.extend(terminal['retreat_after_contact'][accepted].cpu().tolist())
                releases.extend(terminal['release_after_contact'][accepted].cpu().tolist())
                for key in diagnostic_keys:
                    diagnostics[key].extend(terminal[key][accepted].cpu().tolist())
                cap_fractions.extend(terminal['command_cap_fraction'][accepted].cpu().tolist())
                axial_cap_fractions.extend(terminal['axial_command_cap_fraction'][accepted].cpu().tolist())
            if not remaining.any():
                break
    if len(successes) != episodes:
        raise RuntimeError(f'Only {len(successes)} of {episodes} evaluation episodes completed')
    result = dict(label=label, seed=args.seed+10000, episodes=len(successes),
        sampling='fixed_equal_quota_per_environment',
        success_rate=sum(successes)/len(successes), mean_peak_force_n=sum(peaks)/len(peaks),
        max_peak_force_n=max(peaks), mean_episode_seconds=sum(lengths)/len(lengths),
        mean_return=sum(returns)/len(returns), wall_seconds=time.time()-started)
    result['task_profile']=args.task_profile
    if args.task_profile=='post_exit_alignment':
        result['success_means']='actualXY<1mm angle<.05rad projected shaft fits, safely outside3..35mm, actual force<2N held8; NOT insertion'
        result['by_alignment_case']={str(c):dict(episodes=sum(v==c for v in diagnostics['alignment_case']),
            successes=sum(s for s,v in zip(successes,diagnostics['alignment_case']) if v==c)) for c in range(4)}
    if args.task_profile=='insertion_with_recovery':
        result['supervisor_enabled']=env.unwrapped.recovery_enabled
        result['recovery_memory']=args.recovery_memory
        result['by_recovery_case']={str(c):dict(episodes=sum(v==c for v in diagnostics['recovery_case']),successes=sum(s for s,v in zip(successes,diagnostics['recovery_case']) if v==c),main_actions={key:sum(a for a,v in zip(diagnostics[key],diagnostics['recovery_case']) if v==c)/max(1,sum(v==c for v in diagnostics['recovery_case'])) for key in ['main_control_steps','main_z_action_mean','main_z_clipped_fraction','main_xy_clipped_fraction','main_rot_clipped_fraction']}) for c in range(4)}
        result['training_retry_termination_active']=env.unwrapped.cfg.train_stop_failed_retry
        result['initial_insertion_checkpoint']=args.initial_insertion_checkpoint
        result['late_initial_checkpoint']=args.late_initial_checkpoint
        if args.late_initial_checkpoint:result['late_initial_specialist']={'episodes_used':sum(v>0 for v in diagnostics['late_initial_control_steps']),'control_steps':sum(diagnostics['late_initial_control_steps']),'actor_inputs':33,'action_outputs':6,'teacher':False,'domain':'initial mode only; observed20..50mm depth,angle<.04,valid measuredforce<8N','early_policy_exactly_retained':True}
        result['near_alignment_checkpoint']=args.near_alignment_checkpoint
        if args.near_alignment_checkpoint:result['near_alignment_specialist']={'episodes_used':sum(v>0 for v in diagnostics['near_alignment_control_steps']),'control_steps':sum(diagnostics['near_alignment_control_steps']),'actor_inputs':33,'action_outputs':6,'teacher':False,'current_actor':True,'domain':'Native align mode,observed outside3..50mm,XYnorm<2mm,angle<.04,valid measuredF<8N; native first handover zero and Z0 retained'}
        result['deep_recovery_checkpoint']=args.deep_recovery_checkpoint
        if args.deep_recovery_checkpoint:result['deep_recovery_specialist']={'episodes_used':sum(v>0 for v in diagnostics['deep_recovery_control_steps']),'control_steps':sum(diagnostics['deep_recovery_control_steps']),'actor_inputs':33,'action_outputs':6,'teacher':False,'domain':'withdrawal only; retry>=2 or observeddepth>20mm or observedangle>.08; latch until native exit','ordinary_recovery_exactly_retained':True,'current_observation_actor':True,'GRU_tracked_but_not_used_by_expert_actor':True}
        if args.late_initial_checkpoint:result['late_initial_specialist'].update(learned_position_axes=3 if isinstance(env.unwrapped.late_initial_model,LateXYZActorCriticRecurrent) else 2,original_rotation_outputs_unchanged=True)
        result['initial_prealign']=args.initial_prealign
        result['action_diagnostic_sampling']='policy output before wrapper clipping; frozen initial output before environment clipping'
        result['fixture_curriculum']={'center_x_mm':args.recovery_center_x_mm,'offset_mm':args.recovery_offset_mm,'cases':args.case_profile,'original_axial_replay':args.case_profile in ('replay','negative_curriculum'),'original_positive_lateral_replay':args.case_profile=='negative_curriculum','offset_min_mm':env.unwrapped.cfg.recovery_offset_min*1000 if env.unwrapped.cfg.recovery_offset_min>=0 else None}
        result['recovery_chain']={
            'prepared_contact_episodes':sum(x>=5. for x in diagnostics['handover_force_n']),
            'triggered_episodes':sum(x>0 for x in diagnostics['recovery_attempts']),
            'exit_episodes':sum(x>0 for x in diagnostics['recovery_exits']),
            'physical_exit_episodes':sum(x>0 for x in diagnostics['recovery_true_exits']),
            'physical_released_exit_episodes':sum(x>0 for x in diagnostics['recovery_true_released_exits']),
            'realigned_episodes':sum(x>0 for x in diagnostics['recovery_realignments']),
            'successes_after_trigger':sum(s for s,a in zip(successes,diagnostics['recovery_attempts']) if a>0),
            'end_in_retreat':sum(x==1 for x in diagnostics['recovery_end_mode']),
            'end_in_realign':sum(x==2 for x in diagnostics['recovery_end_mode']),
            'trigger_without_wall_marker':sum(a>0 and f<5 for a,f in zip(diagnostics['recovery_attempts'],diagnostics['recovery_trigger_wall_force_n']))}
    if args.task_profile=='contact_withdrawal_only':
        result['success_means']='withdrawal >=3mm, outward command, force<2N held8steps after physical5N contact; NOT insertion'
        result['preparation_contact_episodes']=sum(v>=5. for v in diagnostics['handover_force_n'])
        result['preparation_wall_contact_episodes']=sum(v>=5. for v in diagnostics['handover_wall_force_n'])
        result['by_recovery_case']={str(case):dict(episodes=sum(c==case for c in diagnostics['recovery_case']),
            successes=sum(s for s,c in zip(successes,diagnostics['recovery_case']) if c==case)) for case in range(4)}
    result['by_delay_steps'] = {
        str(delay): dict(episodes=delays.count(delay),
                        success_rate=sum(s for s,d in zip(successes,delays) if d==delay)/delays.count(delay))
        for delay in sorted(set(delays))}
    result['force_limit_exceedances'] = sum(p > 100. for p in peaks)
    if contact_seen!=[p>=5. for p in peaks]:raise RuntimeError('Contact flags disagree with true peak force')
    result['contact_diagnostics']={'threshold_n':5.,'sensor':'true net force on link6 assembly',
        'retreat_definition':'physical depth decrease >=2mm from first contact with outward command',
        'release_definition':'true contact force dropped below2N after a >=5N contact'}
    for name,mask in [('contacted',contact_seen),('without_contact',[not c for c in contact_seen])]:
        selected=[i for i,c in enumerate(mask) if c]
        result['contact_diagnostics'][name]={'episodes':len(selected),
            'successes':sum(successes[i] for i in selected),
            'success_rate':sum(successes[i] for i in selected)/len(selected) if selected else None,
            'retreat_episodes':sum(retreats[i] for i in selected),
            'released_episodes':sum(releases[i] for i in selected),
            'force_failures':sum(reasons[i]==1 for i in selected)}
    result['terminal_counts'] = {name: reasons.count(code) for code,name in
        [(1,'force_limit'),(2,'geometry_limit'),(3,'timeout'),(4,'success')]}
    result['by_friction_scale'] = {
        f'{scale:.6f}': dict(episodes=friction_scales.count(scale),
            success_rate=sum(s for s,f in zip(successes,friction_scales) if f==scale)/friction_scales.count(scale))
        for scale in sorted(set(friction_scales))}
    result['successful_mean_peak_force_n'] = (
        sum(p for p,s in zip(peaks,successes) if s)/sum(successes) if sum(successes) else None)
    # Broad bins have enough samples even though response scales are continuous.
    result['by_stiffness_scale'] = {}
    for low,high in [(0.,.85),(.85,1.15),(1.15,float('inf'))]:
        selected=[i for i,x in enumerate(stiffness_scales) if low<=x<high]
        if selected:
            result['by_stiffness_scale'][f'{low:.2f}-{high:.2f}'] = dict(
                episodes=len(selected),success_rate=sum(successes[i] for i in selected)/len(selected))
    for name,values,bounds in [('peg_mass_g',[x*1000 for x in peg_masses],[0.,30.,60.,float('inf')]),
                              ('grip_axial_mm',[x*1000 for x in grip_offsets],[-float('inf'),-3.,3.,float('inf')])]:
        result[f'by_{name}']={}
        for low,high in zip(bounds[:-1],bounds[1:]):
            selected=[i for i,x in enumerate(values) if low<=x<high]
            if selected:
                result[f'by_{name}'][f'{low:.1f}-{high:.1f}']=dict(episodes=len(selected),
                    success_rate=sum(successes[i] for i in selected)/len(selected))
    result['startup_randomization_seed']=args.seed
    def quantiles(values):
        if not values:
            return None
        tensor=torch.tensor(values,dtype=torch.float64)
        return dict(minimum=float(tensor.min()),p05=float(tensor.quantile(.05)),mean=float(tensor.mean()),median=float(tensor.quantile(.5)),
                    p95=float(tensor.quantile(.95)),p99=float(tensor.quantile(.99)),maximum=float(tensor.max()))
    result['peak_force_quantiles_n']=quantiles(peaks)
    if args.task_profile=='post_exit_alignment':
        result['alignment_terminal_errors']={key:quantiles(diagnostics[key]) for key in ['xy_error_m','orientation_error_rad']}
        result['projected_insertion_depth_mm']=args.alignment_insertion_depth_mm
    result['command_cap_fraction']=quantiles(cap_fractions)
    result['axial_command_cap_fraction']=quantiles(axial_cap_fractions)
    result['successful_peak_force_quantiles_n']=quantiles([p for p,s in zip(peaks,successes) if s])
    force_failed=[i for i,code in enumerate(reasons) if code==1]
    result['force_failure_diagnostics']={key:quantiles([diagnostics[key][i] for i in force_failed]) for key in diagnostic_keys}
    result['force_failure_diagnostics'].update(episodes=len(force_failed),
        command_lead_over_5mm=sum(diagnostics['command_lead_m'][i]>.005 for i in force_failed),
        true_depth_below_25mm=sum(diagnostics['true_depth_m'][i]<.025 for i in force_failed))
    timed_out=[i for i,code in enumerate(reasons) if code==3]
    result['timeout_diagnostics']={key:quantiles([diagnostics[key][i] for i in timed_out]) for key in diagnostic_keys}
    result['timeout_diagnostics'].update(episodes=len(timed_out),
        depth_below_25mm=sum(diagnostics['true_depth_m'][i]<.025 for i in timed_out),
        depth_above_55mm=sum(diagnostics['true_depth_m'][i]>.055 for i in timed_out),
        xy_outside_tolerance=sum(diagnostics['xy_error_m'][i]>=env.unwrapped.hole_geometry['success_xy_tolerance'] for i in timed_out),
        shaft_outside_hole=sum(diagnostics['shaft_fits'][i]<.5 for i in timed_out))
    failed=[i for i,code in enumerate(reasons) if code!=4]
    result['failure_examples']=[dict(end_reason=reasons[i],peak_force_n=peaks[i],friction_scale=friction_scales[i],
        peg_mass_kg=peg_masses[i],grip_axial_m=grip_offsets[i],delay_steps=delays[i],
        stiffness_scale=stiffness_scales[i],**{key:diagnostics[key][i] for key in diagnostic_keys})
        for i in failed[:20]]
    # Passive terminal records for all accepted equal-quota episodes, no action changes.
    episode_rows=[dict(index=i,success=bool(successes[i]),end_reason=reasons[i],
        seconds=lengths[i],peak_force_n=peaks[i],delay_steps=delays[i],
        stiffness_scale=stiffness_scales[i],friction_scale=friction_scales[i],
        **{key:diagnostics[key][i] for key in diagnostic_keys}) for i in range(len(successes))]
    record_path=Path(env.unwrapped.cfg.log_dir)/(label+'_episode_records.json')
    record_path.write_text(json.dumps(episode_rows,indent=2))
    result['passive_episode_records']=record_path.name
    print('EVALUATION', json.dumps(result), flush=True)
    return result

def adapt_cad_normalizers(model,geometry):
    trajectory=torch.tensor([geometry['joint_positions'],*geometry['validated_joint_trajectory']],dtype=torch.float32,device=args.device)
    mean=trajectory.mean(dim=0);half_span=(trajectory.max(dim=0).values-trajectory.min(dim=0).values)/2
    report={}
    for key in ['actor_obs_normalizer','critic_obs_normalizer']:
        normalizer=getattr(model,key)
        previous_mean=normalizer._mean[0,6:12].detach().cpu().tolist()
        std=torch.maximum(normalizer._std[0,6:12],half_span.clamp(min=.01))
        with torch.no_grad():
            normalizer._mean[0,6:12].copy_(mean)
            normalizer._std[0,6:12].copy_(std)
            normalizer._var[0,6:12].copy_(std.square())
        report[key]=dict(previous_mean=previous_mean,mean=mean.cpu().tolist(),std=std.cpu().tolist(),count=float(normalizer.count))
    print('CAD_JOINT_NORMALIZATION_ADAPTED',json.dumps(report),flush=True)
    return report

def main():
    # Recover the architecture of recurrent checkpoints during later eval or
    # resume calls, so their memory cannot silently be omitted.
    if args.checkpoint:
        saved_config=Path(args.checkpoint).parent/'agent.json'
        if saved_config.exists():
            saved_agent=json.loads(saved_config.read_text())
            if saved_agent['policy']['class_name'] in ['ActorCriticRecurrent','GatedPostXYZActorCriticRecurrent','PostResidualActorCriticRecurrent','PostSmoothResidualActorCriticRecurrent','CausalEpisodicActorCriticRecurrent','RiskGuidedActorCriticRecurrent']:
                args.recurrent=True
                args.privileged_critic=saved_agent['obs_groups']['critic']==['critic']
    urdf_path = ROOT/'assets/piper_pregrasped.urdf'
    usd_path = ROOT/'assets/piper_pregrasped.usd'
    if args.scene_profile=='challenge_cup':
        cad_directory=Path(args.cad_asset_dir) if args.cad_asset_dir else ROOT/'cad/challenge_cup'
        if args.cad_asset_dir is None and args.checkpoint and (Path(args.checkpoint).parent/'assets/task_geometry.json').exists():
            cad_directory=Path(args.checkpoint).parent/'assets'
        os.environ['PIPER_CAD_ASSET_DIR']=str(cad_directory)
        validation=json.loads((cad_directory/'training_asset_validation.json').read_text())
        if validation['status']!='completed':raise RuntimeError('CAD moving asset validation has not completed')
        print('USING_UPLOADED_STEP_ARTICULATION',validation['source_sha256'],flush=True)
    elif not args.rebuild_robot_asset and usd_path.exists() and usd_path.stat().st_mtime>=urdf_path.stat().st_mtime:
        print('REUSING_PREVALIDATED_ROBOT_USD',flush=True)
    else:
        convert_robot_asset()
    if args.scene_profile=='challenge_cup':
        from cad_env import ChallengeCupCfg as PiperInsertionCfg,ChallengeCupEnv as PiperInsertionEnv
    else:
        from piper_env import PiperInsertionCfg, PiperInsertionEnv
    if args.task_profile=='contact_withdrawal_only':
        if args.scene_profile!='challenge_cup' or args.joint_noise_rad!=0 or args.hole_yaw_noise_rad!=0:
            raise ValueError('Recovery preparation requires CAD and fixed initial joints/yaw')
        if not 3.<=args.recovery_offset_mm<=5.:
            raise ValueError('Recovery rim offset must be within3..5mm')
        from recovery_env import ContactRecoveryCfg as PiperInsertionCfg, ContactRecoveryEnv as PiperInsertionEnv
    if args.task_profile=='insertion_with_recovery':
        if args.scene_profile!='challenge_cup' or not args.recovery_checkpoint:
            raise ValueError('Hybrid requires CAD and an explicit frozen recovery checkpoint')
        from hybrid_env import HybridCfg as PiperInsertionCfg, HybridEnv as PiperInsertionEnv
    if args.task_profile=='post_exit_alignment':
        if args.scene_profile!='challenge_cup' or not args.alignment_bank:raise ValueError('Alignment requires CAD and explicit physical state bank')
        from alignment_env import AlignmentCfg as PiperInsertionCfg, AlignmentEnv as PiperInsertionEnv
    cfg = PiperInsertionCfg()
    if args.task_profile=='post_exit_alignment':
        cfg.alignment_banks=tuple(args.alignment_bank)
        cfg.alignment_sample_random=args.mode=='train'
        if not 1.<=args.alignment_reward_scale_mm<=10.:raise ValueError('Alignment reward scale must be1..10mm')
        cfg.alignment_reward_scale=args.alignment_reward_scale_mm/1000.
        if not 2.<=args.alignment_angle_weight<=20.:raise ValueError('Angle reward weight must be2..20')
        cfg.alignment_angle_weight=args.alignment_angle_weight
        if not 0<=args.alignment_insertion_depth_mm<=55:raise ValueError('Invalid alignment projection depth')
        cfg.alignment_insertion_depth=args.alignment_insertion_depth_mm/1000.
    if args.task_profile=='insertion_with_recovery':
        cfg.fixed_contact_cases=args.fixed_contact_cases
        cfg.block_force_filter=args.block_force_filter
        cfg.train_stop_failed_retry=args.mode=='train'
        cfg.initial_prealign=args.initial_prealign
        cfg.refresh_contact_on_reset=args.initial_prealign
        cfg.recovery_memory=args.recovery_memory
        cfg.recovery_offset=args.recovery_offset_mm/1000.
        cfg.recovery_center_x=args.recovery_center_x_mm/1000.
        cfg.lateral_cases_only=args.case_profile=='lateral'
        cfg.negative_lateral_case_only=args.case_profile=='negative_lateral'
        cfg.coverage_lateral_focus=args.case_profile=='lateral_focus'
        cfg.replay_original_axial_cases=args.case_profile in ('replay','negative_curriculum')
        cfg.replay_original_positive_lateral=args.case_profile=='negative_curriculum'
        if not 0<=args.training_joint_margin_cost<=.2:raise ValueError('Invalid training joint margin cost')
        cfg.training_joint_margin_cost=args.training_joint_margin_cost if args.mode=='train' else 0.

        if args.recovery_offset_min_mm is not None:
            if not .1<=args.recovery_offset_min_mm<=args.recovery_offset_mm<=5:raise ValueError('Invalid offset curriculum range')
            cfg.recovery_offset_min=args.recovery_offset_min_mm/1000.
        if not 0<=args.precontact_fraction<=1:raise ValueError('Invalid contact preparation fraction')
        cfg.precontact_fraction=args.precontact_fraction
    if args.task_profile=='contact_withdrawal_only':cfg.recovery_offset=args.recovery_offset_mm/1000.
    if not 0<=args.joint_noise_rad<=.05 or not 0<=args.hole_yaw_noise_rad<=.2:raise ValueError('Invalid reset noise')
    cfg.joint_noise=args.joint_noise_rad;cfg.hole_yaw_noise=args.hole_yaw_noise_rad
    cfg.reward_profile=args.reward_profile
    cfg.command_position_bound=args.command_position_bound_mm/1000
    cfg.command_rotation_bound=args.command_rotation_bound_rad
    cfg.command_axial_lead_bound=args.command_axial_lead_bound_mm/1000
    if not 0<=cfg.command_axial_lead_bound<=.02:
        raise ValueError('Axial lead limit must be within 0..20 mm')
    if cfg.command_axial_lead_bound and (cfg.command_position_bound or cfg.command_rotation_bound):
        raise ValueError('Axial-only lead limit cannot be combined with total tracking bounds')
    cfg.near_contact_position_step=args.near_contact_position_step_mm/1000
    cfg.near_contact_rotation_step=args.near_contact_rotation_step_rad
    cfg.near_contact_band=args.near_contact_band_mm/1000
    if not 0<=cfg.near_contact_position_step<=.002 or not 0<=cfg.near_contact_rotation_step<=.02 or not .01<=cfg.near_contact_band<=.10:
        raise ValueError('Near-contact increments must be within 0..2 mm / 0..0.02 rad, band 10..100 mm')
    if (cfg.near_contact_position_step or cfg.near_contact_rotation_step) and cfg.command_rotation_bound:
        raise ValueError('Total rotation reanchoring cannot be combined with increment limiting')
    if not 0<=cfg.command_position_bound<=.01 or not 0<=cfg.command_rotation_bound<=.1:
        raise ValueError('Command tracking bounds must be 0..10 mm and 0..0.1 rad')
    cfg.seed = args.seed
    cfg.hole_width = args.hole_width_mm/1000
    cfg.hole_xy_noise = args.hole_xy_noise_mm/1000
    cfg.observation_bias = args.observation_bias_mm/1000
    cfg.action_delay_min, cfg.action_delay_max = args.action_delay_min, args.action_delay_max
    cfg.friction_scale_min, cfg.friction_scale_max = args.friction_scale_min, args.friction_scale_max
    cfg.force_noise_std, cfg.force_bias_bound = args.force_noise_std_n, args.force_bias_bound_n
    cfg.stiffness_scale_min, cfg.stiffness_scale_max = args.stiffness_scale_min, args.stiffness_scale_max
    cfg.peg_mass_min,cfg.peg_mass_max=args.peg_mass_min_g/1000,args.peg_mass_max_g/1000
    cfg.grip_axial_min,cfg.grip_axial_max=args.grip_axial_min_mm/1000,args.grip_axial_max_mm/1000
    cfg.grip_endpoint_fraction=args.grip_endpoint_fraction
    cfg.grip_positive_endpoint_probability=args.grip_positive_endpoint_probability
    cfg.nominal_replay_fraction=args.nominal_replay_fraction
    cfg.nominal_dynamics_fraction=args.nominal_dynamics_fraction
    if not 0<=cfg.nominal_dynamics_fraction<=1:
        raise ValueError('Nominal dynamics fraction must be within 0..1')
    if not 0<=cfg.nominal_replay_fraction<=1:
        raise ValueError('Nominal replay fraction must be within 0..1')
    if not 0<=cfg.grip_positive_endpoint_probability<=1:
        raise ValueError('Positive endpoint probability must be within 0..1')
    if not 0<=cfg.grip_endpoint_fraction<=1:
        raise ValueError('Endpoint fraction must be within 0..1')
    if not .005<=cfg.peg_mass_min<=cfg.peg_mass_max<=.2:
        raise ValueError('Peg mass must be within 5..200 g')
    if not -.015<=cfg.grip_axial_min<=cfg.grip_axial_max<=.015:
        raise ValueError('Grip axial offset must be within -15..15 mm')
    if cfg.nominal_replay_fraction and not (
        cfg.peg_mass_min<=.02<=cfg.peg_mass_max and cfg.grip_axial_min<=0.<=cfg.grip_axial_max):
        raise ValueError('Nominal replay must lie within the declared mass/grip bounds')
    if cfg.grip_axial_min!=0. or cfg.grip_axial_max!=0. or cfg.peg_mass_min!=.02 or cfg.peg_mass_max!=.02:
        cfg.scene.replicate_physics=False
        cfg.sim.enable_scene_query_support=args.mode=='smoke' and args.device=='cpu'
    if not 0 <= cfg.force_noise_std <= 10 or not 0 <= cfg.force_bias_bound <= 10:
        raise ValueError('Force measurement noise and bias bounds must be within 0..10 N')
    if not .25 <= cfg.stiffness_scale_min <= cfg.stiffness_scale_max <= 2.:
        raise ValueError('Stiffness scaling must be within .25..2')
    if not 0 <= cfg.action_delay_min <= cfg.action_delay_max <= 6:
        raise ValueError('Action delay bounds must be within 0..6 policy steps')
    if not 0 < cfg.friction_scale_min <= cfg.friction_scale_max <= 4.:
        raise ValueError('Friction scaling bounds must be within (0,4]')
    cfg.privileged_critic = args.privileged_critic
    cfg.state_space = 33 if args.privileged_critic else 0
    if cfg.hole_xy_noise<0 or cfg.observation_bias<0:
        raise ValueError('Noise bounds must be nonnegative')
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or 'cuda:0'
    if args.scene_profile=='challenge_cup' and args.mode=='smoke' and args.device=='cpu':
        cfg.sim.enable_scene_query_support=True
    cfg.viewer.eye = (1., 1., .8)
    cfg.viewer.lookat = (.35, 0., .2)
    logdir = ROOT / 'logs' / args.run_name
    logdir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(logdir)
    raw = PiperInsertionEnv(cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=1.)
    if args.task_profile=='insertion_with_recovery':
        from hybrid_env import self_test
        self_test()
        frozen_agent=json.loads((Path(args.recovery_checkpoint).parent/'agent.json').read_text())
        frozen_runner=OnPolicyRunner(env,frozen_agent,log_dir=None,device=env.device)
        frozen_runner.load(args.recovery_checkpoint,load_optimizer=False)
        raw.recovery_model=frozen_runner.alg.policy
        raw.recovery_policy=frozen_runner.get_inference_policy(device=env.device)
        for parameter in raw.recovery_model.parameters():parameter.requires_grad_(False)
        if args.alignment_checkpoint:
            alignment_agent=json.loads((Path(args.alignment_checkpoint).parent/'agent.json').read_text())
            alignment_runner=OnPolicyRunner(env,alignment_agent,log_dir=None,device=env.device)
            alignment_runner.load(args.alignment_checkpoint,load_optimizer=False)
            raw.alignment_model=alignment_runner.alg.policy
            raw.alignment_policy=alignment_runner.get_inference_policy(device=env.device)
            for parameter in raw.alignment_model.parameters():parameter.requires_grad_(False)
            raw.supervisor.alignment_xy_tolerance=.0005
            raw.supervisor.alignment_angle_tolerance=.03
        if args.initial_insertion_checkpoint:
            initial_agent=json.loads((Path(args.initial_insertion_checkpoint).parent/'agent.json').read_text())
            initial_runner=OnPolicyRunner(env,initial_agent,log_dir=None,device=env.device)
            initial_runner.load(args.initial_insertion_checkpoint,load_optimizer=False)
            raw.initial_model=initial_runner.alg.policy
            raw.initial_policy=initial_runner.get_inference_policy(device=env.device)
            for parameter in raw.initial_model.parameters():parameter.requires_grad_(False)

    if args.near_alignment_checkpoint:
        assert args.task_profile=='insertion_with_recovery' and args.alignment_checkpoint
        near_agent=json.loads((Path(args.near_alignment_checkpoint).parent/'agent.json').read_text())
        near_runner=OnPolicyRunner(env,near_agent,log_dir=None,device=env.device)
        near_runner.load(args.near_alignment_checkpoint,load_optimizer=False)
        raw.near_alignment_model=near_runner.alg.policy
        assert isinstance(raw.near_alignment_model,CurrentObsAlignmentActorCriticRecurrent)
        raw.near_alignment_policy=near_runner.get_inference_policy(device=env.device)
        base=raw.alignment_model.state_dict()
        for k,v in raw.near_alignment_model.state_dict().items():
            if k not in base:assert k in ['current_actor_mean','current_actor_std','current_actor_mask'],k
            elif not k.startswith('actor.'):assert torch.equal(v,base[k]),('Near alignment frozen state changed',k)
        for parameter in raw.near_alignment_model.parameters():parameter.requires_grad_(False)
    if args.deep_recovery_checkpoint:
        assert args.task_profile=='insertion_with_recovery' and args.recovery_checkpoint
        deep_agent=json.loads((Path(args.deep_recovery_checkpoint).parent/'agent.json').read_text())
        deep_runner=OnPolicyRunner(env,deep_agent,log_dir=None,device=env.device)
        deep_runner.load(args.deep_recovery_checkpoint,load_optimizer=False)
        raw.deep_recovery_model=deep_runner.alg.policy
        assert isinstance(raw.deep_recovery_model,CurrentObsRecoveryActorCriticRecurrent)
        raw.deep_recovery_policy=deep_runner.get_inference_policy(device=env.device)
        base=raw.recovery_model.state_dict()
        for k,v in raw.deep_recovery_model.state_dict().items():
            if k not in base:assert k in ['current_actor_mean','current_actor_std','current_actor_mask'],('Unexpected expert tensor',k)
            elif not k.startswith('actor.'):assert torch.equal(v,base[k]),('Expert frozen encoder/critic/N/std changed',k)
        for parameter in raw.deep_recovery_model.parameters():parameter.requires_grad_(False)
    if args.late_initial_checkpoint:
        assert args.task_profile=='insertion_with_recovery' and args.initial_insertion_checkpoint and not args.initial_prealign
        late_agent=json.loads((Path(args.late_initial_checkpoint).parent/'agent.json').read_text())
        late_runner=OnPolicyRunner(env,late_agent,log_dir=None,device=env.device)
        late_runner.load(args.late_initial_checkpoint,load_optimizer=False)
        raw.late_initial_model=late_runner.alg.policy
        raw.late_initial_policy=late_runner.get_inference_policy(device=env.device)
        base=raw.initial_model.state_dict()
        is_adapter=isinstance(raw.late_initial_model,LateXYActorCriticRecurrent)
        is_xyz=isinstance(raw.late_initial_model,LateXYZActorCriticRecurrent)
        for k,v in raw.late_initial_model.state_dict().items():
            if k not in base:
                assert (is_adapter and (k.startswith('xy_adapter.') or k in ['xy_feature_mean','xy_feature_std'])) or (is_xyz and (k.startswith('position_head.') or k in ['position_mean','position_std'])),('Unexpected specialist tensor',k)
            elif is_adapter or is_xyz:assert torch.equal(v,base[k]),('Specialist frozen base changed',k)
            elif k in ['actor.6.weight','actor.6.bias']:assert torch.equal(v[2:],base[k][2:]),('Late nonXY decoder',k)
            else:assert torch.equal(v,base[k]),('Late must share exact original encoder/body/critic/N/std',k)

        for parameter in raw.late_initial_model.parameters():parameter.requires_grad_(False)
    run_task(env,raw,cfg,logdir)
    if args.scene_profile=='challenge_cup':raw.save_pose_snapshot(logdir/'cad_final_pose_snapshot.json')
    env.close()
    STATUS_PATH.write_text(json.dumps(dict(status='completed',mode=args.mode,
        seed=args.seed,num_envs=args.num_envs,iterations=args.iterations,end_time=time.time()),indent=2))

def convert_robot_asset():
    # The existing image does not autoload its installed importer for the newer
    # Isaac Lab converter. Enable the installed extension before conversion.
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate('isaacsim.asset.importer.urdf', True)
    print('URDF_IMPORTER_ENABLED', manager.is_extension_enabled('isaacsim.asset.importer.urdf'), flush=True)
    InstalledImporterConverter(UrdfConverterCfg(
        asset_path=str(ROOT/'assets/piper_pregrasped.urdf'),
        usd_dir=str(ROOT/'assets'), usd_file_name='piper_pregrasped.usd',
        fix_base=True, merge_fixed_joints=True, make_instanceable=False,
        self_collision=False, collision_from_visuals=False, collider_type='convex_hull',
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=800., damping=45.))))
    # The installed importer instances collision subtrees regardless of its
    # top-level asset setting. Deinstance our generated asset so contact offsets
    # can be configured accurately for the 10 mm peg.
    from pxr import Usd
    stage = Usd.Stage.Open(str(ROOT/'assets/piper_pregrasped.usd'))
    for _ in range(3):
        for prim in stage.Traverse():
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
    stage.GetRootLayer().Save()

def run_task(env,raw,cfg,logdir):
    if args.mode=='collect_alignment':
        if args.task_profile!='insertion_with_recovery':raise ValueError('Collector requires verified hybrid preparer/withdrawal')
        from alignment_env import collect_states
        collect_states(env,raw,logdir,args)
        return
    if args.mode=='smoke' and args.task_profile=='post_exit_alignment':
        from alignment_env import alignment_smoke
        alignment_smoke(env,raw,logdir)
        return
    if args.mode=='smoke' and args.task_profile=='insertion_with_recovery':
        from hybrid_env import hybrid_smoke
        hybrid_smoke(env,raw,logdir)
        return
    if args.mode=='smoke' and args.task_profile=='contact_withdrawal_only':
        from recovery_env import recovery_smoke
        recovery_smoke(env,raw,logdir)
        return
    if args.mode == 'smoke':
        obs = env.get_observations()
        if not torch.isfinite(obs['policy']).all():
            raise RuntimeError('Non-finite initial observation')
        positions, orientations, forces = raw._geometry()
        from isaaclab.utils.math import quat_apply,quat_apply_inverse
        nominal_tip=raw.robot.data.body_pos_w[:,raw.ee_idx]+quat_apply(
            raw.robot.data.body_quat_w[:,raw.ee_idx],raw.nominal_tip_offset)-raw.scene.env_origins
        expected=(quat_apply_inverse(raw.hole_quat,nominal_tip-raw.hole_pos)+raw.hole_estimation_bias)*20.
        if not torch.allclose(obs['policy'][:,:3],expected,atol=1e-6):
            raise RuntimeError('Position estimation bias is not correctly applied to observations')
        if not torch.allclose(obs['policy'][:,24:27],raw._measured_force(forces),atol=1e-6):
            raise RuntimeError('Force measurement noise was not applied correctly')
        if cfg.privileged_critic and not torch.allclose(obs['critic'][:,24:27],forces.clamp(-100,100)*.02):
            raise RuntimeError('Privileged force observations must use true force')
        if cfg.privileged_critic and not torch.allclose(obs['critic'][:,:3],positions*20.,atol=1e-6):
            raise RuntimeError('Critic must use true physical peg position')
        raw.verify_grip_raycast()
        nominal=raw.nominal_dynamics_mask
        if nominal.any():
            assert (raw.action_delay.steps[nominal]==0).all()
            assert (raw.friction_scales[nominal.cpu()]==1.).all()
            assert (raw.stiffness_scales[nominal]==1.).all()
            assert (raw.force_bias[nominal]==0.).all()
            assert (raw.force_noise_scales[nominal]==1.).all()
            print('NOMINAL_DYNAMICS_REPLAY_VERIFIED',int(nominal.sum()),flush=True)
        if args.scene_profile=='challenge_cup':raw.verify_cad_scene()
        if args.scene_profile=='challenge_cup':
            start=raw._geometry()[0].clone()
            for _ in range(20):
                probe=torch.zeros(args.num_envs,6,device=env.device);probe[:,0]=.5
                obs,_,_,_=env.step(probe)
            response=raw._geometry()[0]-start
            minimum_response=.005
            if cfg.near_contact_position_step:
                minimum_response*=min(1.,cfg.near_contact_position_step/.00075)
            if float(response[:,0].median())<minimum_response:raise RuntimeError(f'CAD control direction mismatch: {response.tolist()}')
            print('CAD_RESPONSE_MINIMUM_M',minimum_response,flush=True)
            print('CAD_POSITIVE_X_RESPONSE_M',float(response[:,0].median()),flush=True)
            obs,_=env.reset()
        initial_joints=raw.robot.data.joint_pos.clone()
        axial_probe=None
        if cfg.command_axial_lead_bound:
            clipped=0;max_lead=0.;max_force=0.
            for _ in range(240):
                probe=torch.zeros(args.num_envs,6,device=env.device);probe[:,2]=-.5
                obs,reward,dones,info=env.step(probe)
                if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
                    raise RuntimeError('Non-finite axial guard physical probe')
                clipped+=int(raw.last_axial_cap_active.sum())
                max_lead=max(max_lead,float(raw.last_control_axial_lead.max()))
                max_force=max(max_force,float(raw._geometry()[2].norm(dim=-1).max()))
                if 'terminal' in info and bool(dones.any()):
                    max_force=max(max_force,float(info['terminal']['peak_force_n'].max()))
            assert clipped>0 and max_force>=5.,(clipped,max_force)
            assert max_lead<=cfg.command_axial_lead_bound+1e-6,max_lead
            axial_probe={'steps':240,'clipped_environment_steps':clipped,'max_control_time_lead_m':max_lead,
                         'max_true_assembly_force_n':max_force,'status':'passed'}
            print('AXIAL_GUARD_PHYSICAL_PROBE_PASSED',json.dumps(axial_probe),flush=True)
            obs,_=env.reset()
        print('INITIAL_GEOMETRY', json.dumps(dict(
            tip_relative=positions[0].tolist(), orientation_error=orientations[0].tolist(),
            joint_positions=raw.robot.data.joint_pos[0].tolist())), flush=True)
        for step in range(240):
            action = torch.zeros(args.num_envs,6,device=env.device)
            action[:,2] = -.3 if step < 100 else .3
            if cfg.near_contact_position_step or cfg.near_contact_rotation_step:
                action[:,:2]=.8 if step<100 else -.8
                action[:,4]=.8 if step<100 else -.8
            obs, reward, dones, _ = env.step(action)
            if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
                raise RuntimeError(f'Non-finite smoke step {step}')
            if step % 40 == 0:
                relative, rotation, force = raw._geometry()
                print('SMOKE_STEP',step,'relative',relative[0].tolist(),
                    'force_max',float(force.norm(dim=-1).max()),flush=True)
        if args.scene_profile=='challenge_cup':
            movement=float((raw.robot.data.joint_pos-initial_joints).abs().max())
            if movement<.001:raise RuntimeError('CAD articulation did not move during control check')
            raw.save_pose_snapshot(logdir/'cad_pose_snapshot.json')
            print('CAD_ARTICULATION_MOVEMENT_RAD',movement,flush=True)
        result=dict(status='passed',num_envs=args.num_envs,steps=240,scene_profile=args.scene_profile,
            completed_episodes=raw.completed_episodes,completed_successes=raw.completed_successes,
            hole_geometry=raw.hole_geometry,success_profile=cfg.success_profile,
            observation_bias_m=cfg.observation_bias,hole_xy_noise_m=cfg.hole_xy_noise,
            bias_applied_to='policy_position_observation_only')
        if axial_probe:
            result['axial_guard_physical_probe']=axial_probe
            result['steps']=480
        if args.scene_profile=='challenge_cup':
            result.update(uploaded_model_source_sha256=cfg.geometry['source_sha256'],action_frame=cfg.action_frame,
                          controller_response_validation='positive_hole_X',jacobian_frame='root' if raw.ik_in_root_frame else 'world')
        result.update(action_delay_min=cfg.action_delay_min,action_delay_max=cfg.action_delay_max,
                      friction_scale_min=cfg.friction_scale_min,friction_scale_max=cfg.friction_scale_max)
        result.update(force_noise_std_n=cfg.force_noise_std,force_bias_bound_n=cfg.force_bias_bound,
                      stiffness_scale_min=cfg.stiffness_scale_min,stiffness_scale_max=cfg.stiffness_scale_max,
                      force_labels='true_physics_force_unchanged')
        result.update(peg_mass_min_g=args.peg_mass_min_g,peg_mass_max_g=args.peg_mass_max_g,
                      grip_axial_min_mm=args.grip_axial_min_mm,grip_axial_max_mm=args.grip_axial_max_mm,
                      payload_assignment='exact_mass_COM_inertia_composition_readback',
                      grip_shape_assignment='USD_before_PhysX_physical_raycast',
                      actor_grasp_transform='nominal',critic_grasp_transform='true',
                      simulation_device=cfg.sim.device,
                      physical_raycast_validation='passed_cpu' if cfg.sim.device=='cpu' else 'separate_cpu_probe_required')
        result['grip_endpoint_fraction']=cfg.grip_endpoint_fraction
        result['grip_positive_endpoint_probability']=cfg.grip_positive_endpoint_probability
        result['nominal_replay_fraction']=cfg.nominal_replay_fraction
        result['nominal_dynamics_fraction']=cfg.nominal_dynamics_fraction
        result['nominal_dynamics_environments']=int(raw.nominal_dynamics_mask.sum())
        result['nominal_replay_environments']=int(raw.nominal_replay_mask.sum())
        result['reward_profile']=cfg.reward_profile
        result['potential_alignment_length_scale_mm']=alignment_length_scale(cfg.reward_profile)*1000
        result.update(command_position_bound_mm=args.command_position_bound_mm,
                      command_rotation_bound_rad=args.command_rotation_bound_rad,
                      command_axial_lead_bound_mm=args.command_axial_lead_bound_mm)
        result.update(near_contact_position_step_mm=args.near_contact_position_step_mm,
                      near_contact_rotation_step_rad=args.near_contact_rotation_step_rad,
                      near_contact_band_mm=args.near_contact_band_mm)
        (logdir/'smoke.json').write_text(json.dumps(result,indent=2))
        print('SMOKE_PASSED',json.dumps(result),flush=True)
    else:
        agent = AgentCfg().to_dict()
        if args.exploration_std is not None and not args.checkpoint:agent['policy']['init_noise_std']=args.exploration_std
        agent['obs_groups']['critic']=['critic'] if args.privileged_critic else ['policy']
        if args.recurrent:
            agent['policy'].update(class_name='ActorCriticRecurrent',rnn_type='gru',
                                   rnn_hidden_dim=33,rnn_num_layers=1)
        if args.checkpoint and (Path(args.checkpoint).parent/'agent.json').is_file():
            checkpoint_agent=json.loads((Path(args.checkpoint).parent/'agent.json').read_text())
            if checkpoint_agent['policy']['class_name'] in ['GatedPostXYZActorCriticRecurrent','PostResidualActorCriticRecurrent','PostSmoothResidualActorCriticRecurrent','CausalEpisodicActorCriticRecurrent','RiskGuidedActorCriticRecurrent']:
                assert args.recurrent and checkpoint_agent['obs_groups']['policy']==['policy']
                agent['policy']=checkpoint_agent['policy'].copy()
                agent['obs_groups']=checkpoint_agent['obs_groups'].copy()
        (logdir/'agent.json').write_text(json.dumps(agent,indent=2))
        (logdir/'task.json').write_text(json.dumps(dict(
            seed=args.seed,num_envs=args.num_envs,geometry=cfg.geometry,scene_profile=args.scene_profile,
            joint_noise=cfg.joint_noise,hole_xy_noise=cfg.hole_xy_noise,hole_yaw_noise=cfg.hole_yaw_noise,
            physics_dt=cfg.sim.dt,decimation=cfg.decimation,peg_attachment='fixed_pregrasp',
            success_hold_steps=8,max_success_force_n=40),indent=2))
        task = json.loads((logdir/'task.json').read_text())
        task['action_frame']=cfg.action_frame
        if args.scene_profile=='challenge_cup':raw.verify_cad_scene()
        task['reward_profile'] = getattr(cfg,'reward_profile','dense_v1')
        task.update(contact_force_cost='.02*e+.002*e^2' if cfg.reward_profile in ('contact_v9','alignment_v11','contact_v12') else '.02*e',
                    contact_force_excess='e=clamp(true_force_norm-15,0,150)',
                    entropy_coef=args.entropy_coef,success_force_limit_n=40.,abort_force_limit_n=100.,
                    action_mapping='unchanged_2mm_0.012rad_integrated_IK')
        task.update(potential_alignment_length_scale_mm=alignment_length_scale(cfg.reward_profile)*1000)
        task.update(potential_orientation_length_scale_rad=orientation_length_scale(cfg.reward_profile),
                    contact_episode_threshold_n=5.,contact_retreat_depth_mm=2.)
        task.update(command_position_bound_mm=args.command_position_bound_mm,
                    command_rotation_bound_rad=args.command_rotation_bound_rad,
                    tracking_bound_feedback='encoder_FK_actual_flange_pose_only',
                    control_profile='bounded_tracking_command' if cfg.command_position_bound or cfg.command_rotation_bound else 'original_integrated_command')
        task.update(near_contact_position_step_mm=args.near_contact_position_step_mm,
                    near_contact_rotation_step_rad=args.near_contact_rotation_step_rad,
                    near_contact_band_mm=args.near_contact_band_mm,
                    near_contact_feedback='policy_nominal_tip_estimate_only_no_true_grasp_or_force')
        if cfg.near_contact_position_step or cfg.near_contact_rotation_step:
            task['control_profile']='near_contact_increment_limit'
            task['action_mapping']='2mm_0.012rad_RL_increment_delayed_then_near_contact_norm_limit_before_integrated_IK'
            if cfg.command_position_bound:
                task['control_profile']='near_contact_increment_with_position_antiwindup'
                task['action_mapping']+='then_encoder_FK_position_tracking_bound'
        task['initial_contact_measurement']='alignment_reset_unknown_force_sentinel3_in_existing_force_channels;first_action_discarded_measured_joint_hold_four_existing_substeps;true_force_guards_unchanged;no_extra_steps'
        task['command_axial_lead_bound_mm']=args.command_axial_lead_bound_mm
        task['axial_lead_feedback']='command_target_and_encoder_FK_along_known_hole_axis_only'
        if cfg.command_axial_lead_bound:
            task['control_profile']='near_contact_increment_limit_with_axial_lead_limit'
            task['action_mapping']+='then_axial_target_lead_limit_before_IK'
        task['geometry']['hole_width'] = cfg.hole_width
        task.update(effective_hole_geometry=raw.hole_geometry,
                    success_profile=cfg.success_profile,observation_bias_m=cfg.observation_bias,
                    bias_distribution='per_axis_uniform_episode_constant',
                    force_sampling='maximum_over_four_physics_steps',
                    source_checkpoint=args.checkpoint,warm_start=args.warm_start,
                    compare_checkpoint=args.compare_checkpoint,
                    reference_checkpoints=args.reference_checkpoint,
                    architecture='gru33' if args.recurrent else 'mlp',
                    actor_groups=agent['obs_groups']['policy'],critic_groups=agent['obs_groups']['critic'],
                    exploration_std_override=args.exploration_std)
        task.update(action_delay_min=cfg.action_delay_min,action_delay_max=cfg.action_delay_max,
                    policy_dt_s=raw.step_dt,delay_distribution='uniform_integer_episode_constant',
                    delayed_signal='action_before_integrated_IK_command',
                    friction_scale_min=cfg.friction_scale_min,friction_scale_max=cfg.friction_scale_max,
                    friction_distribution='16_linear_buckets_per_env_run_constant',
                    friction_applied_to='robot_and_four_hole_walls_static_and_dynamic')
        task.update(force_noise_std_n=cfg.force_noise_std,force_bias_bound_n=cfg.force_bias_bound,
                    force_noise_distribution='independent_per_axis_gaussian_per_policy_step_cached',
                    force_bias_distribution='per_axis_uniform_episode_constant',
                    force_noise_applied_to='actor_force_observation_only',
                    privileged_force='true_force',force_reward_and_success='true_force',
                    stiffness_scale_min=cfg.stiffness_scale_min,stiffness_scale_max=cfg.stiffness_scale_max,
                    stiffness_distribution='uniform_continuous_per_environment_run_constant',
                    nominal_stiffness=800.,unchanged_damping=45.)
        task.update(peg_mass_min_g=args.peg_mass_min_g,peg_mass_max_g=args.peg_mass_max_g,
                    grip_axial_min_mm=args.grip_axial_min_mm,grip_axial_max_mm=args.grip_axial_max_mm,
                    payload_distribution='independent_uniform_per_environment_run_constant',
                    grip_distribution='uniform_axial_translation_plus_symmetric_endpoint_mixture_run_constant',
                    grip_endpoint_fraction=cfg.grip_endpoint_fraction,
                    grip_positive_endpoint_probability=cfg.grip_positive_endpoint_probability,
                    nominal_replay_fraction=cfg.nominal_replay_fraction,
                    nominal_replay_environments=int(raw.nominal_replay_mask.sum()),
                    nominal_replay_mass_g=20.,nominal_replay_grip_mm=0.,
                    payload_assignment='replace_20g_peg_component_only_exact_mass_COM_inertia',
                    non_payload_inertia='preserved_imported_composite_minus_nominal_peg',
                    actor_grasp_transform='nominal',critic_grasp_transform='true',
                    pose_reward_and_success='true_physical_tip_and_shaft',
                    replicate_physics=cfg.scene.replicate_physics)
        task.update(nominal_dynamics_fraction=cfg.nominal_dynamics_fraction,
                    nominal_dynamics_environments=int(raw.nominal_dynamics_mask.sum()),
                    nominal_dynamics={'delay_steps':0,'friction_scale':1.,'stiffness_scale':1.,
                                      'force_bias_n':0.,'force_noise_std_n':1.})
        task.update(task_profile=args.task_profile,recovery_offset_mm=args.recovery_offset_mm if args.task_profile=='contact_withdrawal_only' else None)
        if args.task_profile=='post_exit_alignment':
            task.update(alignment_banks=raw.bank_manifests,alignment_reward_scale_mm=args.alignment_reward_scale_mm,
                          alignment_angle_weight=args.alignment_angle_weight,
                        alignment_insertion_depth_mm=args.alignment_insertion_depth_mm,
                        preparation='separate collector; no scripted approach or retreat inside PPO episodes',
                        alignment_reward='gamma.99 dense distance potential, terminal potential0, plus success100',
                        success_profile=cfg.success_profile,episode_length_s=cfg.episode_length_s,
                        axial_action=0.,policy_observations=33,GRU_reset='only at actual alignment episode boundaries',
                        reset_targets='saved Cartesian commands and joint servo targets restored',
                        bank_sampling='random' if args.mode=='train' else 'deterministic matching per-env bank sequences')
        if args.task_profile=='insertion_with_recovery':
            import hashlib
            task.update(recovery_checkpoint=args.recovery_checkpoint,recovery_checkpoint_sha256=hashlib.sha256(Path(args.recovery_checkpoint).read_bytes()).hexdigest(),
                        recovery_memory=args.recovery_memory,fixed_contact_cases=args.fixed_contact_cases,precontact_fraction=args.precontact_fraction,recovery_offset_mm=args.recovery_offset_mm,
                        precontact_preparer='slow approach to actual5N wall/net contact; zero immediate reward; no scripted retreat' if args.fixed_contact_cases else None,
                        fixture_center_shift_x_m=cfg.recovery_center_x if cfg.fixed_contact_cases else 0.,
                        supervisor_inputs='actor33 only; force>=10N2steps near mouth; estimated retreat5mm vector-forceEMA(.25)<5N8steps; alignment1mm/.08rad8steps',
                        command_on_switch='preserved; delay_buffer_preserved',recovery_max_attempts=3,episode_length_s=cfg.episode_length_s,
                        realignment='insertion policy XY/rotation; zero axial action',frozen_recovery=True,block_force_filter=args.block_force_filter,block_filter_alpha=.25,block_filter_fast_path_n=40.)
            if args.alignment_checkpoint:
                task.update(alignment_checkpoint=args.alignment_checkpoint,
                    alignment_checkpoint_sha256=hashlib.sha256(Path(args.alignment_checkpoint).read_bytes()).hexdigest(),
                    realignment='frozen independent alignment skill; zero axial action',frozen_alignment=True,
                    alignment_GRU='cold reset on handover; first step zero; next step starts cold',
                    alignment_gate='estimatedXY0.5mm/angle0.03rad8steps; forceEMA<5N',
                    delay_buffer_on_alignment_switch='preserved')
        if args.task_profile=='contact_withdrawal_only':
            task.update(success_profile=cfg.success_profile,recovery_center_x_m=cfg.recovery_center_x,preparation='fixed four rim directions; physical approach until5N; no learned action during preparation',
                        preparation_action_z=-.1,preparation_max_steps=240,episode_length_s=8.,
                        learned_task='withdrawal only; no retry or insertion claim',preparation_reward=0.)
        (logdir/'task.json').write_text(json.dumps(task,indent=2))
        if args.scene_profile=='challenge_cup':
            task.update(uploaded_model_source_sha256=cfg.geometry['source_sha256'],
                        cad_asset_directory=os.environ['PIPER_CAD_ASSET_DIR'],
                        jacobian_frame='root' if raw.ik_in_root_frame else 'world',
                        action_mapping='hole_frame_2mm_0.012rad_increments_transformed_to_world_before_integrated_IK',
                        force_observation_frame='hole',geometry_source='uploaded_STEP_articulated_left_arm_plus_original_station',
                        workpiece_origin='experimental_peg_and_socket_not_in_STEP',
                        physics_parameter_source=cfg.geometry['physics_parameter_source'])
            (logdir/'task.json').write_text(json.dumps(task,indent=2))
        runner = OnPolicyRunner(env,agent,log_dir=str(logdir),device=env.device)
        if args.checkpoint:
            source=torch.load(args.checkpoint,map_location=env.device,weights_only=False)
            has_memory=any(key.startswith('memory_a.') for key in source['model_state_dict'])
            if args.recurrent and not has_memory:
                if not args.warm_start:
                    raise ValueError('MLP to recurrent transfer requires --warm_start')
                from transfer_memory import initialize_memory_transfer
                initialize_memory_transfer(runner.alg.policy,source['model_state_dict'])
            else:
                runner.load(args.checkpoint,load_optimizer=not args.warm_start)
            if args.warm_start:
                runner.current_learning_iteration = 0
            elif args.mode=='train':
                # This installed runner records the last completed update in
                # its checkpoint, so resume at the following index.
                runner.current_learning_iteration=int(source['iter'])+1
                runner.alg.learning_rate=runner.alg.optimizer.param_groups[0]['lr']
            if args.exploration_std is not None:
                policy = runner.alg.policy
                with torch.no_grad():
                    if hasattr(policy,'std'):
                        policy.std.fill_(args.exploration_std)
                    else:
                        policy.log_std.fill_(torch.tensor(args.exploration_std).log().item())
        if args.adapt_cad_normalization:
            if args.scene_profile!='challenge_cup' or not args.checkpoint:
                raise ValueError('CAD normalization adaptation requires a CAD scene and source checkpoint')
            task['cad_normalization_adaptation']=adapt_cad_normalizers(runner.alg.policy,cfg.geometry)
            task['normalization_profile']='CAD_trajectory_joint_mean_and_span_only_other_features_preserved'
        if args.init_axial_bias is not None:
            if args.scene_profile!='challenge_cup' or not -.6<=args.init_axial_bias<=0:
                raise ValueError('Invalid CAD axial initialization')
            head=[m for m in runner.alg.policy.actor.modules() if isinstance(m,torch.nn.Linear)][-1]
            if head.out_features!=6:raise RuntimeError('Unexpected actor output size')
            with torch.no_grad():head.weight.zero_();head.bias.zero_();head.bias[2]=args.init_axial_bias
            task['actor_initialization']=dict(type='trainable_axial_output_bias',axial_mean=args.init_axial_bias,
                all_six_outputs_remain_trainable=True,prior='known_insertion_direction_only')
            print('CAD_AXIAL_POLICY_INITIALIZATION',args.init_axial_bias,flush=True)
        task['first_update_index']=runner.current_learning_iteration
        task['effective_learning_rate']=runner.alg.learning_rate
        (logdir/'task.json').write_text(json.dumps(task,indent=2))
        if args.privileged_critic:
            # Check that changing the critic's hidden training state cannot
            # alter the actor's action, including at the recurrent warm start.
            model=runner.alg.policy
            initial=env.get_observations()
            altered=initial.clone()
            altered['critic'].fill_(1000.)
            reset_mask=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            with torch.inference_mode():
                model.reset(reset_mask)
                action_a=model.act_inference(initial)
                model.reset(reset_mask)
                action_b=model.act_inference(altered)
                if not torch.allclose(action_a,action_b,atol=1e-6):
                    raise RuntimeError('Actor action leaks privileged critic observations')
                model.reset(reset_mask)
            print('ACTOR_PRIVILEGED_STATE_INDEPENDENCE_PASSED',flush=True)
            if args.recurrent:
                # Match the runner's initial state: both memories must be
                # uninitialized together when rollout storage is allocated.
                model.memory_a.hidden_state=None
                model.memory_c.hidden_state=None
        if args.mode=='train' and args.task_profile=='insertion_with_recovery':
            from masked_ppo import install
            install(runner,raw,logdir,env,args)
        if args.mode == 'train':
            runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        if args.task_profile=='insertion_with_recovery':
            raw.cfg.train_stop_failed_retry=False
            raw.cfg.training_joint_margin_cost=0.
            raw.cfg.recovery_offset_min=-1. # Evaluation tests the hardest edge, never a mixture average.
        # Held-out randomization seed shared by both compared controllers.
        env.seed(args.seed+10000)
        raw.completed_episodes=raw.completed_successes=0
        baseline=None if args.skip_straight_baseline else evaluate(env,label='fixed_straight_insertion',episodes=args.eval_episodes)
        env.seed(args.seed+10000)
        raw.completed_episodes=raw.completed_successes=0
        if args.mirror_teacher_probe or args.mirror_dagger_fit:
            assert args.mode=='eval' and args.iterations==0 and args.task_profile=='insertion_with_recovery'
            mirror_cp=Path(args.mirror_teacher_checkpoint or args.checkpoint)
            mirror_agent=json.loads((mirror_cp.parent/'agent.json').read_text())
            mirror_runner=OnPolicyRunner(env,mirror_agent,log_dir=None,device=env.device)
            mirror_runner.load(str(mirror_cp),load_optimizer=False)
            raw.mirror_model=mirror_runner.alg.policy
            raw.mirror_policy=mirror_runner.get_inference_policy(device=env.device)
            for parameter in raw.mirror_model.parameters():parameter.requires_grad_(False)
            (Path(logdir)/'mirror_diagnostic_protocol.json').write_text(json.dumps(dict(status='diagnostic_only',scope='post-exit case2 only; othercases use one unchanged main checkpoint',actor_inputs=33,missing_force_sentinel_preserved=True,weights_fitted=False,checkpoint=str(mirror_cp),checkpoint_sha256=hashlib.sha256(mirror_cp.read_bytes()).hexdigest(),kinematic_audit='mirror_audit_20261015_r1/result.json',formal_success_claim=False),indent=2))
        if args.mirror_dagger_fit:
            assert not args.mirror_teacher_probe
            plain_agent=json.loads((mirror_cp.parent/'agent.json').read_text())
            plain_runner=OnPolicyRunner(env,plain_agent,log_dir=None,device=env.device)
            plain_runner.load(str(mirror_cp),load_optimizer=False)
            raw.training_teacher_model=plain_runner.alg.policy
            raw.training_teacher_policy=plain_runner.get_inference_policy(device=env.device)
            for parameter in raw.training_teacher_model.parameters():parameter.requires_grad_(False)
            from safe_mirror_dagger import fit
            fit(env,raw,runner,logdir,args)
            env.seed(args.seed+10000);raw.completed_episodes=raw.completed_successes=0
        if args.mirror_teacher_probe:env.seed(args.seed+10000) # Constructor-independent paired physics/noise stream.
        if args.boundary_mode=='train':
            from boundary_rl import train as boundary_train
            boundary_train(env,raw,runner,logdir,args)
            return
        if args.joint5_input and not args.deep_ppo_updates and not args.critic_phase_pilot and not args.critic_domain_collect:
            from joint_input_upgrade import expose_joint5
            expose_joint5(raw.deep_recovery_model)
        policy=runner.get_inference_policy(device=env.device)
        if args.boundary_mode:
            assert args.hole_width_mm==14 and args.task_profile=='insertion_with_recovery'
            from boundary_rl import Controller
            boundary=Controller(raw,runner,args);policy=boundary.act_inference
            if args.independent_boundary_history:
                assert not args.post_ppo_updates and not args.history_proof
                from frozen_boundary_history import FrozenBoundaryHistory,precision
                precision();feature_history=FrozenBoundaryHistory(raw,boundary);policy=boundary.act_inference
        evaluation_policy_adapter=None
        if args.boundary_mode:
            assert not args.post_ppo_updates
            from qualified_policy_adapter import NativePolicy
            brake_hook=None
            if args.fixed_final_retry_brake:
                from brake_wrapper import BrakeHook
                brake_hook=BrakeHook(raw,boundary)
            evaluation_policy_adapter=NativePolicy(boundary,brake_hook)
            policy=evaluation_policy_adapter.act_inference
        if args.critic_domain_collect:
            from critic_conditioned_collect import collect as collect_critic_domain
            collect_critic_domain(env,raw,boundary,logdir,args)
            return
        if args.critic_phase_pilot:
            from critic_phase_pilot import train as train_critic_phase
            train_critic_phase(env,raw,boundary,logdir,args)
            return
        if args.deep_ppo_updates:
            from deep_joint_ppo import train as train_deep_joint
            train_deep_joint(env,raw,boundary,logdir,args)
            return
        if args.critic_shadow:
            from context_ppo import Shadow
            critic_shadow=Shadow(raw,boundary)
        if args.initial_case_replay:
            assert args.mode=='eval' and not args.initial_ppo_updates and not args.handoff_ppo_updates and not args.recurrent_alignment_updates and not args.post_ppo_updates
            from case_replay import Replay
            case_replay=Replay(raw,args.initial_case_replay,logdir)
        raw.alignment_axial_cap=args.alignment_axial_cap
        task["alignment_axial_cap"]=args.alignment_axial_cap
        (logdir/"task.json").write_text(json.dumps(task,indent=2))
        if args.history_proof:
            from history_proof import run as run_history_proof
            run_history_proof(env,raw,boundary,logdir,args)
            return
        if args.paired_diagnostic:
            from paired_diagnostic import run as run_paired
            run_paired(env,raw,boundary,logdir,args)
            return
        if args.post_ppo_updates:
            from recurrent_post import train as train_post
            train_post(env,raw,boundary,logdir,args)
            return
        if args.recurrent_alignment_updates:
            from recurrent_alignment import train as train_recurrent
            train_recurrent(env,raw,boundary,logdir,args)
            return
        if args.handoff_ppo_updates:
            from handoff_ppo import train as train_handoff
            train_handoff(env,raw,boundary,logdir,args)
            return
        if args.initial_ppo_updates:
            from initial_ppo import train as train_initial
            train_initial(env,raw,boundary,logdir,args)
            return
        if args.initial_trace:
            from initial_trace import Trace
            initial_trace=Trace(raw,logdir)
        if args.handoff_trace:
            from handoff_trace import HandoffTrace
            handoff_trace=HandoffTrace(raw,logdir)
        if args.withdrawal_probe:
            from withdrawal_probe import Probe
            withdrawal_probe=Probe(raw,boundary,logdir,args.withdrawal_detail)
        learned=evaluate(env,policy,label='ppo',episodes=args.eval_episodes)
        if args.withdrawal_probe:withdrawal_probe.finalize()
        if evaluation_policy_adapter is not None:evaluation_policy_adapter.finalize(logdir)
        if args.handoff_trace:handoff_trace.finalize()
        if args.initial_case_replay:case_replay.finalize()
        if args.initial_trace:initial_trace.finalize()
        if args.boundary_mode:boundary.finalize(logdir)
        if args.scene_profile=='challenge_cup':raw.save_pose_snapshot(logdir/'cad_ppo_pose_snapshot.json')
        results=dict(ppo=learned)
        if baseline is not None:results['baseline']=baseline
        results['ppo']['training_teacher_active']=args.mirror_teacher_probe
        if args.mirror_teacher_probe:results['ppo']['label']='frozen_mirrored_policy_diagnostic_only_not_single_student'
        results['ppo']['alignment_profile']={'tracking_mm':3,'step_mm':.5,'rotation_step_rad':.002,'band_mm':50}
        if args.task_profile=='insertion_with_recovery' and args.evaluate_without_recovery:
            raw.recovery_enabled=False
            env.seed(args.seed+10000)
            results['ppo_without_recovery']=evaluate(env,policy,label='ppo_without_recovery',episodes=args.eval_episodes)
            raw.recovery_enabled=True
        references=([args.compare_checkpoint] if args.compare_checkpoint else [])+args.reference_checkpoint
        for index,reference_path in enumerate(references):
            reference_agent_path=Path(reference_path).parent/'agent.json'
            reference_agent=json.loads(reference_agent_path.read_text())
            reference_runner=OnPolicyRunner(env,reference_agent,log_dir=None,device=env.device)
            with torch.inference_mode():
                reference_runner.load(reference_path,load_optimizer=False)
                if args.adapt_cad_normalization:adapt_cad_normalizers(reference_runner.alg.policy,cfg.geometry)
            env.seed(args.seed+10000)
            reference_policy=reference_runner.get_inference_policy(device=env.device)
            label='reference_ppo' if index==0 else f'reference_ppo_{index}'
            results[label]=evaluate(env,reference_policy,label=label,episodes=args.eval_episodes)
            results[label]['checkpoint']=reference_path
            results[label]['cad_normalization_adapted']=args.adapt_cad_normalization
        if args.task_profile=='insertion_with_recovery':raw.recovery_enabled=True
        (logdir/'evaluation.json').write_text(json.dumps(results,indent=2))

exit_code = 0
try:
    main()
except BaseException as error:
    traceback.print_exc()
    print('PIPER_RUN_FAILED', flush=True)
    STATUS_PATH.write_text(json.dumps(dict(status='failed',error=str(error),time=time.time()),indent=2))
    exit_code = 1
finally:
    # Some installed URDF GUI plugins hang during headless shutdown. The task
    # and evaluation records are already saved before this bounded cleanup.
    def shutdown_timeout():
        print('SHUTDOWN_TIMEOUT_AFTER_RESULTS_SAVED',flush=True)
        os._exit(exit_code)
    timer = threading.Timer(20.,shutdown_timeout)
    timer.daemon = True
    timer.start()
    try:
        app.close(wait_for_replicator=False)
    except SystemExit:
        pass
    finally:
        timer.cancel()
sys.exit(exit_code)
