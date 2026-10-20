"""One complete frozen-policy native cohort in a fresh physical domain."""
import copy, hashlib, json, random, shutil, time
from pathlib import Path
import numpy as np
import torch
from boundary_rl import model_hash
from context_ppo import Shadow, metrics, predict, sha, utc
from frozen_boundary_history import FrozenBoundaryHistory, precision
from joint_input_upgrade import expose_joint5
from deep_joint_hook import DeepHook
from brake_wrapper import BrakeHook
from native_groups import arrays
from joint_storage import cpu, MINFREE
from guarded_value import value_gate_passed
from domain_value_upgrade import upgrade_critic
from conditioned_shadow import ConditionedShadow
from withdrawal_probe import Probe
from history_value import upgrade_history, history_inputs
from history_shadow import HistoryShadow
from history_data import save_history, save_lzma_npz
SEEDS = [2811, 2813, 2815, 2817, 2819, 2821, 2823, 2825, 2827, 2829, 2831, 2833, 2835, 2837, 2839, 2841, 2843, 2845, 2847, 2849, 2851, 2853, 2855, 2857, 2873, 2889]
DOMAIN_NAMES = ['friction_scales', 'stiffness_scales', 'peg_masses', 'grip_axial_offsets', 'force_noise_scales']

def collect(env, raw, boundary, logdir, args):
    assert args.critic_domain_collect and (args.seed in SEEDS or (args.critic_equivalence and args.seed == 1125)) and args.joint5_input and (not args.deep_ppo_updates) and (not args.critic_phase_pilot)
    assert args.fixed_contact_cases and args.boundary_mode == 'eval' and (raw.num_envs == 128) and (raw.cfg.episode_length_s == 12) and (not raw.cfg.train_stop_failed_retry)
    assert not any([args.initial_case_replay, args.post_ppo_updates, args.critic_shadow, args.paired_diagnostic, args.fixed_final_retry_brake, args.independent_boundary_history])
    precision()
    D = Path(logdir)
    equiv = args.critic_equivalence
    conditioned = args.critic_conditioning == 'on'
    assert conditioned or equiv
    index = 0 if equiv else SEEDS.index(args.seed)
    blind = not equiv and index >= 24
    start = time.monotonic()
    source = Path(args.deep_recovery_checkpoint)
    assert sha(source) == '055571a29e42fc6e36b347823cb5aa57e6ea1505af6fb59226ab5108d33eabdf'
    assert args.case_profile == ('lateral_focus' if not equiv and (not blind) else 'all')
    expected_offset_mm = 5.0
    assert float(args.recovery_offset_mm) == expected_offset_mm and args.recovery_offset_min_mm is None
    assert float(raw.cfg.recovery_offset) == expected_offset_mm / 1000.0 and raw.cfg.recovery_offset_min < 0
    assert float(args.recovery_center_x_mm) == -4.5 and float(raw.cfg.recovery_center_x) == -0.0045
    cases = raw.case_ids.detach().clone()
    eid = torch.arange(128, device=raw.device)
    expected = torch.where(eid // 16 % 2 == 1, eid % 2 + 2, eid % 4) if not equiv and (not blind) else eid % 4
    assert torch.equal(cases, expected)
    model = raw.deep_recovery_model
    original = torch.load(source, map_location=raw.device, weights_only=False)['model_state_dict']
    assert all((torch.equal(v, original[k]) for k, v in model.state_dict().items()))
    root = copy.deepcopy(model).eval()
    expose_joint5(model)
    if conditioned:
        upgrade_critic(boundary.model)
    if args.critic_history:
        upgrade_history(boundary.model)
    if args.conditioned_critic_checkpoint:
        assert conditioned and blind and (not equiv)
        candidate = torch.load(args.conditioned_critic_checkpoint, map_location=raw.device, weights_only=False)
        assert candidate['force_history_encoding'] == 'half_level_minus_valid_quarter_difference'
        assert candidate['value_training_protocol'] == 'balanced_four_mean_mse_tail_decay_force_smooth'
        assert candidate['value_prediction_protocol'] == 'balanced_four_history_masks_p50' and candidate['value_inference_masks'] == 4
        boundary.model.load_state_dict(candidate['state_dict'])
    for m in [root, model, boundary.model]:
        for p in m.parameters():
            p.requires_grad_(False)
    initial = cpu(model.state_dict())
    frozen = {n: model_hash(m) for n, m in boundary.models().items()}
    boundary_hash = model_hash(boundary.model)
    bridge = FrozenBoundaryHistory(raw, boundary)
    shadow = (HistoryShadow if args.critic_history else ConditionedShadow if conditioned else Shadow)(raw, boundary)
    hook = DeepHook(raw, boundary, root)
    brake = BrakeHook(raw, boundary, hook, release_on=args.final_retry_release)
    raw.fixed_final_retry_hook = brake
    probe_logger = Probe(raw, boundary, D, False) if equiv else None
    value_hashes = [hashlib.sha256() for _ in range(128)]
    context_hashes = [hashlib.sha256() for _ in range(128)]
    context75_hashes = [hashlib.sha256() for _ in range(128)]
    domain = {n: cpu(getattr(raw, n)).numpy() for n in DOMAIN_NAMES}
    domain_hashes = {n: hashlib.sha256(v.tobytes()).hexdigest() for n, v in domain.items()}
    save_lzma_npz(D / 'domain_parameters.npz', **domain)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    obs, _ = env.reset()
    boundary.reset(torch.ones(128, dtype=torch.bool, device=raw.device))
    assert not bridge.h.any()
    active = torch.ones(128, dtype=torch.bool, device=raw.device)
    hook.active = active.clone()
    records = []
    ends = {}
    reset_offsets = raw.training_episode_offset.detach().clone()
    assert reset_offsets.shape == (128,) and torch.equal(reset_offsets, torch.full_like(reset_offsets, expected_offset_mm / 1000.0))
    save_lzma_npz(D / 'case_metadata.npz', case_ids=cases.cpu().numpy(), reset_offset_m=reset_offsets.cpu().numpy(), configured_offset_mm=np.array(expected_offset_mm, dtype=np.float32), action_delay_steps=raw.action_delay.steps.detach().cpu().numpy())
    (D / 'domain_progress.json').write_text(json.dumps(dict(phase='collecting', seed=args.seed, index=index, blind=blind, actor_updates=0), indent=2))
    for tick in range(730):
        alive = active.clone()
        with torch.no_grad():
            action = boundary.act_inference(obs)
            obs, reward, done, info = env.step(action)
            done = done.bool()
            term = info.get('terminal', {})
            for j, i in enumerate(term.get('env_ids', torch.empty(0, dtype=torch.long)).cpu().tolist()):
                if bool(alive[i]):
                    ends[i] = dict(env=index * 128 + i, native_env=i, seed=args.seed, success=bool(term['success'][j]), end_reason=int(term['end_reason'][j]), peak_force_n=float(term['peak_force_n'][j]), seconds=float(term['seconds'][j]), attempts=int(term['recovery_attempts'][j]), exits=int(term['recovery_exits'][j]), realignments=int(term['recovery_realignments'][j]), end_mode=int(term['recovery_end_mode'][j]))
            records.append((hook.cache, shadow.feature.clone(), shadow.value.clone(), reward.float().clone() * alive, done.clone(), alive, boundary.public.clone(), hook.compose_cache, hook.compose_cache['clipped'], hook.compose_cache['delayed'], shadow.extra6.clone(), shadow.truth6.clone(), shadow.latency1.clone(), shadow.release1.clone(), shadow.control4.clone(), shadow.queue12.clone(), shadow.joint_target6.clone(), shadow.force_encoding.clone()))
            if equiv:
                values = shadow.value.cpu().numpy()
                core = shadow.feature[:, :73].contiguous().cpu().numpy()
                for i in range(128):
                    value_hashes[i].update(values[i].tobytes())
                    context_hashes[i].update(core[i].tobytes())
                    context75_hashes[i].update(shadow.feature[i].cpu().numpy().tobytes())
            boundary.reset(done)
            active &= ~done
            hook.active = active.clone()
            if args.critic_history:
                shadow.reset(done)
                shadow.active = active.clone()
        if not active.any():
            break
    assert not active.any() and len(ends) == 128 and (not brake.brake.latched.any())
    rows = [ends[i] for i in range(128)]
    (D / 'episode_outcomes.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in rows)))
    M = torch.stack([r[0]['selected'] & r[5] for r in records])
    C = torch.stack([r[1] for r in records])
    V = torch.stack([r[2] for r in records])
    rewards = torch.stack([r[3] for r in records])
    done = torch.stack([r[4] for r in records])
    returns = torch.zeros_like(V)
    future = torch.zeros(128, device=raw.device)
    phase = torch.zeros_like(V)
    for t in reversed(range(len(records))):
        future = rewards[t] + 0.99 * future * ~done[t]
        returns[t] = future
    for e in range(128):
        q = M[:, e].nonzero().flatten()
        phase[q, e] = torch.arange(len(q), device=raw.device) / max(len(q) - 1, 1)
    eid = (torch.arange(128, device=raw.device) + index * 128)[None, :].expand_as(M)[M]
    held = eid // 4 % 4 == 0
    counts = M.sum(0).cpu().tolist()
    extra = sorted([e for e in range(128) if e // 4 % 4 == 0 and counts[e] > 0], key=lambda e: (-counts[e], e))[:4]
    probes = [0, 1, 2, 3, 32, 33, 34, 35]
    probes += [e for e in extra if e not in probes]
    probe = torch.tensor(probes, device=raw.device)
    from history_value import encoded_force_sequence
    force_difference = torch.stack([r[17] for r in records])
    raw_forces = torch.stack([r[11][..., 3:] for r in records])
    force_alive = torch.stack([r[5] for r in records])
    expected_difference = encoded_force_sequence(raw_forces)
    assert torch.equal(force_difference[force_alive], expected_difference[force_alive])
    difference_hashes = [hashlib.sha256(force_difference[:, e][force_alive[:, e]].contiguous().cpu().numpy().tobytes()).hexdigest() for e in range(128)]
    public = torch.stack([r[6] for r in records])
    extra6 = torch.stack([r[10] for r in records])
    truth6 = torch.stack([r[11] for r in records])
    latency1 = torch.stack([r[12] for r in records])
    delays = torch.stack([r[7]['delay'] for r in records])
    release1 = torch.stack([r[13] for r in records])
    release_hold = torch.stack([r[7]['release_hold'] for r in records])
    assert torch.equal(release1[..., 0], release_hold.float() * 0.125)
    control4 = torch.stack([r[14] for r in records])
    queue12 = torch.stack([r[15] for r in records])
    raw_queue12 = torch.stack([r[7]['preaction_queue12'] for r in records])
    live = torch.stack([r[5] for r in records])
    assert torch.equal(queue12[live], raw_queue12[live])
    raw_queue_hashes = [hashlib.sha256(raw_queue12[:, e][live[:, e]].contiguous().cpu().numpy().tobytes()).hexdigest() for e in range(128)]
    assert raw_queue_hashes == [hashlib.sha256(queue12[:, e][live[:, e]].contiguous().cpu().numpy().tobytes()).hexdigest() for e in range(128)]
    supervisor_force = torch.stack([r[7]['supervisor_force_ema'] for r in records])
    supervisor_trigger = torch.stack([r[7]['supervisor_trigger_depth'] for r in records])
    assert torch.equal(control4, torch.cat((supervisor_force * 0.01, supervisor_trigger[..., None] * 50.0), -1))
    joint_target6 = torch.stack([r[16] for r in records])
    raw_j6 = torch.stack([r[7]['preaction_joint_target6'] for r in records])
    raw_targets = torch.stack([r[7]['raw_preaction_target6'] for r in records])
    raw_joints = torch.stack([r[7]['raw_preaction_joint6'] for r in records])
    assert torch.equal(joint_target6[live], raw_j6[live])
    assert torch.equal(raw_joints[live], public[..., 6:12][live])
    j6_hashes = [hashlib.sha256(raw_j6[:, e][live[:, e]].contiguous().cpu().numpy().tobytes()).hexdigest() for e in range(128)]
    fixed_probe = probe
    save_history(D / 'complete_history.npz', history_inputs(C, public), torch.stack([r[5] for r in records]), done)
    save_history(D / 'complete_force_history.npz', truth6[..., 3:], torch.stack([r[5] for r in records]), done)
    save_lzma_npz(D / 'native_probes.npz', joint_target_probe_ids=fixed_probe.cpu().numpy(), raw_preaction_joint_target6=raw_targets[:, fixed_probe].contiguous().cpu().numpy(), **arrays(records, probe, hook, brake), critic_control4=control4[:, probe].cpu().numpy(), supervisor_force_ema=supervisor_force[:, probe].cpu().numpy(), supervisor_trigger_depth=supervisor_trigger[:, probe].cpu().numpy(), critic_release1=release1[:, probe].cpu().numpy(), supervisor_release_hold=release_hold[:, probe].cpu().numpy(), critic_latency1=latency1[:, probe].cpu().numpy(), critic_privileged6=extra6[:, probe].cpu().numpy(), privileged_truth6=truth6[:, probe].cpu().numpy())
    save_lzma_npz(D / 'selected_data.npz', critic_joint_target6=joint_target6[M].contiguous().cpu().numpy(), critic_queue12=queue12[M].cpu().numpy(), critic_control4=control4[M].cpu().numpy(), supervisor_force_ema=supervisor_force[M].cpu().numpy(), supervisor_trigger_depth=supervisor_trigger[M].cpu().numpy(), critic_release1=release1[M].cpu().numpy(), supervisor_release_hold=release_hold[M].cpu().numpy(), critic_latency1=latency1[M].cpu().numpy(), action_delay_steps=delays[M].cpu().numpy(), critic_privileged6=extra6[M].cpu().numpy(), privileged_truth6=truth6[M].cpu().numpy(), public_pose3=public[M][:, :3].cpu().numpy(), critic_features=C[M].cpu().numpy(), public_force3=public[M][:, 24:27].cpu().numpy(), case_id=cases[None].expand_as(M)[M].cpu().numpy(), prior_value=V[M].cpu().numpy(), target=returns[M].cpu().numpy(), phase=phase[M].cpu().numpy(), env=eid.cpu().numpy(), tick=torch.arange(len(records), device=raw.device)[:, None].expand_as(M)[M].cpu().numpy())
    deep_train_episodes = int(((M.sum(0) > 0) & ((torch.arange(128, device=raw.device) + index * 128) // 4 % 4 != 0)).sum())
    deep_held_episodes = int(((M.sum(0) > 0) & ((torch.arange(128, device=raw.device) + index * 128) // 4 % 4 == 0)).sum())
    episode_coverage = deep_train_episodes >= 24 and deep_held_episodes >= 8
    score = metrics(V[M], returns[M], phase[M], held) if held.any() else None
    coverage = int(held.sum()) > 200 and int((~held).sum()) > 500
    forces = sum((r['peak_force_n'] > 100 for r in rows))
    bridge.verify()
    assert {n: model_hash(m) for n, m in boundary.models().items()} == frozen
    assert model_hash(boundary.model) == boundary_hash
    assert all((torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items()))
    assert all((np.array_equal(cpu(getattr(raw, n)).numpy(), domain[n]) for n in DOMAIN_NAMES))
    assert not any((p.requires_grad for m in [root, model, boundary.model] for p in m.parameters()))
    torch.save(dict(final_retry_release=bool(args.final_retry_release), deep_state_dict=initial, boundary_state_dict=cpu(boundary.model.state_dict()), domain={n: cpu(getattr(raw, n)) for n in DOMAIN_NAMES}, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate(), all_actor_and_critic_frozen=True), D / 'frozen_snapshot.pt')
    if probe_logger:
        probe_logger.finalize()
    result = dict(force_history_encoding='half_level_minus_valid_quarter_difference', force_encoding_online_raw_history_exact=True, force_encoding_per_env_sha256=difference_hashes, value_training_protocol='balanced_four_mean_mse_tail_decay_force_smooth', value_prediction_protocol='balanced_four_history_masks_p50', explicit_predecision_joint_target6=True, joint_target_native_envs=128, joint_target_all128_live_shadow_mirror_exact=True, joint_positions_all128_match_raw_public=True, joint_target_all128_live_sha256=hashlib.sha256(''.join(j6_hashes).encode()).hexdigest(), selected_raw_joint_target6_sha256=hashlib.sha256(raw_j6[M].contiguous().cpu().numpy().tobytes()).hexdigest(), raw_probe_joint_targets_sha256=hashlib.sha256(raw_targets[:, fixed_probe].contiguous().cpu().numpy().tobytes()).hexdigest(), joint_target_selected_unique_counts=[int(torch.unique(joint_target6[M][:, j]).numel()) for j in range(6)], joint_target_selected_max_abs=[float(joint_target6[M][:, j].abs().max()) for j in range(6)], explicit_preaction_queue12=True, all128_live_queue_raw_mirror_exact=True, raw_queue_envs=128, raw_queue_all128_sha256=hashlib.sha256(''.join(raw_queue_hashes).encode()).hexdigest(), raw_queue_probe_live_sha256={str(int(e)): raw_queue_hashes[int(e)] for e in probes}, selected_raw_queue_sha256=hashlib.sha256(raw_queue12[M].contiguous().cpu().numpy().tobytes()).hexdigest(), final_retry_release=bool(args.final_retry_release), controller_policy_changed=bool(args.final_retry_release), coverage_scope='native128_steps;independent24train8held_on_predeclared_pair256', logical_training_domain=index // 2 if not blind and (not equiv) else None, configured_offset_mm=expected_offset_mm, reset_offset_sha256=hashlib.sha256(reset_offsets.cpu().numpy().tobytes()).hexdigest(), case_profile=args.case_profile, case_ids_sha256=hashlib.sha256(cases.cpu().numpy().tobytes()).hexdigest(), deep_training_episodes=deep_train_episodes, deep_heldout_episodes=deep_held_episodes, independent_episode_coverage_passed=episode_coverage, history_enabled=args.critic_history, complete_history42=True, complete_trueforce3_history=True, explicit_preaction_latency1=True, explicit_preaction_release1=True, explicit_preaction_controller4=True, context75_per_env_sha256=[h.hexdigest() for h in context75_hashes] if equiv else None, conditioned=conditioned, critic_dimensions=75 if conditioned else 73, equivalence_diagnostic=equiv, training_forbidden=equiv, value_per_env_sha256=[h.hexdigest() for h in value_hashes] if equiv else None, context73_per_env_sha256=[h.hexdigest() for h in context_hashes] if equiv else None, conditioned_checkpoint_sha256=sha(args.conditioned_critic_checkpoint) if args.conditioned_critic_checkpoint else None, status='collected', seed=args.seed, index=index, blind=blind, episodes=128, successes=sum((r['success'] for r in rows)), force_aborts=forces, force_gate_passed=forces <= 12, coverage_passed=coverage, heldout_steps=int(held.sum()), train_steps=int((~held).sum()), selected_steps=int(M.sum()), selected_episodes=int((M.sum(0) > 0).sum()), selected_steps_by_native_env=counts, selected_steps_by_case={str(c): sum((counts[e] for e in range(128) if int(cases[e]) == c)) for c in range(4)}, native_ticks=len(records), native_live_steps=int(torch.stack([r[5] for r in records]).sum()), prior_metrics=score, value_gate_passed=bool(score and value_gate_passed(score)), blind_gate_passed=bool(blind and forces <= 12 and coverage and score and value_gate_passed(score)), actor_updates=0, value_fit_attempts=0, frozen_role_hashes=frozen, boundary_model_hash=boundary_hash, deep_state_hash=model_hash(model), domain_hashes=domain_hashes, domain_joint_hash=hashlib.sha256(''.join(domain_hashes.values()).encode()).hexdigest(), boundary_checkpoint_sha256=sha(args.boundary_checkpoint), source_deep_sha256=sha(source), new_joint_column_max_abs=float(model.actor[0].weight[:, 10].abs().max()), all_actor_and_critic_frozen=True, wall_seconds=time.monotonic() - start)
    assert result['new_joint_column_max_abs'] == 0
    assert shutil.disk_usage(D).free >= MINFREE
    (D / 'domain_collect_result.json').write_text(json.dumps(result, indent=2))
    (D / 'domain_progress.json').write_text(json.dumps(dict(phase='completed', seed=args.seed, episodes=128, actor_updates=0), indent=2))
    print('CRITIC_DOMAIN_COLLECT', json.dumps(result), flush=True)
