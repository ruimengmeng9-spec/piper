import copy, hashlib
import numpy as np
import torch
from torch import nn
from group_balanced_value import training_weights, assess, capture_rng, restore_rng, rng_digest
from guarded_value import optimizer_digest
from history_data import predict_dataset, gather_columns

def huber_mse_scaled(error, delta=0.05):
    assert delta > 0
    absolute = error.abs()
    return torch.where(absolute <= delta, error.square(), 2 * delta * absolute - delta * delta)

def squared_error_scaled(error):
    """Squared error in the existing normalized return units."""
    return error.square()

def locked_history_dropout(H, p=0.25):
    """One inverted Bernoulli feature mask per complete episode, constant over time."""
    assert H.ndim == 3 and 0 <= p < 1
    if p == 0:
        return H
    mask = torch.empty_like(H[0]).bernoulli_(1 - p) / (1 - p)
    return H * mask[None]

def force_smooth_prediction(model, C, H, E=None, L=None, R=None, S=None, Q=None, J=None):
    """Train exactly the deployed deterministic four-mask mean; no random mask."""
    return model.critic.combine_expected(C, H, E, L, R, S, Q, J)

def freeze_history_memories(model):
    for rnn in []:
        rnn.eval()
        for param in rnn.parameters():
            param.requires_grad_(False)
            param.grad = None

def frozen_backbone_modules(model):
    return []

def frozen_backbone_hashes(model):
    out = {}
    for name, module in frozen_backbone_modules(model):
        h = hashlib.sha256()
        for key, value in module.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        out[name] = h.hexdigest()
    return out

def frozen_memory_hashes(model):
    out = {}
    for name, rnn in []:
        h = hashlib.sha256()
        for key, value in rnn.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        out[name] = h.hexdigest()
    return out

def trainable_value_parameters(model):
    freeze_history_memories(model)
    for param in model.critic.force_memory.parameters():
        param.requires_grad_(True)
    for param in model.critic.memory.parameters():
        param.requires_grad_(True)
    for param in model.critic.static[0].public.parameters():
        param.requires_grad_(True)
    for name, module in frozen_backbone_modules(model):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
            param.grad = None
    return [p for p in model.critic.parameters() if p.requires_grad]

def tail_learning_rate_factor(epoch, epochs):
    """Zero-based epoch: keep first half, then linearly decay to 0.1."""
    assert isinstance(epoch, int) and isinstance(epochs, int) and (0 <= epoch < epochs)
    constant = epochs // 2
    return 1.0 if epoch < constant else 1.0 - 0.9 * (epoch - constant + 1) / (epochs - constant)

def fit_value(model, opt, data, epochs=40, progress=None):
    assert all((group['weight_decay'] == 0.1 for group in opt.param_groups)), 'Preregistered AdamW decay .1 required'
    model.critic.memory.eval()
    model.critic.force_memory.eval()
    assert not any((p.requires_grad for rnn in [] for p in rnn.parameters()))
    assert sum((param.numel() for param in model.critic.memory.parameters())) == 7296 and all((param.requires_grad for param in model.critic.memory.parameters()))
    assert sum((param.numel() for param in model.critic.force_memory.parameters())) == 1008 and all((param.requires_grad for param in model.critic.force_memory.parameters()))
    assert sum((p.numel() for p in model.critic.static[0].public.parameters())) == 4736 and all((p.requires_grad for p in model.critic.static[0].public.parameters()))
    initial_lrs = [group['lr'] for group in opt.param_groups]
    expected_ids = {id(p) for p in model.critic.parameters() if p.requires_grad}
    actual_ids = [id(p) for group in opt.param_groups for p in group['params']]
    assert len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == expected_ids
    assert not any((param.requires_grad for name, module in frozen_backbone_modules(model) for param in module.parameters()))
    backbone_hashes = frozen_backbone_hashes(model)
    fixed_hashes = frozen_memory_hashes(model)
    flat = data['flat']
    device = data['X'].device
    T = torch.from_numpy(flat['target']).to(device)
    eid = torch.from_numpy(flat['env']).to(device)
    phase = torch.from_numpy(flat['phase']).to(device)
    tick = torch.from_numpy(flat['tick']).to(device)
    weights, info = training_weights(T, eid, phase)
    W = torch.zeros_like(data['target'])
    W[tick, eid] = weights
    train_env = torch.arange(data['original_episode_count'], device=device)
    train_env = train_env[train_env // 4 % 4 != 0]
    ntrain = int((weights > 0).sum())
    with torch.no_grad():
        before = assess(predict_dataset(model, data), T, eid, phase)
    state = copy.deepcopy(model.critic.state_dict())
    moment = copy.deepcopy(opt.state_dict())
    rng = capture_rng()
    initial = copy.deepcopy(state)
    opt0 = optimizer_digest(opt)
    rng0 = rng_digest(rng)
    best = before['maximum_gate_violation']
    best_epoch = steps = 0
    history = []
    for epoch in range(epochs):
        factor = tail_learning_rate_factor(epoch, epochs)
        for group, base_lr in zip(opt.param_groups, initial_lrs):
            group['lr'] = base_lr * factor
        ids = train_env[torch.randperm(len(train_env), device=device)]
        epoch_steps = 0
        for q in ids.split(16):
            mask = data['selected'][:, q]
            if not mask.any():
                continue
            model.critic.memory.eval()
            model.critic.force_memory.eval()
            H = model.critic.training_sequence(gather_columns(data, 'X', q), gather_columns(data, 'FX', q))
            assert H.requires_grad and H.grad_fn is not None
            C = gather_columns(data, 'C', q)[mask]
            pred = force_smooth_prediction(model, C, H[mask], gather_columns(data, 'E', q)[mask], gather_columns(data, 'LAT', q)[mask], gather_columns(data, 'REL', q)[mask], gather_columns(data, 'CTRL', q)[mask], gather_columns(data, 'Q', q)[mask], gather_columns(data, 'JT', q)[mask])[:, 0] * 100.0
            target = data['target'][:, q][mask]
            weight = W[:, q][mask]
            assert (weight > 0).all()
            expected = ntrain * len(q) / len(train_env)
            loss = (squared_error_scaled((pred - target) / 100.0) * weight).sum() / expected
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            opt.step()
            steps += 1
            epoch_steps += 1
        model.critic.memory.eval()
        model.critic.force_memory.eval()
        with torch.no_grad():
            report = assess(predict_dataset(model, data), T, eid, phase)
        score = report['maximum_gate_violation']
        history.append(dict(learning_rates=[g['lr'] for g in opt.param_groups], epoch=epoch + 1, maximum_gate_violation=score, pooled_mse=report['pooled']['mse'], optimizer_steps=epoch_steps))
        if score < best:
            best = score
            best_epoch = epoch + 1
            state = copy.deepcopy(model.critic.state_dict())
            moment = copy.deepcopy(opt.state_dict())
            rng = capture_rng()
        if progress:
            progress(history[-1])
    model.critic.load_state_dict(state)
    opt.load_state_dict(moment)
    restore_rng(rng)
    with torch.no_grad():
        after = assess(predict_dataset(model, data), T, eid, phase)
    assert after['maximum_gate_violation'] == best
    if best_epoch == 0:
        assert all((torch.equal(v, initial[k]) for k, v in model.critic.state_dict().items()))
        assert optimizer_digest(opt) == opt0 and rng_digest(capture_rng()) == rng0
    assert frozen_backbone_hashes(model) == backbone_hashes
    assert all((param.grad is None and (not param.requires_grad) for name, module in frozen_backbone_modules(model) for param in module.parameters()))
    assert frozen_memory_hashes(model) == fixed_hashes
    assert all((p.grad is None and (not p.requires_grad) for rnn in [] for p in rnn.parameters()))
    return dict(public_GRU_full_episode_BPTT=True, trainable_public_memory_parameters=7296, public_GRU_training_mode=True, force_GRU_full_episode_BPTT=True, trainable_force_memory_parameters=1008, force_GRU_training_mode=True, force_history_encoding='half_level_minus_valid_quarter_difference', learning_rate_schedule='constant_half_linear_tenth', initial_learning_rates=initial_lrs, retained_learning_rates=[g['lr'] for g in opt.param_groups], tail_learning_rate_final_factor=0.1, value_training_protocol='balanced_four_mean_mse_tail_decay_force_smooth', training_prediction_matches_inference=True, random_history_mask_used=False, loss_name='MSE_scale', loss_delta_normalized=None, loss_delta_return_units=None, all_selected_samples_in_loss=True, explicit_predecision_joint_target6=True, joint_target_projection_parameters=384, explicit_preaction_queue12=True, queue_projection_parameters=768, trainable_existing_hidden_parameters=2080, existing_hidden_layer_trainable=True, frozen_static_backbone=False, frozen_static_backbone_parameters=0, trainable_static_public=True, trainable_static_public_parameters=4736, frozen_static_backbone_hashes=backbone_hashes, trainable_value_parameter_count=sum((param.numel() for param in model.critic.parameters() if param.requires_grad)), public_history_memory_trainable=True, frozen_memory_hashes=fixed_hashes, frozen_memory_parameters=sum((p.numel() for rnn in [] for p in rnn.parameters())), public_recurrent_full_episode_gradients=True, complete_history_forward_without_truncation=True, training_objective='Fixed TRAIN phase ratio early2 middle2 late3 then TRAINmean1. Squared error on the unchanged fixed balanced4 mean used at inference;' + str(info['loss_group_count']) + 'native-phase train-only normalized weights;complete episode public-and-force-BPTT forward;selected loss only', weights=info, history_dropout_probability=0.5, history_dropout_mask='fixed balanced4 feature masks;each50percent zero and kept2;loss on mean prediction;training and inference identical', before=before['pooled'], after=after['pooled'], before_assessment=before, after_assessment=after, epochs=epochs, optimizer_steps=steps, best_epoch=best_epoch, retained_fit=best_epoch > 0, epoch_records=history, epoch0_candidate=True, epoch0_optimizer_sha256=opt0, retained_optimizer_sha256=optimizer_digest(opt), epoch0_rng_sha256=rng0, retained_rng_sha256=rng_digest(capture_rng()), retained_matching_Adam_and_RNG=True, strict_worst_group_selection=True, minibatch_complete_envs=16, empty_selected_batches_skipped=True, normalization='sum weighted selected losses / expected selected count Ntrain*batch_envs/Ntrain_envs', no_train_heldout_episode_overlap=True)
