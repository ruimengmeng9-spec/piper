"""Training-only group loss and predeclared worst-group epoch selection."""
import copy
import hashlib
import math
import random

import numpy as np
import torch
from torch import nn

from context_ppo import metrics, predict
from guarded_value import optimizer_digest, value_gate_passed


def phase_masks(phase):
    return [phase <= 1 / 3, (phase > 1 / 3) & (phase < 2 / 3), phase >= 2 / 3]


def training_weights(target, eid, phase):
    held=((eid//4)%4)==0
    train=~held
    assert int(held.sum())>200 and int(train.sum())>500
    native_count=int(eid.max())//128+1
    assert native_count in [8,16,24] and torch.unique(eid//128).tolist()==list(range(native_count))
    ntrain=int(train.sum()); rows=[]
    for domain in range(native_count):
        for name,mask in zip(['early','middle','late'],phase_masks(phase)):
            q=train&(eid//128==domain)&mask;n=int(q.sum())
            assert n>=2,'Undercovered training domain-phase group'
            y=target[q].double();ids,inverse,counts=torch.unique(eid[q],return_inverse=True,return_counts=True);k=len(ids)
            means=torch.zeros(k,device=y.device,dtype=torch.float64).scatter_add_(0,inverse,y)/counts
            mean=means.mean()
            second=torch.zeros(k,device=y.device,dtype=torch.float64).scatter_add_(0,inverse,(y-mean).square())/counts
            variance=float(second.mean());assert math.isfinite(variance) and variance>=0
            rows.append((domain,name,q,n,ids,inverse,counts,k,float(mean),variance))
    medians={name:float(np.median([x[-1] for x in rows if x[1]==name])) for name in ['early','middle','late']}
    assert all(math.isfinite(v) and v>=0 for v in medians.values())
    weight=torch.zeros_like(target);groups=[]
    for domain,name,q,n,ids,inverse,counts,k,mean,variance in rows:
        denominator=medians[name];phase_multiplier=3. if name=='late' else 2.
        mass=phase_multiplier*ntrain/(3*native_count*k*max(denominator,1e-6))
        raw=mass/counts[inverse].double();weight[q]=raw.to(weight.dtype)
        groups.append(dict(domain=domain,phase=name,phase_loss_multiplier=phase_multiplier,train_steps=n,independent_episodes=k,episode_ids=ids.cpu().tolist(),episode_selected_counts=counts.cpu().tolist(),train_target_variance=variance,normalization_variance=denominator,train_episode_balanced_mean=mean,per_episode_raw_mass=mass,raw_sample_weight_min=float(raw.min()),raw_sample_weight_max=float(raw.max())))
    normalizer=float(weight[train].double().mean());assert math.isfinite(normalizer) and normalizer>0
    weight/=normalizer
    assert torch.equal(weight[held],torch.zeros_like(weight[held])) and abs(float(weight[train].mean())-1)<1e-5
    return weight,dict(variance_normalization_protocol='TRAIN_phase_median_of_native_episode_variances',TRAIN_phase_variance_medians=medians,phase_relative_weights=[2.,2.,3.],native_domain_count=native_count,loss_group_count=3*native_count,groups=groups,raw_training_weight_mean=normalizer,episode_balanced=True,variance_definition='equal episode mean of within-episode squared deviation from equal-episode group mean; normalization uses median of native TRAIN variances within same phase',sample_weight_formula='phase_multiplier(early2,middle2,late3)*Ntrain/(3*native_count*Kgroup*selected_steps_episode_phase*max(TRAIN_phase_median_variance,1e-6)) then TRAINmean1',heldout_weights_zero=True,gradient_weights_use_training_targets_only=True)


def assess(prediction, target, eid, phase):
    held = ((eid // 4) % 4) == 0
    native_count = int(eid.max()) // 128 + 1
    assert native_count in [8,16,24] and torch.unique(eid // 128).tolist()==list(range(native_count))
    pooled = metrics(prediction, target, phase, held)
    domains = [metrics(prediction, target, phase, held & (eid // 256 == d)) for d in range(native_count//2)]
    blocks = [metrics(prediction, target, phase, held & (eid // 128 == d)) for d in range(native_count)]
    violations = []
    for report in [pooled] + domains + blocks:
        values = [report['mse_ratio'] / .9, report['residual_variance_ratio'] / .9]
        values += [v['mse_ratio'] / 1.05 for v in report['phases'].values()]
        assert all(math.isfinite(v) for v in values), 'Nonfinite heldout score or empty phase'
        violations.extend(values)
    return dict(pooled=pooled, domains=domains, blocks=blocks, maximum_gate_violation=max(violations),
                native_domain_count=native_count,assessed_groups=1+native_count//2+native_count,all_value_gates_passed=all(value_gate_passed(v) for v in [pooled] + domains + blocks))


def capture_rng():
    return dict(torch=torch.get_rng_state().clone(),
                cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                numpy=copy.deepcopy(np.random.get_state()), python=copy.deepcopy(random.getstate()))


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def rng_digest(state):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.cpu().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype), x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                add(v)
        else:
            h.update(repr(x).encode())
    add(state)
    return h.hexdigest()


def fit_value(model, opt, C, T, eid, phase, epochs=40):
    held = ((eid // 4) % 4) == 0
    train = ~held
    weight, info = training_weights(T, eid, phase)
    with torch.no_grad():
        before = assess(predict(model, C), T, eid, phase)
    state = copy.deepcopy(model.critic.state_dict())
    initial_state = copy.deepcopy(state)
    moment = copy.deepcopy(opt.state_dict())
    selected_rng = capture_rng()
    epoch0_rng_hash = rng_digest(selected_rng)
    epoch0_opt_hash = optimizer_digest(opt)
    best = before['maximum_gate_violation']
    best_epoch = steps = 0
    history = []
    for epoch in range(epochs):
        ids = train.nonzero().flatten()
        ids = ids[torch.randperm(len(ids), device=C.device)]
        for q in ids.split(1024):
            loss = (((predict(model, C[q]) - T[q]) / 100).square() * weight[q]).mean()
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), 1.)
            opt.step(); steps += 1
        with torch.no_grad():
            report = assess(predict(model, C), T, eid, phase)
        score = report['maximum_gate_violation']
        history.append(dict(epoch=epoch + 1, maximum_gate_violation=score,
                            pooled_mse=report['pooled']['mse']))
        if score < best:
            best = score; best_epoch = epoch + 1
            state = copy.deepcopy(model.critic.state_dict())
            moment = copy.deepcopy(opt.state_dict())
            selected_rng = capture_rng()
    model.critic.load_state_dict(state)
    opt.load_state_dict(moment)
    restore_rng(selected_rng)
    with torch.no_grad():
        after = assess(predict(model, C), T, eid, phase)
    assert after['maximum_gate_violation'] <= before['maximum_gate_violation']
    assert after['maximum_gate_violation'] == best
    assert rng_digest(capture_rng()) == rng_digest(selected_rng)
    if best_epoch == 0:
        assert all(torch.equal(v, initial_state[k]) for k, v in model.critic.state_dict().items())
        assert optimizer_digest(opt) == epoch0_opt_hash
        assert rng_digest(capture_rng()) == epoch0_rng_hash
    return dict(training_objective='equal24 native-phase means with train-only variance normalization',
                weights=info, before=before['pooled'], after=after['pooled'],
                before_assessment=before, after_assessment=after,
                best_epoch=best_epoch, optimizer_steps=steps, epochs=epochs,
                epoch_records=history, epoch0_candidate=True, retained_fit=best_epoch > 0,
                epoch0_optimizer_sha256=epoch0_opt_hash, retained_optimizer_sha256=optimizer_digest(opt),
                epoch0_rng_sha256=epoch0_rng_hash, retained_rng_sha256=rng_digest(capture_rng()),
                heldout_rule='complete env//4 modulo4 equals0;blind excluded from all fit and selection',
                strict_worst_group_selection=True, retained_matching_Adam_and_RNG=True)
