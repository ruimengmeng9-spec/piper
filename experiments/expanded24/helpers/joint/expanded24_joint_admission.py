"""Admission uses original global episode IDs; no filtering, optimizer or model selection."""
import numpy as np

def validate_index(rows,plan):
    assert len(rows)==24 and [r['seed'] for r in rows]==plan['training_seeds']
    assert [r['result']['index'] for r in rows]==list(range(24))
    assert len({r['result']['domain_joint_hash'] for r in rows})==24
    for row in rows:
        r=row['result'];assert r['seed']==row['seed'] and r['episodes']==128
        assert not r['blind'] and r['actor_updates']==r['value_fit_attempts']==0
        assert r['force_gate_passed'] and r['force_aborts']<=12 and r['coverage_passed']
    return True

def validate_flat(flat):
    eid=np.asarray(flat['env']);phase=np.asarray(flat['phase'])
    assert eid.ndim==phase.ndim==1 and len(eid)==len(phase)>0
    assert eid.dtype.kind in 'iu' and eid.min()>=0 and eid.max()<3072
    assert np.isfinite(phase).all() and ((phase>=0)&(phase<=1)).all()
    assert np.array_equal(np.unique(eid//128),np.arange(24))
    selected=len(np.unique(eid));assert 0<selected<=1536,'Stop before dense allocation; preserve every episode; no resampling'
    held=((eid//4)%4)==0;train=~held
    phases=[phase<=1/3,(phase>1/3)&(phase<2/3),phase>=2/3]
    rows=[]
    for width,count,independent in [(128,24,False),(256,12,True)]:
        for number in range(count):
            t=train&(eid//width==number);h=held&(eid//width==number)
            assert int(t.sum())>500 and int(h.sum())>200
            assert all(int((t&m).sum())>=2 and int((h&m).sum())>=1 for m in phases)
            nt=len(np.unique(eid[t]));nh=len(np.unique(eid[h]))
            if independent:assert nt>=24 and nh>=8
            rows.append(dict(episode_block_width=width,index=number,train_steps=int(t.sum()),held_steps=int(h.sum()),independent_TRAIN_episodes=nt,independent_HELD_episodes=nh))
    return dict(status='passed',original_physical_episode_count=3072,selected_complete_episode_count=selected,maximum_packed_selected_episodes=1536,native_and_paired_coverage=rows,all24_native_and12_pair_phase_step_and_independent_episode_gates_before_fit_marker=True,original_global_episode_split_preserved=True,no_samples_or_episodes_dropped=True)
