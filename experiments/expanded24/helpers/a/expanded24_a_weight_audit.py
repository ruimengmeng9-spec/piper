"""Independent NumPy-only two-pass TRAIN phase-median weight oracle."""
import numpy as np

def audit_episode_weights(flat,info,actual=None):
    t=flat['target'];eid=flat['env'];phase=flat['phase'];train=((eid//4)%4)!=0;ntrain=int(train.sum());native=int(eid.max())//128+1
    assert info['phase_relative_weights']==[2.,2.,3.]
    assert info['variance_normalization_protocol']=='TRAIN_phase_median_of_native_episode_variances'
    assert native in [8,16,24] and info['episode_balanced'] and info['loss_group_count']==3*native and len(info['groups'])==3*native
    rows=[]
    for domain in range(native):
        for name,pm in [('early',phase<=1/3),('middle',(phase>1/3)&(phase<2/3)),('late',phase>=2/3)]:
            q=train&(eid//128==domain)&pm;y=t[q].astype('float64');ids,inv,counts=np.unique(eid[q],return_inverse=True,return_counts=True)
            assert q.sum()>=2 and len(ids)>0
            mean=float(np.mean([y[inv==i].mean() for i in range(len(ids))]))
            variance=float(np.mean([np.mean((y[inv==i]-mean)**2) for i in range(len(ids))]))
            rows.append((domain,name,q,ids,inv,counts,mean,variance))
    # Even-sized median is the mean of the two middle values; independent of Torch implementation.
    medians={}
    for name in ['early','middle','late']:
        v=sorted(x[-1] for x in rows if x[1]==name)
        medians[name]=float((v[native//2-1]+v[native//2])/2)
        assert np.isclose(medians[name],info['TRAIN_phase_variance_medians'][name],rtol=1e-9,atol=1e-8)
    raw_weight=np.zeros_like(t)
    for seen,(domain,name,q,ids,inv,counts,mean,variance) in enumerate(rows):
        denominator=medians[name];multiplier=3. if name=='late' else 2.;mass=multiplier*ntrain/(3*native*len(ids)*max(denominator,1e-6));raw=mass/counts[inv]
        r=info['groups'][seen]
        assert r['phase_loss_multiplier']==multiplier
        assert r['domain']==domain and r['phase']==name and r['train_steps']==int(q.sum()) and r['independent_episodes']==len(ids) and r['episode_ids']==ids.tolist() and r['episode_selected_counts']==counts.tolist()
        for a,b in [(variance,r['train_target_variance']),(denominator,r['normalization_variance']),(mean,r['train_episode_balanced_mean']),(mass,r['per_episode_raw_mass']),(float(raw.min()),r['raw_sample_weight_min']),(float(raw.max()),r['raw_sample_weight_max'])]:assert np.isclose(a,b,rtol=1e-9,atol=1e-8),(a,b)
        raw_weight[q]=raw.astype(raw_weight.dtype)
        sums=np.array([raw_weight[q][inv==i].astype('float64').sum() for i in range(len(ids))]);assert np.allclose(sums,mass,rtol=2e-6,atol=1e-6)
    normalizer=float(raw_weight[train].astype('float64').mean());assert np.isclose(normalizer,info['raw_training_weight_mean'],rtol=2e-7,atol=1e-8)
    expected=raw_weight/normalizer;assert not expected[~train].any() and np.isclose(expected[train].mean(),1.,rtol=2e-6)
    if actual is not None:assert np.allclose(actual,expected,rtol=3e-6,atol=1e-7)
    return dict(status='passed',TRAIN_phase_force_smoothalization_independently_verified=True,fixed_TRAIN_phase_223_loss_independently_verified=True,groups=len(rows),TRAIN_steps=ntrain,HELD_weights_zero=True,equal_episode_phase_mass_verified=True,TRAIN_only_balanced_variance_independently_recomputed=True)
