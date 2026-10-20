import copy,hashlib,io,json,sys,tarfile
from pathlib import Path
import numpy as np
import torch
L=Path('/workspace/piper-insertion/logs');E='reliability_v2_expanded24_a_native_20261020';D=L/E;S=L/(E+'_source/executable');sys.path.insert(0,str(S))
from boundary_rl import Residual
from domain_value_upgrade import upgrade_critic
from history_value import upgrade_history,history_inputs
from history_data import load_history
from expanded24_a_volume_checks import history,verify_offset_metadata
from frozen_boundary_history import precision
precision();torch.set_num_threads(2);A=L/'reliability_v2_shrinkage_a_native_20261018_review.tar.gz';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(A)==json.loads((D/'plan.json').read_text())['known_equivalence_archive_sha256'];F=L/(E+'_release_random');r=json.loads((F/'domain_collect_result.json').read_text());probe=json.loads((F/'withdrawal_probe.json').read_text());assert r['history_enabled'] and r['training_forbidden'] and r['actor_updates']==r['value_fit_attempts']==0
with tarfile.open(A) as z:
 prefix='reliability_v2_shrinkage_a_native_20261018_release_random/';old=json.load(z.extractfile(prefix+'domain_collect_result.json'));op=json.load(z.extractfile(prefix+'withdrawal_probe.json'))
 for k in ['physical_per_env_sha256','feature_per_env_sha256']:assert probe[k]==op[k]
 for k in ['value_per_env_sha256','context73_per_env_sha256','domain_joint_hash','frozen_role_hashes','deep_state_hash']:assert r[k]==old[k]
 assert z.extractfile(prefix+'episode_outcomes.jsonl').read()==(F/'episode_outcomes.jsonl').read_bytes()
 for name in ['native_probes.npz','selected_data.npz','domain_parameters.npz','case_metadata.npz']:
  a=np.load(io.BytesIO(z.extractfile(prefix+name).read()));b=np.load(F/name)
  for k in a.files:assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(),k
 oldfx,oldfa,oldfd=history(z.extractfile(prefix+'complete_force_history.npz').read(),3);newfx,newfa,newfd=history((F/'complete_force_history.npz').read_bytes(),3);assert oldfx.tobytes()==newfx.tobytes() and np.array_equal(oldfa,newfa) and np.array_equal(oldfd,newfd)
 oldx,olda,oldd=history(z.extractfile(prefix+'complete_history.npz').read());newx,newa,newd=history((F/'complete_history.npz').read_bytes());assert oldx.tobytes()==newx.tobytes() and np.array_equal(olda,newa) and np.array_equal(oldd,newd)
verify_offset_metadata(np.load(F/'case_metadata.npz'),r,5)
state=torch.load(L/'reliability_v2_boundary_pilot1_random/boundary_last.pt',map_location='cpu',weights_only=False)['state_dict'];m=Residual();m.load_state_dict(state);upgrade_critic(m);upgrade_history(m);m.eval();x,alive,done=load_history(F/'complete_history.npz');q=np.load(F/'selected_data.npz');C=torch.from_numpy(q['critic_features']);tick=q['tick'];env=q['env'];assert np.array_equal(x[tick,env],history_inputs(torch.from_numpy(q['critic_features']),torch.cat((torch.zeros(len(tick),24),torch.from_numpy(q['public_force3']),torch.zeros(len(tick),6)),-1)).numpy())
with torch.no_grad():H=m.critic.sequence(torch.from_numpy(x),torch.from_numpy(load_history(F/'complete_force_history.npz')[0]));value=m.critic.combine_expected(C,H[tick,env],torch.from_numpy(q['critic_privileged6']),torch.from_numpy(q['critic_latency1']),torch.from_numpy(q['critic_release1']),torch.from_numpy(q['critic_control4']),torch.from_numpy(q['critic_queue12']),torch.from_numpy(q['critic_joint_target6']))[:,0]*100.
err=float(abs(value-torch.from_numpy(q['prior_value'])).max());assert err<1e-3
v=dict(status='passed',old_archive_sha256=sha(A),baseline='previously dual-verified force42history/static75 known1125 cohort128;no fresh baseline rerun',all128_physical_action_feature_hashes_exact=True,all128_complete_force42_histories_exact=True,all4_diagnostic_cases_unchanged=True,all128_context73_plus_fixed_domain2_value_hashes_exact=True,all_original_known1125_outcomes_probes_data_exact=True,complete42_histories_lossless_and_selected_semantics_exact=True,CPU_prior_errors=[err],this_turn_diagnostic_episodes=128,valid_new_history_episodes=128,archived_reference_episodes=128,actor_updates=0,value_fit_attempts=0,not_independent_acceptance=True,training_authorized_only_after_local_verification=True);(D/'equivalence_result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
