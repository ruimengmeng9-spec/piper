import shutil as _storage_shutil
def _namespace_free_bytes(target):
    assert _storage_shutil.disk_usage('/data/shijinsheng/piper-insertion').free>=5045649408,'Original data filesystem reserve must remain intact'
    return _storage_shutil.disk_usage(target).free
import argparse,datetime as dt,fcntl,hashlib,json,shutil,subprocess,time,traceback
from pathlib import Path
R=Path('/home/shijinsheng/piper-training-workspace-20261019');W='/workspace/piper-insertion';I='sha256:0c1687cdae03921a28307bb3c0471174b69d37741c3cbb9a68cfd4e1640fbba8'
MINFREE=max(4*1024**3,6656262144-int(1.5*1024**3))
P=argparse.ArgumentParser();P.add_argument('--release-guard',action='store_true');P.add_argument('--offset-mm',type=int,choices=[4,5],default=5);P.add_argument('--case-profile',choices=['all','negative_lateral','lateral_focus'],default='all');P.add_argument('--history',action='store_true');P.add_argument('--conditioning',choices=['off','on'],default='on');P.add_argument('--equivalence',action='store_true');P.add_argument('--conditioned-critic');P.add_argument('--deep-updates',type=int,default=0);P.add_argument('--withdrawal-detail',action='store_true');P.add_argument('--fixed-brake',action='store_true');P.add_argument('--proof',choices=['root','bridge']);P.add_argument('--feature-history',action='store_true');P.add_argument('--paired',action='store_true');P.add_argument('--pair-scale',type=float,default=1.);P.add_argument('--critic-shadow',action='store_true');P.add_argument('--post-updates',type=int,default=0);P.add_argument('--case-replay',action='store_true');P.add_argument('--axial-cap',action='store_true');P.add_argument('--handoff-trace',action='store_true');P.add_argument('--handoff-ppo-updates',type=int,default=0);P.add_argument('--handoff-contact',action='store_true');P.add_argument('--initial-trace',action='store_true');P.add_argument('--initial-ppo-updates',type=int,default=0);P.add_argument('--filter',choices=['instant','ema'],default='instant');P.add_argument('--trace',action='store_true');P.add_argument('--group',required=True);P.add_argument('--source',default='reliability_v2_brake_recovery_20261017_source');P.add_argument('--seed',type=int,default=1100);P.add_argument('--episodes',type=int,default=128);P.add_argument('--initial',default='initial_gain_calibration_20261015_r2/initial_gain115.pt');P.add_argument('--post',default='specialist_stage_20261015_postrnn1_fit/fitted_recurrent_post_actor.pt');P.add_argument('--recovery',default='specialist_stage_20261015_releasehead2_fit/release_candidate/model_release_actor.pt');P.add_argument('--align',default='pose_verified_20261015_r1/model_goal_mlp.pt');P.add_argument('--flags',nargs='*',default=[]);P.add_argument('--profiles',nargs='+',default=['fixed','random']);P.add_argument('--late-initial',default=None);P.add_argument('--deep-recovery',default=None);P.add_argument('--near-align',default=None);P.add_argument('--boundary-mode',choices=['zero','train','eval'],required=True);P.add_argument('--boundary-checkpoint',default=None);P.add_argument('--boundary-updates',type=int,default=6);A=P.parse_args();D=R/'logs'/A.group
def out(c):return subprocess.check_output(c,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def active():return out(['docker','ps','--filter','name=shijinsheng-piper','--format','{{.Names}}'])
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def opt(a,k,v=None):
 a=list(a)
 while k in a:i=a.index(k);del a[i:i+2]
 if v is not None:a.extend([k,str(v)])
 return a
def status(**f):(D/'status.json').write_text(json.dumps(dict(server_utc=utc(),**f),indent=2));print(json.dumps(f),flush=True)
assert A.group.startswith('reliability_v2_') and not D.exists() and not active()
assert all(f in ['late_xy_teacher','late_xy_fit','align_xy_teacher','align_xy_fit','near_align_teacher'] for f in A.flags);A.flags=['--'+f for f in A.flags]
assert A.seed in [1124,2810,2812,2814,2816,2818,2820,2822,2824,2826,2828,2830,2832,2834,2836,2838,2840,2842,2844,2846,2848,2850,2852,2854,2856,2872,2888] and A.episodes in [128,512]
assert A.boundary_mode=='eval'
assert A.case_profile==('all' if A.equivalence or A.conditioned_critic else 'lateral_focus')
assert A.offset_mm==5
assert A.deep_updates==0 and A.handoff_contact and not A.case_replay and A.profiles==['random'] and not A.feature_history
assert _namespace_free_bytes(R)>=MINFREE+32*1024**2
assert not A.paired or (A.seed==1084 and A.profiles==['random'] and A.episodes==128 and A.handoff_contact and not A.post_updates and not A.case_replay and A.pair_scale in [0.,1.])
assert not A.proof
assert not A.feature_history or not (A.proof or A.post_updates)
assert not A.proof or (A.seed==1082 and A.episodes==128 and A.profiles==['random'] and A.handoff_contact and not A.post_updates)
assert not A.initial_ppo_updates
assert not A.post_updates
assert not A.post_updates or (A.seed==1098 and A.post_updates==4 and A.profiles==['random'] and A.episodes==128 and A.handoff_contact and not A.case_replay and not A.axial_cap)
assert not A.handoff_ppo_updates
assert _namespace_free_bytes(R)>=MINFREE,'Round2 additional1.5GiB cap/reserve reached'
assert A.group.startswith(('reliability_v2_expanded24_a_native_20261020_','reliability_v2_expanded24_a_20261020_'))
# Wait for the unchanged GPU admission limits BEFORE creating a job directory.
deadline=time.monotonic()+60;stable=0;resource_samples=[]
while time.monotonic()<deadline:
 memory_mib,util_percent=map(int,out(['nvidia-smi','--id=2','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).split(','));resource_samples.append(dict(memory_free_MiB=memory_mib,utilization_percent=util_percent));stable=stable+1 if memory_mib>=14000 and util_percent<=5 and not active() else 0
 if stable>=3:break
 time.sleep(2)
assert stable>=3,'GPU admission unchanged;no native job directory/samples created'
D.mkdir();runtime=R/'logs'/A.source/'executable';(D/'executable').mkdir();shutil.copyfile(runtime/'source_manifest.json',D/'executable/source_manifest.json');M=load(D/'executable/source_manifest.json')
for n,h in M['executable_hashes'].items():assert sha(runtime/n)==h
source_plan=load(R/'logs'/A.source.removesuffix('_source')/'plan.json');shared=R/'logs/expanded24_shared_source_20261020.tar.gz';assert sha(shared)==source_plan['source_archive_sha256'];(D/'shared_source_reference.json').write_text(json.dumps(dict(schema='shared-source-v1',source_archive_sha256=sha(shared),source_manifest_sha256=sha(runtime/'source_manifest.json'),logical_executable_prefix=A.group+'/executable/'),indent=2))
assert sha(R/'piper_env.py')=='933d4a4ccdbef7fe3f999402e31286cb02b92e84854626befceae773673bee8e'
lock=(D/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
paths=([A.conditioned_critic] if A.conditioned_critic else [])+[A.initial,A.post,A.recovery,A.align]+([A.boundary_checkpoint] if A.boundary_checkpoint else [])+([A.late_initial] if A.late_initial else [])+([A.deep_recovery] if A.deep_recovery else [])+([A.near_align] if A.near_align else []);sealed={p:sha(R/'logs'/p) for p in paths}
(D/'plan.json').write_text(json.dumps(dict(native_GPU_admission_samples=resource_samples,final_retry_release=A.release_guard,controller_policy_changed=A.release_guard,server_utc=utc(),training_seeds=[2811,2813,2815,2817,2819,2821,2823,2825,2827,2829,2831,2833,2835,2837,2839,2841,2843,2845,2847,2849,2851,2853,2855,2857],development_seeds=[1190,1191],formal_seeds_reserved=[1600,1601,1602,1603],completed_formal_seeds=[1311,1201,1202,1203,1300,1301,1302,1303,1400,1401,1402,1403,1500,1501,1502,1503],roles=dict(initial=A.initial,late_initial=A.late_initial,deep_recovery=A.deep_recovery,near_alignment=A.near_align,post=A.post,recovery=A.recovery,alignment=A.align),checkpoint_hashes=sealed,seed=A.seed,episodes=A.episodes,profiles=A.profiles,flags=A.flags,physical_gates_unchanged=True,maximum_net_new_bytes=128*1024**2,initial_free_bytes=_namespace_free_bytes(R),reserve_free_bytes=MINFREE),indent=2))
try:
 results={};previous_boundary=A.boundary_checkpoint
 for profile in A.profiles:
  assert profile in ['fixed','random','original16','payload16'];index=['fixed','random','original16','payload16'].index(profile)
  assert not active() and _namespace_free_bytes(R)>=MINFREE
  free,use=map(int,out(['nvidia-smi','--id=2','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).split(','));assert free>=14000 and use<=5
  template='shijinsheng-piper-hybrid-'+dict(fixed='heldout',random='random_heldout',original16='regression',payload16='payload_regression')[profile]+'-20261014-v2';v=json.loads(out(['docker','inspect',template]))[0];assert v['Image']==I
  a=v['Config']['Cmd'];a=a[a.index('--scene_profile'):]
  for k in ['--compare_checkpoint','--reference_checkpoint']:a=opt(a,k)
  for k,val in {'--mode':'eval','--iterations':0,'--seed':A.seed+index,'--eval_episodes':A.episodes,'--checkpoint':W+'/logs/'+A.post,'--initial_insertion_checkpoint':W+'/logs/'+A.initial,'--alignment_checkpoint':W+'/logs/'+A.align,'--recovery_checkpoint':W+'/logs/'+A.recovery}.items():a=opt(a,k,val)
  if index<2 and A.late_initial:a=opt(a,'--late_initial_checkpoint',W+'/logs/'+A.late_initial)
  if index<2 and A.deep_recovery:a=opt(a,'--deep_recovery_checkpoint',W+'/logs/'+A.deep_recovery)
  if index<2 and A.near_align:a=opt(a,'--near_alignment_checkpoint',W+'/logs/'+A.near_align)
  if index>=2:
   for k in ['--initial_insertion_checkpoint','--alignment_checkpoint']:a=opt(a,k)
   for k,val in {'--checkpoint':W+'/logs/ppo_cad_response_refine_seed166_20261013/model_2099.pt','--recovery_checkpoint':W+'/logs/ppo_cad_recovery_seed214_20261014_v3/model_399.pt','--compare_checkpoint':W+'/logs/ppo_cad_response_refine_seed166_20261013/model_2099.pt'}.items():a=opt(a,k,val)
  if index<2:
   a=opt(a,'--boundary_mode',A.boundary_mode);a=opt(a,'--boundary_updates',A.boundary_updates)
   if previous_boundary:a=opt(a,'--boundary_checkpoint',W+'/logs/'+previous_boundary)
  name=A.group+'_'+profile;F=R/'logs'/name;assert not F.exists();F.mkdir()
  # Assets are immutable and referenced in place, saving51MiB per run.
  a=opt(a,'--run_name',name)+['--skip_straight_baseline']+A.flags
  if index<2:a+=['--block_force_filter',A.filter]
  if A.initial_trace:a+=['--initial_trace']
  if A.handoff_trace or A.group.endswith('_known') or A.case_replay:a+=['--handoff_trace']
  if A.case_replay:a+=['--initial_case_replay',W+'/logs/reliability_v2_brake_recovery_20261017_seen_case.json']
  if index<2 and A.axial_cap:a+=['--alignment_axial_cap']
  if A.handoff_contact:a+=['--fixed_contact_cases','--precontact_fraction','1.0']
  if A.handoff_ppo_updates:a+=['--handoff_ppo_updates',str(A.handoff_ppo_updates)]
  if A.critic_shadow:a+=['--critic_shadow']
  if A.paired:a+=['--paired_diagnostic','--pair_scale',str(A.pair_scale)]
  if A.proof:a+=['--history_proof',A.proof]
  if A.feature_history and index<2:a+=['--independent_boundary_history']
  a=opt(a,'--case_profile',A.case_profile)
  assert '--recovery_offset_min_mm' not in a
  a=opt(a,'--recovery_offset_mm',A.offset_mm)
  a+=['--joint5_input','--critic_domain_collect','--critic_conditioning',A.conditioning]
  if A.release_guard:a+=['--final_retry_release']
  if A.equivalence:a+=['--critic_equivalence']
  if A.history:a+=['--critic_history']
  if A.conditioned_critic:a+=['--conditioned_critic_checkpoint',W+'/logs/'+A.conditioned_critic]
  if A.withdrawal_detail:a+=['--withdrawal_detail']
  if index<2 and A.fixed_brake:a+=['--fixed_final_retry_brake']
  if A.post_updates:a+=['--post_ppo_updates',str(A.post_updates)]
  c='shijinsheng-piper-'+name.replace('_','-')
  command=['docker','run','-d','--pull','never','--name',c,'--gpus','device=2','--cpus=16','--ulimit','core=0','--network','none','--ipc','private','--shm-size=4g','--entrypoint','bash','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1','-e','OMNI_KIT_ALLOW_ROOT=1','-e','DISPLAY=','-e','NVIDIA_VISIBLE_DEVICES=2','-e','OMP_NUM_THREADS=8','-e','MKL_NUM_THREADS=8','-v',str(R)+':'+W,'-v',str(R)+':'+str(R)+':ro','-v','/data/shijinsheng/piper-insertion:/data/shijinsheng/piper-insertion:ro','-v',str(R/'logs/kit')+':/isaac-sim/kit/logs','-v',str(R/'cache/kit')+':/root/.cache/ov','-v',str(R/'cache/compute')+':/root/.nv/ComputeCache','-w',W,I,'-lc','run=$1;exe=$2;shift 2;/isaac-sim/python.sh "$exe" "$@" > "logs/$run/console.log" 2>&1;result=$?;/isaac-sim/python.sh record_process_exit.py "$result" train.py --run_name "$run";exit "$result"','piper-reliability-v2',name,W+'/logs/'+A.source+'/executable/train.py']+a
  (F/'launch_manifest.json').write_text(json.dumps(dict(final_retry_release=A.release_guard,server_utc=utc(),arguments=a,docker_command=command,source_hashes=M['executable_hashes'],checkpoint_hashes=sealed),indent=2))
  identifier=out(command);status(stage=profile,state='running',container=c,container_id=identifier)
  deadline=time.monotonic()+2400
  while time.monotonic()<deadline:
   p=F/'status.json'
   if p.exists() and load(p).get('process_exit_code') is not None:
    x=load(p);assert x['status']=='completed' and x['process_exit_code']==0,x
    if out(['docker','inspect','--format','{{.State.Running}}',c])=='false':break
   time.sleep(10)
  else:raise TimeoutError('Leave existing process intact: '+c)
  if A.proof:
   v=load(F/'history_proof.json');results[profile]=dict(episodes=v['episodes'],success_rate=v['successes']/v['episodes'],force_limit_exceedances=v['force_aborts'],diagnostic_only=True)
  elif A.paired:
   v=load(F/'paired_result.json');results[profile]=dict(episodes=v['complete_episodes'],success_rate=v['successes']/v['complete_episodes'],force_limit_exceedances=v['force_aborts'],diagnostic_only=True)
  elif A.group.startswith(('reliability_v2_expanded24_a_native_20261020_','reliability_v2_expanded24_a_20261020_')):
   v=load(F/'domain_collect_result.json');ep=v['episodes'];results[profile]=dict(episodes=ep,success_rate=v['successes']/ep,force_limit_exceedances=v['force_aborts'],training_collection_only=True)
  elif A.post_updates:
   v=load(F/'post_train_result.json');assert v['status'] in ['completed_critic_pilot','completed','stopped_safety','stopped_value_gate','stopped_budget','stopped_storage'],v
   ep=sum(x['full_episodes'] for x in v['batches']);results[profile]=dict(episodes=ep,success_rate=sum(x['successes'] for x in v['batches'])/ep,force_limit_exceedances=sum(x['over100N'] for x in v['batches']),training_only=True)
   previous_boundary=A.boundary_checkpoint
  else:results[profile]=load(F/'evaluation.json')['ppo']
  (D/'development.json').write_text(json.dumps(results,indent=2))
 assert all(sha(R/'logs'/p)==h for p,h in sealed.items())
 status(state='completed',stage='completed',results={k:dict(episodes=v['episodes'],successes=round(v['success_rate']*v['episodes']),over100N=v['force_limit_exceedances']) for k,v in results.items()},source_checkpoint_files_unchanged=True,trainable_weights_updated=False,actor_weights_updated=False)
except BaseException as e:status(state='failed',error=str(e));traceback.print_exc();raise

