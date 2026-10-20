"""Stage3D: cold-start alignment episodes replaying physically exited states.

Preparation happens in a separate collector, not inside PPO rollouts. Joint
positions/velocities, accumulated Cartesian commands and servo targets are
preserved. Only axial action is suppressed; no automatic XY/orientation servo
or privileged state is used by the controller.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from isaaclab.utils import configclass
from cad_env import ChallengeCupCfg, ChallengeCupEnv
from contact_reward import contact_cost


def collect_states(env,raw,logdir,args):
    env.reset()
    records=[];finished=set();failures=[]
    for _ in range(720):
        actions=torch.zeros(env.num_envs,6,device=env.device);actions[:,2]=-.35
        obs,reward,dones,info=env.step(actions)
        assert torch.isfinite(obs['policy']).all()
        relative,orientation,force=raw._geometry()
        eligible=(raw.supervisor.mode==2)&(raw.true_released_exit_count>=1)&(force.norm(dim=-1)<2.)
        eligible &= (relative[:,2]>=.004)&(relative[:,2]<=.03)&~dones.bool()
        for i in eligible.nonzero().flatten().tolist():
            if i in finished:continue
            finished.add(i)
            records.append(dict(env_id=i,case_id=i%4,
                q=raw.robot.data.joint_pos[i].cpu().numpy().copy(),qdot=raw.robot.data.joint_vel[i].cpu().numpy().copy(),
                targets=raw.targets[i].cpu().numpy().copy(),
                command_pos=(raw.command_pos[i]-raw.scene.env_origins[i]).cpu().numpy().copy(),
                command_quat=raw.command_quat[i].cpu().numpy().copy(),
                hole_pos=raw.hole_pos[i].cpu().numpy().copy(),hole_quat=raw.hole_quat[i].cpu().numpy().copy(),
                relative=relative[i].cpu().numpy().copy(),orientation=orientation[i].cpu().numpy().copy(),
                peak_force_n=float(raw.peak_force[i]),handover_force_n=float(raw.handover_force[i]),
                final_force_n=float(force[i].norm()),withdrawal_m=float(raw.true_trigger_depth[i]+relative[i,2])))
        if dones.any():
            terminal=info['terminal']
            for j,i in enumerate(terminal['env_ids'].tolist()):
                if i not in finished:
                    finished.add(i);failures.append(dict(env_id=i,end_reason=int(terminal['end_reason'][j]),
                        peak_force_n=float(terminal['peak_force_n'][j]),handover_force_n=float(terminal['handover_force_n'][j])))
        if len(finished)==env.num_envs:break
    assert len(finished)==env.num_envs,(len(finished),env.num_envs)
    assert len(records)>=env.num_envs*.75,(len(records),failures)
    assert all(sum(r['case_id']==c for r in records)>=max(1,env.num_envs//16) for c in range(4))
    records.sort(key=lambda r:r['env_id'])
    keys=[k for k in records[0] if k!='env_id']
    arrays={key:np.stack([r[key] for r in records]) for key in keys}
    arrays['source_sha256']=np.array(raw.geometry['source_sha256'])
    np.savez_compressed(Path(logdir)/'post_exit_states.npz',**arrays)
    metadata=dict(status='completed',seed=args.seed,requested=env.num_envs,captured=len(records),
        first_episode_only=True,selection='actual withdrawal>=3mm and force<2N after physical contact; safely outside mouth',
        commanded_targets='preserved',failures=failures,by_case={str(c):sum(r['case_id']==c for r in records) for c in range(4)},
        args=vars(args),source_sha256=raw.geometry['source_sha256'],
        bank_sha256=hashlib.sha256((Path(logdir)/'post_exit_states.npz').read_bytes()).hexdigest())
    (Path(logdir)/'state_bank.json').write_text(json.dumps(metadata,indent=2))
    print('POST_EXIT_STATE_BANK_SAVED',json.dumps({k:v for k,v in metadata.items() if k!='args'}),flush=True)


@configclass
class AlignmentCfg(ChallengeCupCfg):
    refresh_contact_on_reset=True
    episode_length_s=6.
    alignment_banks=()
    alignment_reward_scale=.005
    alignment_angle_weight=2.
    alignment_sample_random=False
    alignment_insertion_depth=0.
    success_profile='trueXY1mm_angle05_projected_shaft_fit_free_force2N_hold8_v1'


class AlignmentEnv(ChallengeCupEnv):
    def __init__(self,cfg,**kwargs):
        parts=[];self.bank_manifests=[]
        for name in cfg.alignment_banks:
            path=Path(name)
            with np.load(path,allow_pickle=False) as data:
                assert str(data['source_sha256'])==cfg.geometry['source_sha256']
                parts.append({key:np.array(data[key]) for key in data.files if key!='source_sha256'})
            self.bank_manifests.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert parts,'Alignment requires an explicit physical post-exit state bank'
        self.bank_offsets=np.cumsum([0]+[len(p['q']) for p in parts[:-1]]).tolist()
        self._bank_numpy={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
        # Reject unsafe/inconsistent collector records, rather than hiding them during resets.
        assert np.all(self._bank_numpy['withdrawal_m']>=.003)
        assert np.all(self._bank_numpy['final_force_n']<2.)
        assert np.all(self._bank_numpy['handover_force_n']>=5.)
        super().__init__(cfg,**kwargs)
        self.bank={key:torch.as_tensor(value,device=self.device) for key,value in self._bank_numpy.items()}
        self.bank_size=len(self.bank['q'])
        self.bank_cursor=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.bank_indices=torch.zeros_like(self.bank_cursor)
        self.bank_case_ids=torch.zeros_like(self.bank_cursor)
        self.initial_tip_relative=torch.zeros(self.num_envs,3,device=self.device)
        self.predicted_shaft_fit=torch.zeros(self.num_envs,device=self.device)

    def _pre_physics_step(self,actions):
        relative,orientation,_=self._geometry()
        self.previous_alignment_potential=self._potential(relative,orientation)
        applied=actions.clone();applied[:,2]=0.
        super()._pre_physics_step(applied)

    def _potential(self,relative,orientation):
        xy=relative[:,:2].norm(dim=-1)/self.cfg.alignment_reward_scale
        angle=orientation.norm(dim=-1)/.05
        return -5.*xy.clamp_max(20.)-self.cfg.alignment_angle_weight*angle.clamp_max(20.)

    def _get_dones(self):
        previous_hold=self.hold.clone();previous_success=self.ever_success.clone()
        super()._get_dones()
        relative,orientation,force=self._geometry();norm=force.norm(dim=-1)
        predicted=relative.clone()
        if self.cfg.alignment_insertion_depth:predicted[:,2]=-self.cfg.alignment_insertion_depth
        self.predicted_shaft_fit=self._shaft_fits(predicted).float()
        valid=(relative[:,:2].abs().amax(-1)<.001)&(orientation.norm(dim=-1)<.05)
        valid &= self._shaft_fits(relative)&(relative[:,2]>=.003)&(relative[:,2]<=.035)&(norm<2.)
        valid &= self.predicted_shaft_fit.bool()
        self.hold=torch.where(valid,previous_hold+1,torch.zeros_like(previous_hold))
        success=self.hold>=8;self.ever_success=previous_success|success
        self.end_hold_steps.copy_(self.hold.float())
        failure=(norm>100.)|(relative[:,:2].norm(dim=-1)>.12)|(relative[:,2]<.001)|(relative[:,2]>.05)
        timeout=self.episode_length_buf>=self.max_episode_length-1
        self.end_reason[:]=0;self.end_reason[timeout]=3;self.end_reason[failure]=2
        self.end_reason[norm>100.]=1;self.end_reason[success]=4
        return success|failure,timeout

    def _get_rewards(self):
        relative,orientation,force=self._geometry()
        # Dense distance costs stay informative at5mm errors; goal accuracy is
        # independent of this curriculum scale and remains1mm in all stages.
        potential=self._potential(relative,orientation)
        potential=torch.where(self.reset_buf,torch.zeros_like(potential),potential)
        reward=.99*potential-self.previous_alignment_potential-.02
        reward-=.002*self.actions.square().sum(-1)
        reward-=contact_cost(force.norm(dim=-1),self.cfg.reward_profile)
        success=self.hold>=8
        reward+=100.*success.float()
        reward-=10.*(self.reset_terminated&~success).float()+5.*self.reset_time_outs.float()
        self.episode_returns+=reward
        self.extras['log']={'alignment/xy_m':relative[:,:2].norm(dim=-1).mean(),
            'alignment/angle_rad':orientation.norm(dim=-1).mean(),'alignment/success_step':success.float().mean()}
        return reward

    def _reset_idx(self,ids):
        if ids is None:ids=self.robot._ALL_INDICES
        initialized=hasattr(self,'bank')
        if initialized:
            extra={'alignment_bank_index':self.bank_indices[ids].clone(),'alignment_case':self.bank_case_ids[ids].clone(),
                   'alignment_projected_fit':self.predicted_shaft_fit[ids].clone()}
        super()._reset_idx(ids)
        if not initialized:return
        self.extras['terminal'].update(extra)
        if self.cfg.alignment_sample_random:
            selected=torch.randint(self.bank_size,(len(ids),),device=self.device)
        else:
            selected=(ids+self.bank_cursor[ids]*self.num_envs)%self.bank_size
        self.bank_cursor[ids]+=1
        self.bank_indices[ids]=selected;self.bank_case_ids[ids]=self.bank['case_id'][selected].long()
        self.initial_tip_relative[ids]=self.bank['relative'][selected].float()
        self.hole_pos[ids]=self.bank['hole_pos'][selected].float()
        self.hole_quat[ids]=self.bank['hole_quat'][selected].float()
        for wall,offset in zip(self.walls,self.wall_offsets):
            from isaaclab.utils.math import quat_apply
            shifted=quat_apply(self.hole_quat[ids],torch.tensor(offset,device=self.device).expand(len(ids),-1))
            wall.write_root_pose_to_sim(torch.cat((self.hole_pos[ids]+self.scene.env_origins[ids]+shifted,self.hole_quat[ids]),-1),env_ids=ids)
        q=self.bank['q'][selected].float();qdot=self.bank['qdot'][selected].float()
        self.robot.write_joint_state_to_sim(q,qdot,env_ids=ids)
        self.targets[ids]=self.bank['targets'][selected].float()
        self.robot.set_joint_position_target(self.targets[ids],env_ids=ids)
        self.command_pos[ids]=self.bank['command_pos'][selected].float()+self.scene.env_origins[ids]
        self.command_quat[ids]=self.bank['command_quat'][selected].float()
        self.command_initialized[ids]=True


def alignment_smoke(env,raw,logdir):
    raw.verify_cad_scene();peak=0.;error=0.;probes=[]
    for offset in raw.bank_offsets:
        assert offset%env.num_envs==0,'Smoke offset must map exactly to bank'
        raw.bank_cursor.fill_(offset//env.num_envs)
        env.reset()
        relative=raw._geometry()[0]
        error=max(error,float((relative-raw.initial_tip_relative).abs().max()))
        assert error<.00005,('state restoration FK error',error)
        for _ in range(24):
            obs,reward,dones,info=env.step(torch.zeros(env.num_envs,6,device=env.device))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            assert not dones.any(),'Holding bank state unexpectedly terminated'
            peak=max(peak,float(raw._geometry()[2].norm(dim=-1).max()))
        if raw.device=='cpu':raw.verify_cad_scene()
        probes.append(dict(bank_offset=offset,bank_indices=raw.bank_indices.tolist()))
    assert peak<5.,('restored free state has contact',peak)
    assert (raw.actions[:,2]==0.).all()
    result=dict(status='passed',task='post_exit_alignment_only',simulation_device=raw.device,
        fk_restoration_error_m=error,max_hold_force_n=peak,bank_states=raw.bank_size,banks=raw.bank_manifests,
        source_sha256=raw.geometry['source_sha256'],axial_action=0.,command_and_joint_targets='restored_not_cleared',bank_probes=probes)
    (Path(logdir)/'smoke.json').write_text(json.dumps(result,indent=2));print('ALIGNMENT_REPLAY_PHYSICAL_SMOKE_PASSED',json.dumps(result),flush=True)
