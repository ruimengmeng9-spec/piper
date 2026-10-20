"""Insertion on the uploaded challenge-cup station, using its articulated CAD geometry."""
import json
import os
from pathlib import Path
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from piper_env import PiperInsertionCfg,PiperInsertionEnv

ROOT=Path("/workspace/piper-insertion")
CAD=Path(os.environ.get('PIPER_CAD_ASSET_DIR',str(ROOT/'cad/challenge_cup')))
GEOMETRY=json.loads((CAD/'task_geometry.json').read_text())

@configclass
class ChallengeCupCfg(PiperInsertionCfg):
    geometry=GEOMETRY
    scene_profile='challenge_cup'
    action_frame='hole'
    robot_cfg=PiperInsertionCfg().robot_cfg.copy()
    robot_cfg.spawn.usd_path=str(CAD/'challenge_cup_robot.usdc')
    robot_cfg.init_state.pos=tuple(GEOMETRY['robot_base_position'])
    robot_cfg.init_state.rot=tuple(GEOMETRY['robot_base_quaternion'])
    robot_cfg.init_state.joint_pos={f'joint{i+1}':v for i,v in enumerate(GEOMETRY['joint_positions'])}

class ChallengeCupEnv(PiperInsertionEnv):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        self.verify_jacobian_frame()

    def verify_jacobian_frame(self):
        import numpy as np
        import xml.etree.ElementTree as ET
        from scipy.spatial.transform import Rotation
        r=ET.parse(ROOT/'vendor/agx_arm_urdf/piper/urdf/piper_description.urdf').getroot()
        transforms=[]
        for i in range(1,7):
            o=r.find(f"joint[@name='joint{i}']/origin");t=np.eye(4)
            t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix();t[:3,3]=np.fromstring(o.get('xyz'),sep=' ');transforms.append(t)
        def fk(q):
            t=np.eye(4)
            for o,angle in zip(transforms,q):
                turn=np.eye(4);turn[:3,:3]=Rotation.from_rotvec([0,0,angle]).as_matrix();t=t@o@turn
            return t[:3,3]
        q=self.robot.data.joint_pos[0,self.arm_ids].detach().cpu().numpy();J=np.zeros((3,6));eps=1e-5
        for i in range(6):
            plus=q.copy();minus=q.copy();plus[i]+=eps;minus[i]-=eps;J[:,i]=(fk(plus)-fk(minus))/(float(plus[i])-float(minus[i]))
        rq=self.robot.data.root_quat_w[0].detach().cpu().numpy();R=Rotation.from_quat([*rq[1:],rq[0]]).as_matrix()
        full=self.robot.root_physx_view.get_jacobians()[0,self.jac_idx][:,self.arm_ids].detach().cpu().numpy()
        actual=full[:3]
        root_error=float(np.max(np.abs(actual-J)));world_error=float(np.max(np.abs(actual-R@J)))
        self._ik_com_offset=self.robot.root_physx_view.get_coms()[:,self.ee_idx,:3].to(self.device).clone()
        eq=self.robot.data.body_quat_w[0,self.ee_idx].detach().cpu().numpy()
        re=Rotation.from_quat([*eq[1:],eq[0]]).as_matrix()@self._ik_com_offset[0].cpu().numpy()
        def skew(v):
            x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        world_com_error=float(np.max(np.abs(actual+skew(re)@full[3:]-R@J)))
        root_com_error=float(np.max(np.abs(actual+skew(R.T@re)@full[3:]-J)))
        candidates=[(world_error,'world',False),(root_error,'root',False),(world_com_error,'world',True),(root_com_error,'root',True)]
        error,chosen,about_com=min(candidates)
        assert error<.0005,(candidates,actual,J)
        self.ik_in_root_frame=chosen=='root';self.jacobian_about_com=about_com
        report=dict(jacobian_frame=chosen,jacobian_reference='COM' if about_com else 'link_origin',
            validated_origin_jacobian_error_m=error,raw_root_error_m=root_error,raw_world_error_m=world_error,
            corrected_root_error_m=root_com_error,corrected_world_error_m=world_com_error)
        Path(self.cfg.log_dir,'cad_jacobian_frame.json').write_text(json.dumps(report,indent=2))
        print('CAD_JACOBIAN_FRAME_VERIFIED',json.dumps(report),flush=True)

    def _spawn_workspace(self):
        ground=sim_utils.CuboidCfg(size=(100.,100.,.02),collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.18,.18,.18)))
        ground.func('/World/ground',ground,translation=(0,0,-.01))
        station=sim_utils.UsdFileCfg(usd_path=str(CAD/'challenge_cup_station.usdc'))
        station.func('/World/envs/env_0/Station',station)
        # Explicit experimental support, absent from the supplied CAD.
        from scipy.spatial.transform import Rotation
        hq=GEOMETRY['hole_quaternion'];r=Rotation.from_quat([*hq[1:],hq[0]]).as_matrix()
        height=max(.03,GEOMETRY['hole_center'][2]-.04*(abs(r[2,0])+abs(r[2,1]))+min(0.,-.06*r[2,2]))
        fixture=sim_utils.CuboidCfg(size=(.10,.10,height),collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.32,.34,.37)))
        fixture.func('/World/envs/env_0/ExperimentalFixture',fixture,
            translation=(GEOMETRY['hole_center'][0],GEOMETRY['hole_center'][1]-.03,height/2))

    def verify_cad_scene(self):
        from pxr import Usd,UsdGeom,UsdPhysics
        root=self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
        assert root.GetCustomDataByKey('source_sha256')==self.geometry['source_sha256']
        indices=[p.GetCustomDataByKey('source_mesh_index') for p in Usd.PrimRange(root)
                 if p.GetCustomDataByKey('source_mesh_index') is not None]
        assert sorted(indices)==sorted([*range(1,104),207,*range(222,230)])
        assert self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/link6/cad/mesh_228').IsValid()
        assert self.sim.stage.GetPrimAtPath('/World/envs/env_0/Station').IsValid()
        assert len(self.arm_ids)==6 and len(self.robot.body_names)==7
        mass=self.robot.root_physx_view.get_masses()[:,self.ee_idx]
        expected_mass=(self.expected_payload_mass.to(mass.device) if hasattr(self,'expected_payload_mass')
                       else torch.full_like(mass,.666131115))
        assert torch.allclose(mass,expected_mass,atol=2e-5),(mass,expected_mass)
        if self.device=='cpu':
            import omni.physx
            self.sim.step(render=False)
            self.scene.update(self.cfg.sim.dt)
            self.sim.render()
            ray_origin=tuple((self.scene.env_origins[0]+torch.tensor([.15,0.,1.],device=self.device)).tolist())
            hit=omni.physx.get_physx_scene_query_interface().raycast_closest(ray_origin,(0,0,-1),1.1)
            assert hit['hit'] and 'Station' in str(hit['collision']),hit
            assert abs(float(hit['position'][2])-float(self.scene.env_origins[0,2])-.0082)<.0001,hit
            from isaaclab.utils.math import quat_apply
            axis=quat_apply(self.robot.data.body_quat_w[:,self.ee_idx],torch.tensor([0.,0.,1.],device=self.device).expand(self.num_envs,-1))
            tip=self.robot.data.body_pos_w[:,self.ee_idx]+quat_apply(self.robot.data.body_quat_w[:,self.ee_idx],self.tip_offset)
            for i in range(self.num_envs):
                # In post-exit poses the ray origin can be inside a hole wall.
                # Inspect all hits so the wall cannot hide the actual peg.
                hits=[]
                def collect_hit(hit):
                    if isinstance(hit,dict):hits.append(dict(hit))
                    else:hits.append(dict(collision=str(hit.collision),distance=float(hit.distance)))
                    return True
                omni.physx.get_physx_scene_query_interface().raycast_all(tuple((tip[i]+.02*axis[i]).tolist()),tuple((-axis[i]).tolist()),.03,collect_hit)
                peg=[h for h in hits if str(h.get('collision'))==f'/World/envs/env_{i}/Robot/link6/collisions/mesh_5/box']
                assert peg,(hits,tip[i].tolist(),axis[i].tolist())
                hit=min(peg,key=lambda h:h['distance'])
                assert abs(hit['distance']-.02)<.0001,hit
        print('UPLOADED_CAD_SCENE_VERIFIED',self.geometry['source_sha256'],len(indices),flush=True)

    def save_pose_snapshot(self,path):
        q=self.robot.data.body_quat_w[0].detach().cpu().tolist()
        pos=(self.robot.data.body_pos_w[0]-self.scene.env_origins[0]).detach().cpu().tolist()
        Path(path).write_text(json.dumps(dict(source_sha256=self.geometry['source_sha256'],
            body_names=self.robot.body_names,body_position=pos,body_quaternion_wxyz=q,
            joint_positions=self.robot.data.joint_pos[0].detach().cpu().tolist(),geometry=self.geometry,
            hole_position=self.hole_pos[0].detach().cpu().tolist(),hole_quaternion_wxyz=self.hole_quat[0].detach().cpu().tolist()),indent=2))
