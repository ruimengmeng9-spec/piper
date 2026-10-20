"""GPU PhysX insertion task using the official six-axis Piper geometry.

The peg is rigidly attached to the locked gripper. This is an insertion-only
baseline; grasping, slip, vision and real hardware deployment are out of scope.
"""
import json
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    compute_pose_error, quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul,
)
from task_geometry import hole_geometry
from action_delay import ActionDelay
from contact_reward import contact_cost, alignment_length_scale, orientation_length_scale
from command_bound import limit_vector, limit_downward_lead

ROOT = Path(__file__).resolve().parent
GEOMETRY = json.loads((ROOT / 'assets/geometry.json').read_text())

@configclass
class PiperInsertionCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 5.0
    action_space = 6
    observation_space = 33
    state_space = 0
    is_finite_horizon = True
    sim = sim_utils.SimulationCfg(dt=1/240, render_interval=4,
        physx=sim_utils.PhysxCfg(solver_type=1, min_position_iteration_count=8,
                                min_velocity_iteration_count=1))
    scene = InteractiveSceneCfg(num_envs=256, env_spacing=1.5, replicate_physics=True)
    robot_cfg = ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROOT / 'assets/piper_pregrasped.usd'),
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0003, rest_offset=0.),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,
                max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8,
                solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., GEOMETRY['robot_base_height']),
            joint_pos={f'joint{i+1}': v for i, v in enumerate(GEOMETRY['joint_positions'])}),
        actuators={'arm': ImplicitActuatorCfg(joint_names_expr=['joint[1-6]'],
                    stiffness=800.0, damping=45.0, effort_limit_sim=30.0, velocity_limit_sim=2.0)},
        soft_joint_pos_limit_factor=0.95)
    joint_noise = 0.01
    hole_xy_noise = 0.004
    hole_yaw_noise = 0.04
    reward_profile = 'potential_v2'
    hole_width = .016
    observation_bias = 0.
    privileged_critic = False
    action_delay_min = 0
    action_delay_max = 0
    friction_scale_min = 1.
    friction_scale_max = 1.
    force_noise_std = 0.
    force_bias_bound = 0.
    stiffness_scale_min = 1.
    stiffness_scale_max = 1.
    peg_mass_min = .02
    peg_mass_max = .02
    grip_axial_min = 0.
    grip_axial_max = 0.
    grip_endpoint_fraction = 0.
    grip_positive_endpoint_probability = .5
    nominal_replay_fraction = 0.
    nominal_dynamics_fraction = 0.
    command_position_bound = 0.
    command_rotation_bound = 0.
    command_axial_lead_bound = 0.
    near_contact_position_step = 0.
    near_contact_rotation_step = 0.
    near_contact_band = .03
    success_profile = 'shaft_containment_v3'
    geometry = GEOMETRY
    scene_profile = 'simplified'
    action_frame = 'world'

class PiperInsertionEnv(DirectRLEnv):
    cfg: PiperInsertionCfg

    def __init__(self, cfg, **kwargs):
        self.geometry = cfg.geometry
        self.hole_geometry = hole_geometry(self.geometry['peg_width'],cfg.hole_width)
        cfg.robot_cfg.spawn.collision_props.contact_offset = self.hole_geometry['contact_offset']
        super().__init__(cfg, **kwargs)
        self.arm_ids, names = self.robot.find_joints([f'joint{i}' for i in range(1, 7)], preserve_order=True)
        self.ee_idx = self.robot.find_bodies('link6')[0][0]
        self.jac_idx = self.ee_idx - 1 if self.robot.is_fixed_base else self.ee_idx
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(
            command_type='pose', use_relative_mode=False, ik_method='dls'), self.num_envs, self.device)
        self.actions = torch.zeros(self.num_envs, 6, device=self.device)
        self.previous_actions = self.actions.clone()
        self.action_delay = ActionDelay(self.num_envs, 6, cfg.action_delay_max, self.device)
        self.nominal_dynamics_mask=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        if cfg.nominal_dynamics_fraction:
            count=round(self.num_envs*cfg.nominal_dynamics_fraction)
            self.nominal_dynamics_mask[torch.randperm(self.num_envs,device=self.device)[:count]]=True
        self.force_noise_scales=torch.full((self.num_envs,1),cfg.force_noise_std,device=self.device)
        self.force_noise_scales[self.nominal_dynamics_mask]=1.
        self._initialize_friction()
        self._initialize_response()
        self._initialize_payload_mass()
        self.force_bias = torch.zeros(self.num_envs,3,device=self.device)
        self.force_noise_sample = torch.zeros_like(self.force_bias)
        self.targets = self.robot.data.default_joint_pos.clone()
        self.command_pos = torch.zeros(self.num_envs,3,device=self.device)
        self.command_quat = torch.zeros(self.num_envs,4,device=self.device)
        self.command_initialized = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.command_cap_events=torch.zeros(self.num_envs,device=self.device)
        self.axial_lead_cap_events=torch.zeros(self.num_envs,device=self.device)
        self.last_axial_cap_active=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.last_control_axial_lead=torch.zeros(self.num_envs,device=self.device)
        self.episode_controller_steps=torch.zeros(self.num_envs,device=self.device)
        self.tip_offset = torch.tensor([0., 0., self.geometry['tip_offset']], device=self.device).repeat(self.num_envs, 1)
        self.nominal_tip_offset = self.tip_offset.clone()
        self.tip_offset[:,2] += self.grip_axial_offsets.to(self.device)
        self.hole_pos = torch.tensor(self.geometry['hole_center'], device=self.device).repeat(self.num_envs, 1)
        self.hole_estimation_bias = torch.zeros(self.num_envs,3,device=self.device)
        half = self.geometry['peg_width']/2
        self.peg_corners = torch.tensor([[x,y,0.] for x in [-half,half] for y in [-half,half]],
                                       device=self.device)
        self.hole_base_quat = torch.tensor(self.geometry.get('hole_quaternion',[1.,0.,0.,0.]),device=self.device)
        self.hole_quat = self.hole_base_quat.repeat(self.num_envs,1)
        self.down_quat = torch.tensor([0., 0., 1., 0.], device=self.device).repeat(self.num_envs, 1)
        self.hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ever_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.peak_force = torch.zeros(self.num_envs, device=self.device)
        self.contact_seen=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.contact_first_depth=torch.zeros(self.num_envs,device=self.device)
        self.retreat_after_contact=torch.zeros_like(self.contact_seen)
        self.release_after_contact=torch.zeros_like(self.contact_seen)
        self.peak_command_lead = torch.zeros(self.num_envs,device=self.device)
        self.end_depth = torch.zeros(self.num_envs,device=self.device)
        self.end_nominal_depth = torch.zeros(self.num_envs,device=self.device)
        self.end_command_lead = torch.zeros(self.num_envs,device=self.device)
        self.end_orientation_error = torch.zeros(self.num_envs,device=self.device)
        self.end_xy_error = torch.zeros(self.num_envs,device=self.device)
        self.end_shaft_fits = torch.zeros(self.num_envs,device=self.device)
        self.end_contact_force = torch.zeros(self.num_envs,device=self.device)
        self.end_hold_steps = torch.zeros(self.num_envs,device=self.device)
        self.end_down_step = torch.zeros(self.num_envs,device=self.device)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.end_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.total_steps = 0
        self.completed_episodes = 0
        self.completed_successes = 0
        print('PIPER_JOINTS', names, 'BODIES', self.robot.body_names, flush=True)

    def _initialize_payload_mass(self):
        if not hasattr(self,'expected_payload_mass'):
            return
        view = self.robot.root_physx_view
        # This installed API returns full tensors about COM in the link frame.
        # Author mass/principal axes in USD before startup instead of depending
        # on differing tensor setter frame conventions across PhysX versions.
        for name,actual,expected in [('mass',view.get_masses()[:,self.ee_idx],self.expected_payload_mass),
                ('com',view.get_coms()[:,self.ee_idx,:3],self.expected_payload_com),
                ('inertia',view.get_inertias()[:,self.ee_idx],self.expected_payload_inertia)]:
            if not torch.allclose(actual,expected,atol=1e-6):
                raise RuntimeError(f'PhysX payload {name} mismatch: max={float((actual-expected).abs().max())}, '
                                   f'actual={actual[0].tolist()}, expected={expected[0].tolist()}')
        print('PAYLOAD_PHYSICS_ASSIGNMENT_VERIFIED',self.cfg.peg_mass_min,self.cfg.peg_mass_max,
              'NON_PAYLOAD_PROPERTIES_UNEDITED',flush=True)

    def _initialize_grip_shapes(self):
        # These edits happen after USD cloning, before PhysX starts. Physics
        # replication must be disabled so each physical collider is parsed.
        import numpy as np
        from scipy.spatial.transform import Rotation
        from pxr import Gf, UsdGeom, UsdPhysics
        from payload_properties import replace_payload
        lo,hi = self.cfg.grip_axial_min,self.cfg.grip_axial_max
        self.grip_axial_offsets = torch.empty(self.num_envs).uniform_(lo,hi) if lo!=hi else torch.full((self.num_envs,),lo)
        if self.cfg.grip_endpoint_fraction and lo!=hi:
            endpoint_mask=torch.rand(self.num_envs)<self.cfg.grip_endpoint_fraction
            endpoints=torch.where(torch.rand(self.num_envs)<1.-self.cfg.grip_positive_endpoint_probability,lo,hi)
            self.grip_axial_offsets=torch.where(endpoint_mask,endpoints,self.grip_axial_offsets)
        mass_lo,mass_hi=self.cfg.peg_mass_min,self.cfg.peg_mass_max
        self.peg_masses=torch.empty(self.num_envs).uniform_(mass_lo,mass_hi) if mass_lo!=mass_hi else torch.full((self.num_envs,),mass_lo)
        self.nominal_replay_mask=torch.zeros(self.num_envs,dtype=torch.bool)
        if self.cfg.nominal_replay_fraction:
            self.nominal_replay_mask=torch.rand(self.num_envs)<self.cfg.nominal_replay_fraction
            self.grip_axial_offsets[self.nominal_replay_mask]=0.
            self.peg_masses[self.nominal_replay_mask]=.02
        if lo==hi==0. and mass_lo==mass_hi==.02:
            return
        nominal=UsdPhysics.MassAPI(self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/link6'))
        nominal_mass=float(nominal.GetMassAttr().Get())
        nominal_com=np.array(nominal.GetCenterOfMassAttr().Get(),dtype=float)
        nominal_axes=nominal.GetPrincipalAxesAttr().Get()
        nominal_rotation=Rotation.from_quat([*nominal_axes.GetImaginary(),nominal_axes.GetReal()]).as_matrix()
        nominal_inertia=nominal_rotation@np.diag(nominal.GetDiagonalInertiaAttr().Get())@nominal_rotation.T
        self.expected_payload_mass=torch.zeros(self.num_envs)
        self.expected_payload_com=torch.zeros(self.num_envs,3)
        self.expected_payload_inertia=torch.zeros(self.num_envs,9)
        for i,offset in enumerate(self.grip_axial_offsets.tolist()):
            mass,com,inertia=replace_payload(nominal_mass,nominal_com,nominal_inertia,float(self.peg_masses[i]),offset)
            principal,axes=np.linalg.eigh(inertia)
            if np.linalg.det(axes)<0:
                axes[:,0]*=-1
            quaternion=Rotation.from_matrix(axes).as_quat()
            api=UsdPhysics.MassAPI(self.sim.stage.GetPrimAtPath(f'/World/envs/env_{i}/Robot/link6'))
            api.GetMassAttr().Set(mass)
            api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com))
            api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*principal))
            api.GetPrincipalAxesAttr().Set(Gf.Quatf(float(quaternion[3]),Gf.Vec3f(*quaternion[:3])))
            self.expected_payload_mass[i]=mass
            self.expected_payload_com[i]=torch.from_numpy(com)
            self.expected_payload_inertia[i]=torch.from_numpy(inertia.reshape(9))
            if lo==hi==0.:
                continue
            for subtree in ['collisions','visuals']:
                path=f'/World/envs/env_{i}/Robot/link6/{subtree}/mesh_5'
                prim=self.sim.stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise RuntimeError(f'Merged peg shape not found: {path}')
                shape=UsdGeom.Xformable(prim)
                matrix=shape.GetLocalTransformation()
                position=matrix.ExtractTranslation()
                position[2]=.17+offset
                matrix.SetTranslateOnly(position)
                shape.MakeMatrixXform().Set(matrix)
                actual=shape.GetLocalTransformation().ExtractTranslation()[2]
                if abs(actual-(.17+offset))>1e-7:
                    raise RuntimeError('Peg collider translation did not take effect')
        print('GRIP_COLLIDER_USD_ASSIGNMENT_VERIFIED',lo,hi,flush=True)

    def verify_grip_raycast(self):
        if not self.grip_axial_offsets.any():
            return
        if self.device!='cpu':
            # In this image the CPU scene-query tree does not follow GPU
            # articulation poses, even with Fabric disabled. An independent
            # CPU-physics smoke run validates identical authored colliders.
            print('GRIP_RAYCAST_REQUIRES_SEPARATE_CPU_PHYSICS_PROBE',flush=True)
            return
        from omni.physx import get_physx_scene_query_interface
        query=get_physx_scene_query_interface()
        self.sim.step(render=False)
        self.robot.update(self.cfg.sim.dt)
        self.sim.render()
        pos=self.robot.data.body_pos_w[:,self.ee_idx]
        quat=self.robot.data.body_quat_w[:,self.ee_idx]
        axis=quat_apply(quat,torch.tensor([0.,0.,1.],device=self.device).expand(self.num_envs,-1))
        tip=pos+quat_apply(quat,self.tip_offset)
        for i in range(min(self.num_envs,16)):
            origin=(tip[i]+.02*axis[i]).cpu().tolist()
            direction=(-axis[i]).cpu().tolist()
            hits=[]
            def collect_hit(hit):
                if isinstance(hit,dict):
                    hits.append(dict(hit))
                else:
                    hits.append(dict(collision=str(hit.collision),distance=float(hit.distance)))
                return True
            # The initial ray can lie within a hole wall when XY reset noise
            # offsets the peg. Inspect all hits so a closer wall cannot hide
            # the peg geometry that this independent validation targets.
            query.raycast_all(tuple(origin),tuple(direction),.025,collect_hit)
            peg_hits=[hit for hit in hits if hit.get('collision')==
                      f'/World/envs/env_{i}/Robot/link6/collisions/mesh_5/box']
            if not peg_hits or min(abs(hit['distance']-.02) for hit in peg_hits)>.00005:
                raise RuntimeError(f'Physical peg tip raycast failed for environment {i}: {hits}; '
                                   f'origin={origin}, direction={direction}, body={pos[i].tolist()}')
        print('GRIP_PHYSICAL_RAYCAST_VERIFIED',min(self.num_envs,16),flush=True)

    def _initialize_friction(self):
        # A finite set of materials avoids exhausting PhysX's material table.
        # One factor per environment, shared by robot and all four hole walls,
        # is fixed for the run; it is hidden from both observation groups.
        lo, hi = self.cfg.friction_scale_min, self.cfg.friction_scale_max
        self.friction_scales = torch.ones(self.num_envs)
        if lo == hi == 1.:
            return
        buckets = torch.linspace(lo, hi, 16)
        self.friction_scales = buckets[torch.randint(0, 16, (self.num_envs,))]
        self.friction_scales[self.nominal_dynamics_mask.cpu()]=1.
        ids = torch.arange(self.num_envs, dtype=torch.int32)
        for asset in [self.robot, *self.walls]:
            material = asset.root_physx_view.get_material_properties().clone()
            material[:, :, :2] *= self.friction_scales[:, None, None]
            asset.root_physx_view.set_material_properties(material, ids)
            actual = asset.root_physx_view.get_material_properties()
            if not torch.allclose(actual, material, atol=1e-6):
                raise RuntimeError('PhysX material assignment did not take effect')
        print('FRICTION_ASSIGNMENT_VERIFIED', lo, hi, flush=True)

    def _initialize_response(self):
        lo, hi = self.cfg.stiffness_scale_min, self.cfg.stiffness_scale_max
        self.stiffness_scales = torch.ones(self.num_envs,device=self.device)
        if lo == hi == 1.:
            return
        self.stiffness_scales.uniform_(lo, hi) if lo != hi else self.stiffness_scales.fill_(lo)
        self.stiffness_scales[self.nominal_dynamics_mask]=1.
        gains = 800.*self.stiffness_scales[:,None].expand(-1,len(self.arm_ids))
        self.robot.write_joint_stiffness_to_sim(gains,joint_ids=self.arm_ids)
        actual = self.robot.root_physx_view.get_dof_stiffnesses()[:,self.arm_ids].to(self.device)
        if not torch.allclose(actual,gains,atol=1e-5):
            raise RuntimeError('PhysX joint stiffness assignment did not take effect')
        # Keep the implicit actuator's estimated-torque metadata consistent.
        self.robot.actuators['arm'].stiffness.copy_(gains)
        print('RESPONSE_ASSIGNMENT_VERIFIED',lo,hi,'DAMPING_UNCHANGED_45',flush=True)

    def _measured_force(self, force):
        # Cache noise per policy step so repeated reads of the same state agree.
        return (force+self.force_bias+self.force_noise_sample).clamp(-100,100)*.02

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._spawn_workspace()
        self._spawn_task_objects()

    def _spawn_workspace(self):
        ground = sim_utils.CuboidCfg(size=(100.,100.,.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.18,.18,.18)))
        ground.func('/World/ground',ground,translation=(0.,0.,-.01))
        pedestal = sim_utils.CuboidCfg(size=(.15,.15,.23),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25,.25,.25)))
        pedestal.func('/World/envs/env_0/Pedestal',pedestal,translation=(0.,0.,.115))
        sim_utils.CuboidCfg(size=(0.6, 0.5, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.3, 0.34))).func(
            '/World/envs/env_0/Table', sim_utils.CuboidCfg(size=(0.6, 0.5, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.3, 0.34))),
            translation=(0.45, 0., 0.06))

    def _spawn_task_objects(self):
        # Four separate convex walls preserve the hole, unlike a single convex hull.
        self.wall_offsets = self.hole_geometry['wall_offsets']
        sizes = self.hole_geometry['wall_sizes']
        self.walls = []
        from scipy.spatial.transform import Rotation
        hq=self.geometry.get('hole_quaternion',[1.,0.,0.,0.])
        hole_rotation=Rotation.from_quat([*hq[1:],hq[0]])
        for i, (offset, size) in enumerate(zip(self.wall_offsets, sizes)):
            wall = RigidObject(RigidObjectCfg(prim_path=f'/World/envs/env_.*/Wall{i}',
                spawn=sim_utils.CuboidCfg(size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=self.hole_geometry['contact_offset'], rest_offset=0.),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.4, dynamic_friction=0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.75))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(a+b for a,b in zip(self.geometry['hole_center'], hole_rotation.apply(offset))),rot=tuple(hq))))
            self.walls.append(wall)
            self.scene.rigid_objects[f'wall{i}'] = wall
        self.contact = ContactSensor(ContactSensorCfg(
            prim_path='/World/envs/env_.*/Robot/link6', update_period=0., history_length=4))
        self.scene.sensors['peg_contact'] = self.contact
        self.scene.clone_environments(copy_from_source=False)
        self._initialize_grip_shapes()
        if self.device == 'cpu' or not self.cfg.scene.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        self.scene.articulations['robot'] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.)
        light.func('/World/Light', light)

    def _geometry(self):
        pos = self.robot.data.body_pos_w[:, self.ee_idx]
        quat = self.robot.data.body_quat_w[:, self.ee_idx]
        tip = pos + quat_apply(quat, self.tip_offset) - self.scene.env_origins
        relative = quat_apply_inverse(self.hole_quat, tip - self.hole_pos)
        _, orientation = compute_pose_error(pos, quat, pos,
            quat_mul(self.hole_quat, self.down_quat), rot_error_type='axis_angle')
        history = self.contact.data.net_forces_w_history[:,:,0,:]
        peak_index = history.norm(dim=-1).argmax(dim=1)
        force = history[torch.arange(self.num_envs,device=self.device),peak_index]
        if self.cfg.action_frame=='hole':
            orientation=quat_apply_inverse(self.hole_quat,orientation)
            force=quat_apply_inverse(self.hole_quat,force)
        return relative, orientation, force

    def _pre_physics_step(self, actions):
        # Fresh counts every step avoid stale episode records and allow the
        # logger's mean(success_count)/mean(episode_count) to weight episodes.
        self.extras['episode'] = dict(success_count=0.,episode_count=0.,
                                     peak_force_total_n=0.,return_total=0.)
        if self.force_noise_scales.any():
            self.force_noise_sample.normal_().mul_(self.force_noise_scales)
        relative, orientation, _ = self._geometry()
        aligned = torch.exp(-relative[:,:2].norm(dim=-1)/alignment_length_scale(self.cfg.reward_profile)) * torch.exp(-orientation.norm(dim=-1)/orientation_length_scale(self.cfg.reward_profile))
        progress = ((-relative[:,2]+.025)/.05).clamp(0.,1.)
        self.previous_potential = 10.*(1.5*aligned+3.*aligned*progress)
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clamp(-1., 1.).clone()
        applied = self.action_delay.push(self.actions)
        command = applied * torch.tensor([.002, .002, .002, .012, .012, .012], device=self.device)
        if self.cfg.action_frame=='hole':
            command=torch.cat((quat_apply(self.hole_quat,command[:,:3]),quat_apply(self.hole_quat,command[:,3:])),dim=-1)
        pos = self.robot.data.body_pos_w[:, self.ee_idx]
        quat = self.robot.data.body_quat_w[:, self.ee_idx]
        if self.cfg.near_contact_position_step or self.cfg.near_contact_rotation_step:
            nominal_tip=pos+quat_apply(quat,self.nominal_tip_offset)-self.scene.env_origins
            estimated_relative=quat_apply_inverse(self.hole_quat,nominal_tip-self.hole_pos)+self.hole_estimation_bias
            near=estimated_relative[:,2].abs()<=self.cfg.near_contact_band
            limited_pos=limit_vector(command[:,:3],self.cfg.near_contact_position_step)
            limited_rot=limit_vector(command[:,3:],self.cfg.near_contact_rotation_step)
            changed=((command[:,:3]-limited_pos).norm(dim=-1)>1e-7)|((command[:,3:]-limited_rot).norm(dim=-1)>1e-7)
            self.command_cap_events+=(near&changed).float()
            command=torch.where(near[:,None],torch.cat((limited_pos,limited_rot),dim=-1),command)
        self.end_down_step.copy_(-quat_apply_inverse(self.hole_quat,command[:,:3])[:,2])
        reset_mask = ~self.command_initialized
        self.command_pos[reset_mask] = pos[reset_mask]
        self.command_quat[reset_mask] = quat[reset_mask]
        self.command_initialized[:] = True
        from isaaclab.utils.math import apply_delta_pose
        self.command_pos, self.command_quat = apply_delta_pose(self.command_pos,self.command_quat,command)
        self.episode_controller_steps+=1
        if self.cfg.command_axial_lead_bound:
            up_axis=quat_apply(self.hole_quat,torch.tensor([0.,0.,1.],device=self.device).expand(self.num_envs,-1))
            self.command_pos,active=limit_downward_lead(self.command_pos,pos,up_axis,self.cfg.command_axial_lead_bound)
            self.axial_lead_cap_events+=active.float()
            lead=-((self.command_pos-pos)*up_axis).sum(-1)
            self.last_axial_cap_active=active
            self.last_control_axial_lead=lead
            if (lead>self.cfg.command_axial_lead_bound+1e-6).any():
                raise RuntimeError('Axial command lead bound violated before IK')
        if self.cfg.command_position_bound or self.cfg.command_rotation_bound:
            error_pos,error_rot=compute_pose_error(pos,quat,self.command_pos,self.command_quat,rot_error_type='axis_angle')
            bounded_pos=limit_vector(error_pos,self.cfg.command_position_bound)
            bounded_rot=limit_vector(error_rot,self.cfg.command_rotation_bound)
            saturated=(error_pos-bounded_pos).norm(dim=-1)>1e-7
            saturated|=(error_rot-bounded_rot).norm(dim=-1)>1e-7
            self.command_cap_events+=saturated.float()
            # Only encoder/FK feedback and the policy's command are used.
            # No true peg pose, hole bias, material, or force truth enters this layer.
            self.command_pos,self.command_quat=apply_delta_pose(pos,quat,torch.cat((bounded_pos,bounded_rot),dim=-1))
            if self.cfg.command_position_bound and (bounded_pos.norm(dim=-1)>self.cfg.command_position_bound+1e-6).any():
                raise RuntimeError('Translation tracking-command bound violated')
            if self.cfg.command_rotation_bound and (bounded_rot.norm(dim=-1)>self.cfg.command_rotation_bound+1e-6).any():
                raise RuntimeError('Rotation tracking-command bound violated')
        ik_pos,ik_quat=pos,quat
        ik_command_pos,ik_command_quat=self.command_pos,self.command_quat
        if getattr(self,'ik_in_root_frame',False):
            from isaaclab.utils.math import subtract_frame_transforms
            root_pos=self.robot.data.root_pos_w;root_quat=self.robot.data.root_quat_w
            ik_pos,ik_quat=subtract_frame_transforms(root_pos,root_quat,pos,quat)
            ik_command_pos,ik_command_quat=subtract_frame_transforms(root_pos,root_quat,self.command_pos,self.command_quat)
        self.ik.set_command(torch.cat((ik_command_pos,ik_command_quat),dim=-1))
        jac = self.robot.root_physx_view.get_jacobians()[:, self.jac_idx, :, self.arm_ids]
        if getattr(self,'jacobian_about_com',False):
            offset=quat_apply(quat,self._ik_com_offset)
            if self.ik_in_root_frame:offset=quat_apply_inverse(self.robot.data.root_quat_w,offset)
            jac=jac.clone()
            jac[:,:3,:]+=torch.linalg.cross(offset[:,None,:].expand(-1,len(self.arm_ids),-1),jac[:,3:,:].transpose(1,2),dim=-1).transpose(1,2)
        q = self.robot.data.joint_pos[:, self.arm_ids]
        target = self.ik.compute(ik_pos, ik_quat, jac, q)
        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_ids]
        self.targets[:, self.arm_ids] = torch.maximum(torch.minimum(
            q + (target-q).clamp(-.06, .06), limits[:,:,1]), limits[:,:,0])
        self.total_steps += 1

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets)

    def _get_observations(self):
        relative, orientation, force = self._geometry()
        pos = self.robot.data.body_pos_w[:,self.ee_idx]
        quat = self.robot.data.body_quat_w[:,self.ee_idx]
        command_error_pos, command_error_rot = compute_pose_error(
            pos,quat,self.command_pos,self.command_quat,rot_error_type='axis_angle')
        command_error_pos = torch.where(self.command_initialized[:,None],command_error_pos,0.)
        command_error_rot = torch.where(self.command_initialized[:,None],command_error_rot,0.)
        if self.cfg.action_frame=='hole':
            command_error_pos=quat_apply_inverse(self.hole_quat,command_error_pos)
            command_error_rot=quat_apply_inverse(self.hole_quat,command_error_rot)
        # The actor uses the nominal grasp transform, as on hardware without
        # direct object pose tracking. The critic and physical labels use truth.
        nominal_tip=pos+quat_apply(quat,self.nominal_tip_offset)-self.scene.env_origins
        nominal_relative=quat_apply_inverse(self.hole_quat,nominal_tip-self.hole_pos)
        obs = torch.cat(((nominal_relative+self.hole_estimation_bias) * 20., orientation,
            self.robot.data.joint_pos[:, self.arm_ids],
            self.robot.data.joint_vel[:, self.arm_ids] * .1, self.actions,
            self._measured_force(force),
            command_error_pos*20.,command_error_rot), dim=-1)
        observations = {'policy': obs}
        if self.cfg.privileged_critic:
            truth = obs.clone()
            truth[:,:3] = relative*20.
            truth[:,24:27] = force.clamp(-100,100)*.02
            observations['critic'] = truth
        return observations

    def _shaft_fits(self, relative):
        # Check the four peg edges both at the tip and at the opening plane.
        # This catches tilted shafts whose tip alone lies within the hole.
        quat = self.robot.data.body_quat_w[:,self.ee_idx]
        world = quat_apply(quat[:,None,:].expand(-1,4,-1),
                           self.peg_corners[None,:,:].expand(self.num_envs,-1,-1))
        corners = relative[:,None,:]+quat_apply_inverse(
            self.hole_quat[:,None,:].expand(-1,4,-1),world)
        axis = quat_apply_inverse(self.hole_quat,
            quat_apply(quat,torch.tensor([0.,0.,1.],device=self.device).expand(self.num_envs,-1)))
        denominator = torch.where(axis[:,2].abs()>.1,axis[:,2],torch.ones_like(axis[:,2]))
        at_opening = corners-axis[:,None,:]*(corners[:,:,2]/denominator[:,None])[:,:,None]
        maximum = torch.maximum(corners[:,:,:2].abs().amax(dim=(1,2)),
                                at_opening[:,:,:2].abs().amax(dim=(1,2)))
        boundary = self.cfg.hole_width/2-self.hole_geometry['containment_margin']
        return (axis[:,2]<-.9) & (maximum<boundary)

    def _get_dones(self):
        relative, orientation, force = self._geometry()
        depth = -relative[:, 2]
        force_norm = force.norm(dim=-1)
        first_contact=(force_norm>=5.)&~self.contact_seen
        self.contact_first_depth[first_contact]=depth[first_contact]
        self.contact_seen|=force_norm>=5.
        self.retreat_after_contact|=(self.contact_seen & (self.contact_first_depth-depth>=.002)
                                    & (self.end_down_step<-.00005))
        self.release_after_contact|=self.contact_seen&(force_norm<2.)
        pos=self.robot.data.body_pos_w[:,self.ee_idx]
        quat=self.robot.data.body_quat_w[:,self.ee_idx]
        nominal_tip=pos+quat_apply(quat,self.nominal_tip_offset)-self.scene.env_origins
        nominal_relative=quat_apply_inverse(self.hole_quat,nominal_tip-self.hole_pos)
        command_error=quat_apply_inverse(self.hole_quat,self.command_pos-pos)
        self.end_depth.copy_(depth)
        self.end_nominal_depth.copy_(-nominal_relative[:,2])
        self.end_command_lead.copy_(torch.where(self.command_initialized,-command_error[:,2],0.))
        self.end_orientation_error.copy_(orientation.norm(dim=-1))
        self.peak_command_lead=torch.maximum(self.peak_command_lead,self.end_command_lead)
        valid = (relative[:, :2].abs().amax(-1) < self.hole_geometry['success_xy_tolerance'])
        valid &= (orientation.norm(dim=-1) < .12) & self._shaft_fits(relative)
        valid &= (depth >= .025) & (depth <= .055) & (force_norm < 40.)
        self.hold = torch.where(valid, self.hold+1, torch.zeros_like(self.hold))
        self.end_xy_error.copy_(relative[:,:2].abs().amax(-1))
        self.end_shaft_fits.copy_(self._shaft_fits(relative).float())
        self.end_contact_force.copy_(force_norm)
        self.end_hold_steps.copy_(self.hold.float())
        success = self.hold >= 8
        self.ever_success |= success
        self.peak_force = torch.maximum(self.peak_force, force_norm)
        failure = (force_norm > 100.) | (relative[:,:2].norm(dim=-1) > .12)
        failure |= (relative[:,2] < -.06) | (relative[:,2] > .15)
        timeout = self.episode_length_buf >= self.max_episode_length-1
        self.end_reason[:] = 0
        self.end_reason[timeout] = 3
        self.end_reason[failure] = 2
        self.end_reason[force_norm > 100.] = 1
        self.end_reason[success] = 4
        return success | failure, timeout

    def _get_rewards(self):
        relative, orientation, force = self._geometry()
        xy = relative[:, :2].norm(dim=-1)
        depth = -relative[:, 2]
        aligned = torch.exp(-xy / alignment_length_scale(self.cfg.reward_profile)) * torch.exp(-orientation.norm(dim=-1) / orientation_length_scale(self.cfg.reward_profile))
        progress = ((depth + .025) / .05).clamp(0., 1.)
        # Discounted potential shaping rewards improvement rather than dwelling
        # at the rim. Terminal potential is zero for this finite-horizon task.
        potential = 10.*(1.5*aligned+3.*aligned*progress)
        potential = torch.where(self.reset_buf,0.,potential)
        reward = .99*potential-self.previous_potential-.02
        reward -= .002*(self.actions**2).sum(-1) + .005*((self.actions-self.previous_actions)**2).sum(-1)
        force_cost=contact_cost(force.norm(dim=-1),self.cfg.reward_profile)
        reward -= force_cost
        success = self.hold >= 8
        reward += 100.*success.float()
        reward -= 10.*(self.reset_terminated & ~success).float()
        reward -= 5.*self.reset_time_outs.float()
        self.episode_returns += reward
        self.extras['log'] = {'insertion/alignment': aligned.mean(),
            'insertion/depth_m': depth.mean(), 'insertion/contact_force_n': force.norm(dim=-1).mean(),
            'insertion/contact_cost':force_cost.mean(),
            'insertion/success_step': (self.hold >= 8).float().mean()}
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if hasattr(self, 'ever_success'):
            # Preserve the final physics step before automatic reset clears it.
            self.extras['terminal'] = {
                'env_ids': env_ids.clone(),
                'success': self.ever_success[env_ids].clone(),
                'peak_force_n': self.peak_force[env_ids].clone(),
                'return': self.episode_returns[env_ids].clone(),
                'seconds': self.episode_length_buf[env_ids].float().clone()*self.step_dt}
            self.extras['terminal']['delay_steps'] = self.action_delay.steps[env_ids].clone()
            self.extras['terminal']['end_reason'] = self.end_reason[env_ids].clone()
            self.extras['terminal']['friction_scale'] = self.friction_scales[env_ids.cpu()].to(self.device)
            self.extras['terminal']['stiffness_scale'] = self.stiffness_scales[env_ids].clone()
            self.extras['terminal']['peg_mass_kg'] = self.peg_masses[env_ids.cpu()].to(self.device)
            self.extras['terminal']['grip_axial_m'] = self.grip_axial_offsets[env_ids.cpu()].to(self.device)
            self.extras['terminal']['contact_seen']=self.contact_seen[env_ids].clone()
            self.extras['terminal']['retreat_after_contact']=self.retreat_after_contact[env_ids].clone()
            self.extras['terminal']['release_after_contact']=self.release_after_contact[env_ids].clone()
            self.extras['terminal']['command_cap_fraction']=(self.command_cap_events[env_ids]/self.episode_controller_steps[env_ids].clamp_min(1)).clone()
            self.extras['terminal']['axial_command_cap_fraction']=(self.axial_lead_cap_events[env_ids]/self.episode_controller_steps[env_ids].clamp_min(1)).clone()
            for name,values in [('true_depth_m',self.end_depth),('nominal_depth_m',self.end_nominal_depth),
                                ('command_lead_m',self.end_command_lead),('peak_command_lead_m',self.peak_command_lead),
                                ('orientation_error_rad',self.end_orientation_error),
                                ('xy_error_m',self.end_xy_error),('shaft_fits',self.end_shaft_fits),
                                ('contact_force_n',self.end_contact_force),('hold_steps',self.end_hold_steps),
                                ('down_step_m',self.end_down_step)]:
                self.extras['terminal'][name]=values[env_ids].clone()
            self.completed_episodes += len(env_ids)
            self.completed_successes += int(self.ever_success[env_ids].sum())
            if len(env_ids):
                self.extras['episode'] = {
                    'success_count': self.ever_success[env_ids].float().sum(),
                    'episode_count': float(len(env_ids)),
                    'peak_force_total_n': self.peak_force[env_ids].sum(),
                    'return_total': self.episode_returns[env_ids].sum()}
        super()._reset_idx(env_ids)
        n = len(env_ids)
        q = self.robot.data.default_joint_pos[env_ids].clone()
        q[:, self.arm_ids] += (torch.rand(n, 6, device=self.device)*2.-1.)*self.cfg.joint_noise
        qdot = torch.zeros_like(q)
        self.targets[env_ids] = q
        self.robot.write_joint_state_to_sim(q, qdot, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.hole_pos[env_ids] = torch.tensor(self.geometry['hole_center'], device=self.device)
        xy_noise=(torch.rand(n,2,device=self.device)*2.-1.)*self.cfg.hole_xy_noise
        local_noise=torch.cat((xy_noise,torch.zeros(n,1,device=self.device)),dim=-1)
        self.hole_pos[env_ids] += quat_apply(self.hole_base_quat.repeat(n,1),local_noise)
        self.hole_estimation_bias[env_ids] = 0.
        self.force_bias[env_ids] = 0.
        if self.cfg.force_bias_bound:
            self.force_bias[env_ids] = (torch.rand(n,3,device=self.device)*2.-1.)*self.cfg.force_bias_bound
        nominal_ids=env_ids[self.nominal_dynamics_mask[env_ids]]
        self.force_bias[nominal_ids]=0.
        self.force_noise_sample[env_ids] = 0.
        if self.force_noise_scales.any():
            self.force_noise_sample[env_ids] = torch.randn(n,3,device=self.device)*self.force_noise_scales[env_ids]
        if self.cfg.observation_bias:
            self.hole_estimation_bias[env_ids,:2] = (
                torch.rand(n,2,device=self.device)*2.-1.)*self.cfg.observation_bias
        yaw = (torch.rand(n,device=self.device)*2.-1.)*self.cfg.hole_yaw_noise
        self.hole_quat[env_ids] = quat_mul(self.hole_base_quat.repeat(n,1),quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw))
        for wall, offset in zip(self.walls, self.wall_offsets):
            shifted = quat_apply(self.hole_quat[env_ids], torch.tensor(offset,device=self.device).repeat(n,1))
            pose = torch.cat((self.hole_pos[env_ids] + self.scene.env_origins[env_ids] + shifted,
                              self.hole_quat[env_ids]), dim=-1)
            wall.write_root_pose_to_sim(pose, env_ids=env_ids)
            # The walls are kinematic; PhysX rejects dynamic velocity setters.
        self.actions[env_ids] = 0.
        self.action_delay.reset(env_ids, self.cfg.action_delay_min, self.cfg.action_delay_max)
        self.action_delay.steps[nominal_ids]=0
        self.previous_actions[env_ids] = 0.
        self.hold[env_ids] = 0
        self.ever_success[env_ids] = False
        self.peak_force[env_ids] = 0.
        self.contact_seen[env_ids]=False
        self.contact_first_depth[env_ids]=0.
        self.retreat_after_contact[env_ids]=False
        self.release_after_contact[env_ids]=False
        self.peak_command_lead[env_ids] = 0.
        self.command_cap_events[env_ids]=0.
        self.axial_lead_cap_events[env_ids]=0.
        self.episode_controller_steps[env_ids]=0.
        self.episode_returns[env_ids] = 0.
        self.ik.reset(env_ids)
        self.command_initialized[env_ids] = False
