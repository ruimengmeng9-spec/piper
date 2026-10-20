"""Train-only 25% reset-boundary mixture, strictly inside original randomization box."""
import types,torch
from isaaclab.utils.math import quat_apply
def install(raw):
 reset=raw._reset_idx;device=raw.device
 def magnitude(n,d,bound):
  sign=torch.randint(0,2,(n,d),device=device)*2-1
  return sign*(.75+.25*torch.rand(n,d,device=device))*bound
 def on_reset(this,env_ids):
  reset(env_ids);ids=this.robot._ALL_INDICES if env_ids is None else env_ids;ids=ids[ids%4==0]
  if not len(ids):return
  n=len(ids);q=this.robot.data.default_joint_pos[ids].clone();noise=magnitude(n,6,this.cfg.joint_noise);assert (noise.abs()<=this.cfg.joint_noise+1e-8).all();q[:,this.arm_ids]+=noise
  this.targets[ids]=q;this.robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids);this.robot.set_joint_position_target(q,env_ids=ids)
  xy=magnitude(n,2,this.cfg.hole_xy_noise);local=torch.cat((xy,torch.zeros(n,1,device=device)),-1);this.hole_pos[ids]=torch.tensor(this.geometry['hole_center'],device=device)+quat_apply(this.hole_base_quat.expand(n,-1),local)
  this.hole_estimation_bias[ids,:2]=magnitude(n,2,this.cfg.observation_bias)
  for wall,offset in zip(this.walls,this.wall_offsets):
   shifted=quat_apply(this.hole_quat[ids],torch.tensor(offset,device=device).expand(n,-1));wall.write_root_pose_to_sim(torch.cat((this.hole_pos[ids]+this.scene.env_origins[ids]+shifted,this.hole_quat[ids]),-1),env_ids=ids)
 raw._reset_idx=types.MethodType(on_reset,raw)
