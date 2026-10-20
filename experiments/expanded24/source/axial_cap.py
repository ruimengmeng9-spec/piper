"""Prioritize axial target continuity without exceeding the original tracking ball."""
import torch
def project(tracking,axis,radius):
 z=(tracking*axis).sum(-1,keepdim=True).clamp(-radius,radius)
 xy=tracking-(tracking*axis).sum(-1,keepdim=True)*axis
 budget=((radius-z.abs()).clamp_min(0.)*(radius+z.abs())).sqrt()
 xy=xy*(budget/xy.norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp(max=1.)
 result=xy+z*axis
 assert torch.isfinite(result).all() and (result.norm(dim=-1)<=radius+1e-6).all()
 return result
