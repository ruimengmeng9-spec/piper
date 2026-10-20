"""Limit pose tracking vectors using encoder-derived feedback only."""


def limit_vector(vector, bound):
    if bound<=0:
        return vector
    norm=vector.norm(dim=-1,keepdim=True)
    scale=(bound/norm.clamp_min(1e-12)).clamp(max=1.)
    return vector*scale


def limit_downward_lead(target, actual, up_axis, bound):
    """Only remove excessive target advance along the known insertion axis."""
    lead=-((target-actual)*up_axis).sum(-1)
    if bound<=0:
        return target,lead>float('inf')
    excess=(lead-bound).clamp(min=0.)
    return target+excess[:,None]*up_axis,excess>0


if __name__=='__main__':
    import torch
    x=torch.tensor([[0.,0.,0.],[.0003,.0004,0.],[.003,.004,0.]])
    expected=torch.tensor([[0.,0.,0.],[.0003,.0004,0.],[.0009,.0012,0.]])
    assert torch.allclose(limit_vector(x,.0015),expected,atol=1e-9)
    assert torch.equal(limit_vector(x,0.),x)
    assert (limit_vector(x,.0015).norm(dim=-1)<=.0015+1e-9).all()
    assert torch.allclose(limit_vector(torch.tensor([[.03,0.,0.]]),.01),torch.tensor([[.01,0.,0.]]))
    actual=torch.zeros(3,3)
    axis=torch.tensor([[0.,0.,1.]]).expand(3,-1)
    targets=torch.tensor([[.006,.007,-.012],[.002,-.003,-.004],[.004,.005,.02]])
    limited,active=limit_downward_lead(targets,actual,axis,.008)
    assert torch.allclose(limited,torch.tensor([[.006,.007,-.008],[.002,-.003,-.004],[.004,.005,.02]]))
    assert active.tolist()==[True,False,False]
    assert torch.equal(limit_downward_lead(targets,actual,axis,0)[0],targets)
    # The same bound must hold for a non-world-Z insertion axis.
    axis=torch.tensor([[0.,.6,.8]]).expand(3,-1)
    limited,_=limit_downward_lead(targets,actual,axis,.008)
    assert (-((limited-actual)*axis).sum(-1)<=.008+1e-8).all()
    correction=limited-targets
    assert torch.allclose(correction,((correction*axis).sum(-1))[:,None]*axis,atol=1e-8)
    print('COMMAND_BOUND_VECTOR_CHECK_PASSED')
