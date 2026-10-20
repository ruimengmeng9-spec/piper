"""Contact-force costs; physical success and termination stay unchanged."""


def contact_cost(force_norm, profile='potential_v2'):
    excess=(force_norm-15.).clamp(min=0.,max=150.)
    cost=.02*excess
    if profile in ('contact_v9','alignment_v11','contact_v12'):
        cost=cost+.002*excess.square()
    elif profile!='potential_v2':
        raise ValueError(f'Unknown reward profile: {profile}')
    return cost


def alignment_length_scale(profile):
    """Sharper potential only; terminal tolerances and actor features are unchanged."""
    if profile == 'alignment_v11':
        return .002
    if profile == 'contact_v12':
        return .001
    if profile in ('potential_v2', 'contact_v9'):
        return .008
    raise ValueError(f'Unknown reward profile: {profile}')


def orientation_length_scale(profile):
    return .10 if profile=='contact_v12' else .20


if __name__=='__main__':
    import torch
    force=torch.tensor([0.,15.,40.,80.,100.,165.,200.])
    assert torch.allclose(contact_cost(force),torch.tensor([0.,0.,.5,1.3,1.7,3.,3.]))
    assert torch.allclose(contact_cost(force,'contact_v9'),torch.tensor([0.,0.,1.75,9.75,16.15,48.,48.]))
    assert torch.equal(contact_cost(force,'alignment_v11'),contact_cost(force,'contact_v9'))
    assert alignment_length_scale('alignment_v11') == .002
    assert alignment_length_scale('contact_v12') == .001
    assert orientation_length_scale('contact_v12') == .10
    assert torch.equal(contact_cost(force,'contact_v12'),contact_cost(force,'alignment_v11'))
    assert (contact_cost(force,'contact_v9')[1:]>=contact_cost(force,'contact_v9')[:-1]).all()
    print('CONTACT_REWARD_BOUNDARY_CHECK_PASSED')
