import torch
from torch import nn

def history_inputs(C, public):
    assert C.shape[-1] == 75 and public.shape == (*C.shape[:-1],33)
    measured = public[...,24:27]
    assert torch.isfinite(measured).all() and (measured.abs() <= 3.000001).all()
    X = torch.cat((C[..., :33], C[..., 66:75]), -1)
    X[...,24:27] = measured
    return X

def privileged_delta6(public, truth6):
    """Same pre-action observation; fixed physical scales, no fit statistics."""
    assert public.shape[-1] == 33 and truth6.shape == (*public.shape[:-1],6)
    assert torch.isfinite(public).all() and torch.isfinite(truth6).all()
    assert torch.equal((public[...,24:27].abs()>2).any(-1), (truth6[...,3:].abs()>2).any(-1))
    return torch.cat(((truth6[...,:3]-public[...,:3])/.02, (truth6[...,3:]-public[...,24:27])/.04),-1)

def controller_state_inputs(ema, trigger):
    assert ema.shape[-1] == 3 and trigger.shape == ema.shape[:-1]
    assert ema.dtype == trigger.dtype == torch.float32 and ema.device == trigger.device
    assert torch.isfinite(ema).all() and torch.isfinite(trigger).all()
    assert (ema.abs()<=100.00001).all() and ((trigger>=-.015001)&(trigger<=.020001)).all()
    return torch.cat((ema*.01,trigger[...,None]*50.),-1)

def queue_inputs(buffer,steps):
    assert buffer.ndim==3 and buffer.shape[0]==3 and buffer.shape[-1]==6 and steps.shape==(buffer.shape[1],)
    assert torch.isfinite(buffer).all() and (buffer.abs()<=1.).all() and ((steps>=0)&(steps<=2)).all()
    return torch.cat((torch.where((steps>=1)[:,None],buffer[0],0.),torch.where((steps>=2)[:,None],buffer[1],0.)),-1)

def discounted_remaining_fraction(clock):
    """Known finite horizon only: sum(gamma**k,k<remaining)/sum(gamma**k,k<720)."""
    assert clock.dtype==torch.float32 and torch.isfinite(clock).all()
    assert ((clock>=0.)&(clock<=1.)).all(), 'Current controller clock must stay in [0,1]'
    remaining=(1.-clock)*720.
    return -torch.expm1(remaining*__import__('math').log(.99))/(1.-.99**720)

def valid_force_difference(current,previous):
    """Only two available pre-action force samples; normalized difference, not N/s."""
    assert current.shape==previous.shape and current.shape[-1]==3
    assert current.dtype==previous.dtype==torch.float32 and current.device==previous.device
    assert torch.isfinite(current).all() and torch.isfinite(previous).all()
    assert (current.abs()<=3.000001).all() and (previous.abs()<=3.000001).all()
    valid=(current.abs()<=2.).all(-1)&(previous.abs()<=2.).all(-1)
    return torch.where(valid[...,None],(current-previous)*.5,torch.zeros_like(current))

def force_history_inputs(current,previous):
    """Half-level minus quarter valid increment: two-sample mean at half scale."""
    delta=valid_force_difference(current,previous)
    available=(current.abs()<=2.).all(-1)
    return torch.where(available[...,None],current*.5-delta*.5,torch.zeros_like(current))

def encoded_force_sequence(FX):
    assert FX.ndim==3 and len(FX)>0
    previous=torch.cat((torch.full_like(FX[:1],3.),FX[:-1]),0)
    return force_history_inputs(FX,previous)

class HistoryValue(nn.Module):
    def __init__(self, static):
        super().__init__()
        self.static = static
        device = next(static.parameters()).device
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98443)
            self.memory = nn.GRU(42, 32, 1).to(device)
            self.projection = nn.Linear(32, 1, bias=False).to(device)
            nn.init.zeros_(self.projection.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98449)
            self.retry3_projection = nn.Linear(32, 1, bias=False).to(device)
            nn.init.zeros_(self.retry3_projection.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98461)
            self.privileged = nn.Linear(6,64,bias=False).to(device)
            nn.init.zeros_(self.privileged.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98467)
            self.force_memory = nn.GRU(3,16,1).to(device)
            self.force_projection = nn.Linear(16,1,bias=False).to(device)
            self.force_retry3_projection = nn.Linear(16,1,bias=False).to(device)
            nn.init.zeros_(self.force_projection.weight)
            nn.init.zeros_(self.force_retry3_projection.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98473)
            self.latency = nn.Linear(1,64,bias=False).to(device)
            nn.init.zeros_(self.latency.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98479)
            self.fusion = nn.Linear(48,64,bias=False).to(device)
            nn.init.zeros_(self.fusion.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98483)
            self.release_state = nn.Linear(1,64,bias=False).to(device)
            nn.init.zeros_(self.release_state.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98489)
            self.controller_state = nn.Linear(4,64,bias=False).to(device)
            nn.init.zeros_(self.controller_state.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98503)
            self.queue = nn.Linear(12,64,bias=False).to(device)
            nn.init.zeros_(self.queue.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98509)
            self.joint_target = nn.Linear(6,64,bias=False).to(device)
            nn.init.zeros_(self.joint_target.weight)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(98513)
            self.discount_horizon=nn.Linear(1,64,bias=False).to(device)
            nn.init.zeros_(self.discount_horizon.weight)
        self.train(static.training)

    def forward(self, C):
        raise RuntimeError('History critic requires explicit complete history or hidden state')

    def combine(self, C, H, E=None, L=None, R=None, S=None, Q=None,J=None):
        if L is None:
            assert not self.latency.weight.detach().any(), 'Trained latency critic requires explicit preaction L1'
        else:
            assert L.shape == (*C.shape[:-1],1) and torch.isfinite(L).all()
            assert ((L==0.) | (L==.5) | (L==1.)).all()
        if E is None:
            assert not self.privileged.weight.detach().any(), 'Trained privileged critic requires explicit causal E6'
            first = self.static[0](C.reshape(-1,75))
        else:
            assert E.shape == (*C.shape[:-1],6) and torch.isfinite(E).all()
            first = self.static[0](C.reshape(-1,75)) + self.privileged(E.reshape(-1,6))
        if L is not None:
            first = first + self.latency(L.reshape(-1,1))
        if R is None:
            assert not self.release_state.weight.detach().any(), 'Trained release-state critic requires explicit preaction R1'
        else:
            assert R.shape == (*C.shape[:-1],1) and torch.isfinite(R).all()
            assert ((R>=0)&(R<=1)&(R*8==(R*8).round())).all()
            first = first + self.release_state(R.reshape(-1,1))
        assert H.shape == (*C.shape[:-1],48) and torch.isfinite(H).all()
        first = first + self.fusion(H.reshape(-1,48))
        if S is None:
            assert not self.controller_state.weight.detach().any(), 'Trained controller-state critic requires explicit prephysics S4'
        else:
            assert S.shape == (*C.shape[:-1],4) and torch.isfinite(S).all()
            assert (S[...,:3].abs()<=1.000001).all() and ((S[...,3]>=-.750051)&(S[...,3]<=1.000051)).all()
            first = first + self.controller_state(S.reshape(-1,4))
        if Q is None:
            assert not self.queue.weight.detach().any(), 'Trained queue critic requires explicit raw predecision Q12'
        else:
            assert Q.shape==(*C.shape[:-1],12) and Q.dtype==C.dtype and torch.isfinite(Q).all() and (Q.abs()<=1.).all()
            assert L is not None and not Q[L[...,0]==0.].any() and not Q[...,6:][L[...,0]<1.].any()
            first = first + self.queue(Q.reshape(-1,12))
        if J is None:
            assert not self.joint_target.weight.detach().any(), 'Trained J6 critic requires explicit predecision joint target error'
        else:
            assert J.shape==(*C.shape[:-1],6) and J.dtype==C.dtype and torch.isfinite(J).all()
            first = first + self.joint_target(J.reshape(-1,6))
        # C69 is the current prephysics controller clock, already in actor-independent context.
        remaining=discounted_remaining_fraction(C[...,69:70])
        first=first+self.discount_horizon(remaining.reshape(-1,1))
        original = self.static[1:](first).reshape(*C.shape[:-1],2)
        assert H.shape == (*C.shape[:-1],48) and torch.isfinite(H).all()
        public_h, force_h = H[...,:32], H[...,32:]
        # C70 is the CURRENT supervisor attempts/3, never retrospective phase.
        residual = torch.where(C[...,70:71] >= 1., self.retry3_projection(public_h), self.projection(public_h))
        residual = residual + torch.where(C[...,70:71] >= 1., self.force_retry3_projection(force_h), self.force_projection(force_h))
        return original + torch.cat((residual, torch.zeros_like(residual)), -1)

    @staticmethod
    def balanced_history_masks(H):
        assert H.shape[-1]==48 and H.dtype==torch.float32
        keep=(torch.arange(48,device=H.device)[None,:]%4-torch.arange(4,device=H.device)[:,None])%4 < 2
        return keep.to(H.dtype)*2.

    def combine_expected(self,C,H,E=None,L=None,R=None,S=None,Q=None,J=None):
        """Fixed balanced four-mask 50-percent expectation; no random draws or mutable state."""
        base=self.combine(C,H,E,L,R,S,Q,J)
        heads=(self.fusion,self.projection,self.retry3_projection,self.force_projection,self.force_retry3_projection)
        if all(not head.weight.detach().any() for head in heads):
            return base  # Exact shortcut: all four heads are identical when history weights are zero.
        masks=self.balanced_history_masks(H)
        residuals=[self.combine(C,H*mask,E,L,R,S,Q,J)-base for mask in masks]
        return base+torch.stack(residuals,0).mean(0)

    @staticmethod
    def check_force(X, FX):
        assert X.shape[-1] == 42 and FX.shape == (*X.shape[:-1],3)
        assert FX.dtype == X.dtype and FX.device == X.device
        assert torch.isfinite(FX).all() and (FX.abs() <= 3.000001).all()

    def training_sequence(self,X,FX):
        assert X.ndim==3
        self.check_force(X,FX)
        self.memory.train()
        self.memory.flatten_parameters()
        h,_=self.memory(X,X.new_zeros(1,X.shape[1],32))
        self.force_memory.train()
        self.force_memory.flatten_parameters()
        fh,_=self.force_memory(encoded_force_sequence(FX),FX.new_zeros(1,FX.shape[1],16))
        assert h.requires_grad and fh.requires_grad
        return torch.cat((h,fh),-1)

    def sequence(self, X, FX):
        assert X.ndim == 3
        self.check_force(X,FX)
        h, _ = self.memory(X, X.new_zeros(1,X.shape[1],32))
        fh, _ = self.force_memory(encoded_force_sequence(FX), FX.new_zeros(1,FX.shape[1],16))
        return torch.cat((h,fh),-1)

    def step(self, X, hidden, alive, FX, FD):
        self.check_force(X,FX)
        assert hidden.shape == (1,X.shape[0],48) and alive.shape == X.shape[:-1]
        _, h = self.memory(X[None], hidden[...,:32].contiguous())
        assert FD.shape==FX.shape and FD.dtype==FX.dtype and FD.device==FX.device and torch.isfinite(FD).all() and (FD.abs()<=2.).all()
        assert not FD[(FX.abs()>2.).any(-1)].any()
        _, fh = self.force_memory(FD[None], hidden[...,32:].contiguous())
        h = torch.where(alive[None,:,None],torch.cat((h,fh),-1),0.)
        return h[0], h

def upgrade_history(model):
    assert not isinstance(model.critic, HistoryValue)
    model.critic = HistoryValue(model.critic)
