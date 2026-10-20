import hashlib, io, json, tarfile, zipfile
from pathlib import Path
import numpy as np
import torch

def read_npz(path):
    path = Path(path)
    if path.exists():
        return np.load(path)
    ref = json.loads(path.with_suffix('.npz.archive_ref.json').read_text())
    A = Path(ref['archive'])
    assert hashlib.sha256(A.read_bytes()).hexdigest() == ref['archive_sha256']
    with tarfile.open(A) as z:
        data = z.extractfile(ref['member']).read()
    assert hashlib.sha256(data).hexdigest() == ref['sha256']
    return np.load(io.BytesIO(data))

def save_history(path, X, alive, done):
    x = np.ascontiguousarray(X.cpu().numpy(), dtype=np.float32)
    a = alive.cpu().numpy()
    d = done.cpu().numpy()
    x[~a] = 0
    u = x.view(np.uint32)
    delta = u.copy()
    delta[1:] = u[1:] ^ u[:-1]
    packed = np.ascontiguousarray(delta).view(np.uint8).reshape(-1, 4).T.copy()
    fields = dict(packed=packed, shape=np.array(x.shape), alive=a, done=d, content_sha256=np.array(hashlib.sha256(x.tobytes()).hexdigest()))
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_LZMA) as z:
        for name, value in fields.items():
            b = io.BytesIO()
            np.save(b, value, allow_pickle=False)
            z.writestr(name + '.npy', b.getvalue())
    decoded, aa, dd = load_history(path)
    assert np.array_equal(decoded, x) and np.array_equal(aa, a) and np.array_equal(dd, d)

def load_history(path):
    z = read_npz(path)
    shape = tuple(z['shape'])
    raw = z['packed'].T.copy().reshape(-1)
    delta = raw.view(np.uint32).reshape(shape)
    x = np.bitwise_xor.accumulate(delta, axis=0).view(np.float32)
    assert hashlib.sha256(x.tobytes()).hexdigest() == str(z['content_sha256'])
    alive = z['alive']
    done = z['done']
    assert not x[~alive].any()
    assert alive[0].all() and (not (alive[1:] & ~alive[:-1]).any())
    assert np.all((alive & done).sum(0) == 1)
    return (x, alive, done)

def dataset(folders, device):
    """Read each cohort once; only selected full episode histories become dense tensors."""
    folders = [Path(f) for f in folders]
    chunks = []
    retained = []
    maxT = 0
    for i, F in enumerate(folders):
        with read_npz(F / 'selected_data.npz') as z:
            q = {k: z[k] for k in z.files}
        assert np.all(q['env'] // 128 == i)
        chosen = np.unique(q['env'] % 128)
        assert len(chosen) > 0
        chunks.append(q)
        retained.append(chosen)
        with read_npz(F / 'complete_history.npz') as z:
            maxT = max(maxT, int(z['shape'][0]))
    original_N = 128 * len(chunks)
    N = sum((len(v) for v in retained))
    widths = dict(X=42, FX=3, C=75, E=6, LAT=1, REL=1, CTRL=4, Q=12, JT=6)
    out = {name: torch.zeros(maxT, N, width, device=device) for name, width in widths.items()}
    out.update(selected=torch.zeros(maxT, original_N, dtype=torch.bool, device=device), alive=torch.zeros(maxT, original_N, dtype=torch.bool, device=device), target=torch.zeros(maxT, original_N, device=device), phase=torch.zeros(maxT, original_N, device=device))
    lookup = torch.full((original_N,), -1, device=device, dtype=torch.long)
    original = torch.tensor(np.concatenate([e + 128 * i for i, e in enumerate(retained)]), device=device, dtype=torch.long)
    lookup[original] = torch.arange(N, device=device)
    offset = 0
    fieldmap = dict(C='critic_features', E='critic_privileged6', LAT='critic_latency1', REL='critic_release1', CTRL='critic_control4', Q='critic_queue12', JT='critic_joint_target6')
    for i, (F, q, chosen) in enumerate(zip(folders, chunks, retained)):
        X, a, d = load_history(F / 'complete_history.npz')
        FX, fa, fd = load_history(F / 'complete_force_history.npz')
        assert X.shape[1:] == (128, 42) and FX.shape == (*X.shape[:-1], 3) and np.array_equal(a, fa) and np.array_equal(d, fd)
        assert np.array_equal(FX[q['tick'], q['env'] % 128], q['privileged_truth6'][:, 3:])
        count = len(chosen)
        out['X'][:len(X), offset:offset + count] = torch.from_numpy(X[:, chosen].copy()).to(device)
        out['FX'][:len(FX), offset:offset + count] = torch.from_numpy(FX[:, chosen].copy()).to(device)
        out['alive'][:len(a), i * 128:(i + 1) * 128] = torch.from_numpy(a).to(device)
        tick = torch.from_numpy(q['tick']).to(device)
        env = torch.from_numpy(q['env']).to(device)
        packed = lookup[env]
        assert (packed >= 0).all() and out['alive'][tick, env].all()
        out['selected'][tick, env] = True
        for name, key in fieldmap.items():
            out[name][tick, packed] = torch.from_numpy(q[key]).to(device)
        for name in ['target', 'phase']:
            out[name][tick, env] = torch.from_numpy(q[name]).to(device)
        offset += count
    flat = {key: np.concatenate([q[key] for q in chunks]) for key in chunks[0]}
    out.update(chunks=chunks, flat=flat, original_episode_count=original_N, original_to_compact=lookup, compact_to_original=original)
    return out

def predict_dataset(model, data):
    H = model.critic.sequence(data['X'], data['FX'])
    values = []
    for q in data['chunks']:
        C = torch.from_numpy(q['critic_features']).to(H.device)
        tick = torch.from_numpy(q['tick']).to(H.device)
        original = torch.from_numpy(q['env']).to(H.device)
        local = data['original_to_compact'][original]
        assert (local >= 0).all()
        values.append(model.critic.combine_expected(C, H[tick, local], *[torch.from_numpy(q[k]).to(H.device) for k in ['critic_privileged6', 'critic_latency1', 'critic_release1', 'critic_control4', 'critic_queue12', 'critic_joint_target6']])[:, 0] * 100.0)
    return torch.cat(values)

def save_lzma_npz(path, **arrays):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_LZMA, allowZip64=True) as z:
        for name, value in arrays.items():
            assert isinstance(name, str) and '/' not in name and ('\\' not in name)
            value = np.asarray(value)
            assert not value.dtype.hasobject
            b = io.BytesIO()
            np.save(b, value, allow_pickle=False)
            z.writestr(name + '.npy', b.getvalue())

def gather_columns(data, name, original_ids):
    """Keep the original shuffled16-episode batches, including zero-loss episodes."""
    assert name in PACKED_FIELDS
    local = data['original_to_compact'][original_ids]
    values = data[name][:, local.clamp_min(0)]
    return torch.where((local >= 0)[None, :, None], values, torch.zeros_like(values))
PACKED_FIELDS = ('X', 'FX', 'C', 'E', 'LAT', 'REL', 'CTRL', 'Q', 'JT')
