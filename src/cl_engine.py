"""Dataset-agnostic class-incremental engine used for BOTH NSL-KDD and CICIoT2023 (NumPy only).
Models: nn (one hidden layer, 64 ReLU, Adam 1e-3) and lr (softmax regression, SGD 0.05, L2 1e-4).
Strategies: finetune, EWC, replay (m=50/200/1000), joint (full retraining), joint_bal (class-balanced retraining).
Classes (CICIoT2023 categories): 0 Benign, 1 DDoS, 2 DoS, 3 Mirai, 4 Recon, 5 Spoofing, 6 Web-based, 7 BruteForce.
Stages (new classes only arrive; large classes first, rare last): S1 {Benign,DDoS} S2 {DoS} S3 {Mirai} S4 {Recon} S5 {Spoofing} S6 {Web-based,BruteForce}."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '1'
import sys, json, glob, time, multiprocessing as mp
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as prfs
import warnings
warnings.filterwarnings('ignore'); np.seterr(all='ignore')

NAMES = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-based', 'BruteForce']
CIDX = {n: i for i, n in enumerate(NAMES)}
STAGES = [[0, 1], [2], [3], [4], [5], [6, 7]]
K = len(NAMES); EPOCHS, BATCH = 5, 128
G = {}   # arrays shared with workers (inherited through fork)

# ------------------------------------------------------------------ data
def load(npz):
    d = np.load(npz, allow_pickle=True)
    G.update(Xtr=d['Xtr'], ytr=d['ytr'], Xte=d['Xte'], yte=d['yte'])
    return G

# ------------------------------------------------------------------ models (identical to exp3.py)
class Net:
    def __init__(s, kind, rng, din):
        s.kind = kind
        if kind == 'nn':
            s.p = [rng.randn(din, 64).astype(np.float32) * np.sqrt(2.0 / din), np.zeros(64, np.float32),
                   rng.randn(64, K).astype(np.float32) * np.sqrt(1.0 / 64), np.zeros(K, np.float32)]
            s.m = [np.zeros_like(a) for a in s.p]; s.v = [np.zeros_like(a) for a in s.p]; s.t = 0
        else:
            s.p = [np.zeros((din, K), np.float32), np.zeros(K, np.float32)]
    def fwd(s, X):
        if s.kind == 'nn': h = np.maximum(X @ s.p[0] + s.p[1], 0); z = h @ s.p[2] + s.p[3]
        else: h = None; z = X @ s.p[0] + s.p[1]
        z = z - z.max(1, keepdims=True); e = np.exp(z); return h, e / e.sum(1, keepdims=True)
    def grads(s, X, y, w=None):
        h, P = s.fwd(X); n = len(X); dz = P.copy(); dz[np.arange(n), y] -= 1
        if w is not None: dz *= w[:, None]
        dz /= n
        if s.kind == 'nn':
            dh = (dz @ s.p[2].T) * (h > 0); return [X.T @ dh, dh.sum(0), h.T @ dz, dz.sum(0)]
        return [X.T @ dz + 1e-4 * s.p[0], dz.sum(0)]
    def step(s, g):
        if s.kind == 'nn':
            s.t += 1
            for i in range(len(s.p)):
                s.m[i] = 0.9 * s.m[i] + 0.1 * g[i]; s.v[i] = 0.999 * s.v[i] + 0.001 * g[i] ** 2
                mh = s.m[i] / (1 - 0.9 ** s.t); vh = s.v[i] / (1 - 0.999 ** s.t)
                s.p[i] -= 1e-3 * mh / (np.sqrt(vh) + 1e-8)
        else:
            for i in range(len(s.p)): s.p[i] -= 0.05 * g[i]
    def fisher(s, X, y):
        h, P = s.fwd(X); dz = P.copy(); dz[np.arange(len(X)), y] -= 1
        if s.kind == 'nn':
            dh = (dz @ s.p[2].T) * (h > 0)
            F = [(X ** 2).T @ (dh ** 2), (dh ** 2).sum(0), (h ** 2).T @ (dz ** 2), (dz ** 2).sum(0)]
        else:
            F = [(X ** 2).T @ (dz ** 2), (dz ** 2).sum(0)]
        return [f / len(X) for f in F]
    def predict(s, X): return s.fwd(X)[1].argmax(1)

def train(net, idx, rng, w=None, ewc=None):
    Xtr, ytr = G['Xtr'], G['ytr']
    for _ in range(EPOCHS):
        p = idx[rng.permutation(len(idx))]
        for b in range(0, len(p), BATCH):
            bi = p[b:b + BATCH]
            g = net.grads(Xtr[bi], ytr[bi], None if w is None else w[ytr[bi]])
            if ewc is not None and not ewc[3]:
                lam, Fs, anchor, _ = ewc
                g = [gi + lam * Fi * (pi - ai) for gi, Fi, pi, ai in zip(g, Fs, net.p, anchor)]
            net.step(g)
            if ewc is not None and ewc[3]:                  # stable proximal EWC step
                lam, Fs, anchor, _ = ewc; eta = 0.05 if net.kind == 'lr' else 1e-3
                for i in range(len(net.p)):
                    c = eta * lam * Fs[i]; net.p[i] = ((net.p[i] + c * anchor[i]) / (1.0 + c)).astype(np.float32)

def evalfull(net, seen):
    Xte, yte = G['Xte'], G['yte']; m = np.isin(yte, seen); pred = net.predict(Xte[m])
    p, r, f, _ = prfs(yte[m], pred, labels=seen, average=None, zero_division=0)
    return {int(c): dict(p=float(a), r=float(b), f1=float(g)) for c, a, b, g in zip(seen, p, r, f)}

def run(kind, method, seed, m=200, lam=0.0, prox=0):
    Xtr, ytr = G['Xtr'], G['ytr']; din = Xtr.shape[1]
    rng = np.random.RandomState(seed); net = Net(kind, rng, din)
    buf = np.array([], int); seen = []; full = []; samples = 0; Fs = None; anchor = None
    for t, cl in enumerate(STAGES):
        new = np.where(np.isin(ytr, cl))[0]; seen = seen + cl; w = None; ewc = None
        if method in ('joint', 'joint_bal'):
            idx = np.where(np.isin(ytr, seen))[0]; net = Net(kind, rng, din)
            if method == 'joint_bal':
                cnt = np.bincount(ytr[idx], minlength=K).astype(float); w = np.where(cnt > 0, cnt.sum() / (len(seen) * np.maximum(cnt, 1)), 0)
        elif method == 'replay': idx = np.concatenate([new, buf])
        else: idx = new
        if method == 'ewc' and Fs is not None: ewc = (lam, Fs, anchor, prox)
        train(net, idx, rng, w, ewc); samples += len(idx) * EPOCHS
        if method == 'replay':
            buf = np.concatenate([buf] + [rng.choice(np.where(ytr == c)[0], min(m, (ytr == c).sum()), replace=False) for c in cl])
        if method == 'ewc':
            sub = rng.choice(new, min(2000, len(new)), replace=False)
            Fn = net.fisher(Xtr[sub], ytr[sub]); Fs = Fn if Fs is None else [a + b for a, b in zip(Fs, Fn)]
            anchor = [a.copy() for a in net.p]
        full.append(evalfull(net, seen))
    npar = int(sum(a.size for a in net.p))
    return dict(full=full, samples=samples, mem=dict(params=npar, model_bytes=npar * 4, buffer_records=int(len(buf)),
                buffer_bytes=int(len(buf)) * din * 4, ewc_extra_bytes=(2 * npar * 4 if method == 'ewc' else 0)))

CONFIGS = [('finetune', {}), ('replay', dict(m=50)), ('replay', dict(m=200)), ('replay', dict(m=1000)), ('joint', {}), ('joint_bal', {})] + \
          [('ewc', dict(lam=l)) for l in (10., 100., 1e3, 1e4, 1e5)] + [('ewc', dict(lam=l, prox=1)) for l in (10., 100., 1e3, 1e4, 1e5)]
def key(meth, kw): return meth + ''.join(f'_{k}{v:g}' for k, v in kw.items())

def task(args):
    kind, seed = args; t0 = time.time()
    out = {key(m_, kw): run(kind, m_, seed, **kw) for m_, kw in CONFIGS}
    print(f'done {kind} seed {seed} in {time.time()-t0:.0f}s', flush=True)
    return kind, seed, out

def run_all(npz, seeds, procs=4, out_json='cl_results.json'):
    load(npz)
    jobs = [(k, s) for s in range(seeds) for k in ('nn', 'lr')]
    with mp.get_context('fork').Pool(procs) as pool: res = pool.map(task, jobs, chunksize=1)
    out = {'nn': {}, 'lr': {}}
    for kind, seed, o in sorted(res, key=lambda r: (r[0], r[1])):
        for k, v in o.items(): out[kind].setdefault(k, []).append(v)
    json.dump(out, open(out_json, 'w')); return out

# ------------------------------------------------------------------ analysis
def macro(r, t=-1): return float(np.mean([v['f1'] for v in r['full'][t].values()]))
def FB(r):
    T = len(STAGES); R = np.full((T, T), np.nan)
    for t in range(T):
        for i in range(t + 1): R[t, i] = np.mean([r['full'][t][c]['f1'] for c in STAGES[i]])
    return (np.mean([np.nanmax(R[i:T-1, i]) - R[T-1, i] for i in range(T-1)]), np.mean([R[T-1, i] - R[i, i] for i in range(T-1)]))

def analyze(out):
    from scipy import stats
    for kind in ('nn', 'lr'):
        o = {k: [{**r, 'full': [{int(c): v for c, v in st.items()} for st in r['full']]} for r in v] for k, v in out[kind].items()}
        form = (lambda k: k.startswith('ewc') and 'prox1' not in k) if kind == 'nn' else (lambda k: k.startswith('ewc') and 'prox1' in k)
        ewc_keys = [k for k in o if form(k)]
        fin = lambda k: np.array([macro(r) for r in o[k]])
        best = max(ewc_keys, key=lambda k: fin(k).mean())
        print(f'\n===== {kind.upper()}   (EWC form: {"explicit" if kind=="nn" else "proximal"}; strength tuned on test, favours EWC)')
        print('EWC grid:', {k: round(float(fin(k).mean()), 3) for k in ewc_keys})
        rows = [('Fine-tune', 'finetune'), ('EWC (best λ: ' + best + ')', best), ('Replay m=50', 'replay_m50'), ('Replay m=200', 'replay_m200'),
                ('Replay m=1000', 'replay_m1000'), ('Full retraining', 'joint'), ('Retraining, class-balanced', 'joint_bal')]
        for name, k in rows:
            f = fin(k); fb = np.array([FB(r) for r in o[k]]); mem = o[k][0]['mem']; sm = np.mean([r['samples'] for r in o[k]]) / 1e6
            print(f'{name:40s} F1 {f.mean():.3f}±{f.std():.3f}  F {fb[:,0].mean():.3f}  BWT {fb[:,1].mean():.3f}  samples {sm:.2f}M  buf {mem["buffer_records"]} rec ({mem["buffer_bytes"]/1e6:.2f} MB) params {mem["params"]} ewc_extra {mem["ewc_extra_bytes"]/1e3:.0f} KB')
        print('-- paired tests, same seeds: mean diff [95% CI], t p, Wilcoxon p')
        for a, b in [('replay_m200', 'finetune'), ('replay_m200', best), ('replay_m200', 'joint'), ('replay_m1000', 'joint'), ('replay_m200', 'replay_m50'), ('replay_m200', 'joint_bal')]:
            d = fin(a) - fin(b); n = len(d)
            if n < 2: continue
            h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
            try: wp = stats.wilcoxon(fin(a), fin(b)).pvalue
            except ValueError: wp = float('nan')
            print(f'  {a} − {b}: {d.mean():+.3f} [{d.mean()-h:+.3f},{d.mean()+h:+.3f}]  t p={stats.ttest_rel(fin(a), fin(b)).pvalue:.4f}  W p={wp:.4f}')
        print('-- per-class final P/R/F1 (mean over seeds)')
        for name, k in [('Fine-tune', 'finetune'), ('EWC', best), ('Replay m=200', 'replay_m200'), ('Full retraining', 'joint'), ('Retraining, class-balanced', 'joint_bal')]:
            s = f'  {name:26s}'
            for c in range(K):
                g = lambda q: np.mean([r['full'][-1][c][q] for r in o[k]])
                s += f' | {NAMES[c]} {g("p"):.2f}/{g("r"):.2f}/{g("f1"):.2f}'
            print(s)
