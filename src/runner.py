"""Loaders and driver for the two datasets. Both use cl_engine (same models, strategies, metrics).
  NSL-KDD   : official train/test files, 5 categories, 4 stages, fixed split (seeds change init, batch order, exemplars).
  CICIoT2023: one CSV per class (first 5,000 rows of each class file), 34 classes, 5 stages, cleaned, per-seed stratified 70/30 split."""
import os, sys, json, time, multiprocessing as mp
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '1'
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import cl_engine as cg

# ---------------------------------------------------------------- NSL-KDD
def setup_nsl(d):
    cols = [f'f{i}' for i in range(41)] + ['label', 'diff']
    tr = pd.read_csv(os.path.join(d, 'KDDTrain+.txt'), header=None, names=cols)
    te = pd.read_csv(os.path.join(d, 'KDDTest+.txt'), header=None, names=cols)
    dos = 'back land neptune pod smurf teardrop apache2 mailbomb processtable udpstorm worm'.split()
    probe = 'ipsweep nmap portsweep satan mscan saint'.split()
    r2l = 'ftp_write guess_passwd imap multihop phf spy warezclient warezmaster sendmail named snmpgetattack snmpguess xlock xsnoop httptunnel'.split()
    u2r = 'buffer_overflow loadmodule perl rootkit ps sqlattack xterm'.split()
    cat = {'normal': 0}
    for names, c in ((dos, 1), (probe, 2), (r2l, 3), (u2r, 4)): cat.update({a: c for a in names})
    for x in (tr, te): x['y'] = x['label'].map(cat)
    tr, te = tr.dropna(subset=['y']), te.dropna(subset=['y'])
    Xall = pd.get_dummies(pd.concat([tr, te])[[f'f{i}' for i in range(41)]], columns=['f1', 'f2', 'f3']).astype(float)
    Xtr, Xte = Xall.iloc[:len(tr)].values, Xall.iloc[len(tr):].values
    sc = StandardScaler().fit(Xtr)
    cg.G.update(Xtr=sc.transform(Xtr).astype(np.float32), ytr=tr['y'].values.astype(int),
                Xte=sc.transform(Xte).astype(np.float32), yte=te['y'].values.astype(int))
    cg.NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']; cg.K = 5; cg.STAGES = [[0, 1], [2], [3], [4]]
    print('NSL-KDD train', np.bincount(cg.G['ytr']).tolist(), 'test', np.bincount(cg.G['yte']).tolist(), 'features', Xtr.shape[1])

# ---------------------------------------------------------------- CICIoT2023
STAGE_NAMES = [
 ['Benign_Final','DDoS-ICMP_Flood','DDoS-UDP_Flood','DDoS-SYN_Flood','DDoS-TCP_Flood'],
 ['DDoS-ACK_Fragmentation','DDoS-HTTP_Flood','DDoS-ICMP_Fragmentation','DDoS-PSHACK_FLOOD','DDoS-RSTFINFLOOD','DDoS-SlowLoris','DDoS-SynonymousIP_Flood','DDoS-UDP_Fragmentation'],
 ['DoS-HTTP_Flood','DoS-SYN_Flood','DoS-TCP_Flood','DoS-UDP_Flood','Mirai-greeth_flood','Mirai-greip_flood','Mirai-udpplain'],
 ['Recon-HostDiscovery','Recon-OSScan','Recon-PingSweep','Recon-PortScan','VulnerabilityScan','MITM-ArpSpoofing','DNS_Spoofing'],
 ['BrowserHijacking','CommandInjection','DictionaryBruteForce','SqlInjection','Uploading_Attack','XSS','Backdoor_Malware']]
DX = DY = None

def setup_ciciot(d):
    global DX, DY
    names = [n for s in STAGE_NAMES for n in s]
    cg.NAMES = names; cg.K = len(names); cg.STAGES = [[names.index(n) for n in s] for s in STAGE_NAMES]
    X, y = [], []
    for c, n in enumerate(names):
        df = pd.read_csv(os.path.join(d, n + '.csv')).replace([np.inf, -np.inf], np.nan).dropna()
        X.append(df.values.astype(np.float32)); y.append(np.full(len(df), c))
    X = np.concatenate(X); y = np.concatenate(y); n0 = len(X)
    h = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values
    nlab = pd.Series(y).groupby(h).transform('nunique').values
    conflict = nlab > 1; dup = pd.Series(h).duplicated().values & ~conflict; keep = ~conflict & ~dup
    print(f'CICIoT2023 rows {n0} | ambiguous removed {int(conflict.sum())} | exact duplicates removed {int(dup.sum())} | kept {int(keep.sum())}')
    DX, DY = X[keep], y[keep]; print('class counts', np.bincount(DY).tolist())

def split_ciciot(seed, test=0.3):
    rng = np.random.RandomState(1000 + seed); tr, te = [], []
    for c in range(cg.K):
        i = np.where(DY == c)[0]; p = rng.permutation(len(i)); nte = int(round(test * len(i)))
        te.append(i[p[:nte]]); tr.append(i[p[nte:]])
    tr, te = np.concatenate(tr), np.concatenate(te)
    mu = DX[tr].mean(0); sd = DX[tr].std(0) + 1e-6
    f = lambda A: np.clip((A - mu) / sd, -10, 10).astype(np.float32)
    cg.G.update(Xtr=f(DX[tr]), ytr=DY[tr], Xte=f(DX[te]), yte=DY[te])

# ---------------------------------------------------------------- driver
CFG = {'nn': [c for c in cg.CONFIGS if not (c[0] == 'ewc' and c[1].get('prox'))],       # explicit EWC for the network
       'lr': [c for c in cg.CONFIGS if not (c[0] == 'ewc' and not c[1].get('prox'))]}   # proximal EWC for logistic regression
DATASET = None
def _task(a):
    kind, seed = a; t0 = time.time()
    if DATASET == 'ciciot': split_ciciot(seed)
    out = {cg.key(m, kw): cg.run(kind, m, seed, **kw) for m, kw in CFG[kind]}
    print(f'  done {DATASET} {kind} seed {seed} in {time.time()-t0:.0f}s', flush=True); return kind, seed, out

def run_dataset(name, data_dir, seeds=10, procs=2, out_json=None):
    global DATASET
    DATASET = name
    (setup_nsl if name == 'nsl' else setup_ciciot)(data_dir)
    jobs = [(k, s) for s in range(seeds) for k in ('nn', 'lr')]
    with mp.get_context('fork').Pool(procs) as pool: res = pool.map(_task, jobs, chunksize=1)
    out = {'nn': {}, 'lr': {}}
    for kind, seed, o in sorted(res, key=lambda r: (r[0], r[1])):
        for k, v in o.items(): out[kind].setdefault(k, []).append(v)
    out_json = out_json or f'results_{name}_{seeds}seeds.json'
    json.dump(out, open(out_json, 'w')); print('saved', out_json); return out

def set_globals(name):
    """Point the engine at one dataset's class names/stages (needed before analysis when both datasets were run)."""
    if name == 'nsl':
        cg.NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']; cg.K = 5; cg.STAGES = [[0, 1], [2], [3], [4]]
    else:
        names = [n for s in STAGE_NAMES for n in s]
        cg.NAMES = names; cg.K = len(names); cg.STAGES = [[names.index(n) for n in s] for s in STAGE_NAMES]
