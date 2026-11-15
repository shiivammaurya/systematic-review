"""Budget-aware test (Eq. 3) on NSL-KDD. Actions at stages 2-4: replay (m=200) or retrain from scratch.
Budget 1.5M samples; rule: after a tentative replay update, if the validation macro-F1 on OLD classes drops by more than 0.10
and the remaining budget covers a retraining, discard the update and retrain. Validation = 10% of each class's training records
(never trained on, never the test file)."""
import json, multiprocessing as mp
import numpy as np
from sklearn.metrics import f1_score
import cl_engine as cg
M = 200
CFGB = dict(dataset='nsl', budget=1.5e6, taus=(0.10,))     # set by run_budget
def policies():
    T = len(cg.STAGES)
    return ['replay'] + [f'retrain_at_{k}' for k in range(2, T + 1)] + ['retrain_always'] + [f'budget_t{t:g}' for t in CFGB['taus']]

def _run(kind, policy, seed, val, trmask):
    Xtr, ytr = cg.G['Xtr'], cg.G['ytr']; din = Xtr.shape[1]
    def vf1(net, seen):
        m = np.isin(ytr[val], seen)
        return float(np.mean(f1_score(ytr[val][m], net.predict(Xtr[val][m]), labels=seen, average=None, zero_division=0)))
    rng = np.random.RandomState(seed); net = cg.Net(kind, rng, din)
    buf = np.array([], int); seen = []; samples = 0; log = []
    for t, cl in enumerate(cg.STAGES):
        new = np.where(np.isin(ytr, cl) & trmask)[0]; old = list(seen); seen = seen + cl
        all_idx = np.where(np.isin(ytr, seen) & trmask)[0]
        if t == 0:
            cg.train(net, new, rng); samples += len(new) * cg.EPOCHS; act = 'train'
        else:
            cost = len(all_idx) * cg.EPOCHS
            if policy.startswith('budget'):
                before = vf1(net, old); idx = np.concatenate([new, buf]); cg.train(net, idx, rng); samples += len(idx) * cg.EPOCHS
                drop = before - vf1(net, old)
                if drop > float(policy.split('_t')[1]) and samples + cost <= CFGB['budget']:
                    net = cg.Net(kind, rng, din); cg.train(net, all_idx, rng); samples += cost; act = 'retrain'
                else: act = 'replay'
            else:
                do = (policy == 'retrain_always') or (policy.startswith('retrain_at_') and t + 1 == int(policy.split('_')[-1]))
                if do: net = cg.Net(kind, rng, din); cg.train(net, all_idx, rng); samples += cost; act = 'retrain'
                else:
                    idx = np.concatenate([new, buf]); cg.train(net, idx, rng); samples += len(idx) * cg.EPOCHS; act = 'replay'
        buf = np.concatenate([buf] + [rng.choice(np.where((ytr == c) & trmask)[0], min(M, ((ytr == c) & trmask).sum()), replace=False) for c in cl])
        log.append(act)
    pc = cg.evalfull(net, seen)
    return dict(final=float(np.mean([v['f1'] for v in pc.values()])), samples=samples, actions=log)

def _task(a):
    import runner
    kind, seed = a
    if CFGB['dataset'] == 'ciciot': runner.split_ciciot(seed)
    ytr = cg.G['ytr']; vrng = np.random.RandomState(123)
    val = np.concatenate([vrng.choice(np.where(ytr == c)[0], max(1, int(0.1 * (ytr == c).sum())), replace=False) for c in range(cg.K)])
    trmask = np.ones(len(ytr), bool); trmask[val] = False
    return kind, seed, {p: _run(kind, p, seed, val, trmask) for p in policies()}

def run_budget(dataset='nsl', seeds=10, procs=2, budget=1.5e6, taus=(0.10,), out_json='results_budget.json'):
    CFGB.update(dataset=dataset, budget=budget, taus=tuple(taus))
    jobs = [(k, s) for s in range(seeds) for k in ('nn', 'lr')]
    with mp.get_context('fork').Pool(procs) as pool: res = pool.map(_task, jobs, chunksize=1)
    out = {'nn': {p: [] for p in policies()}, 'lr': {p: [] for p in policies()}}
    for kind, seed, o in sorted(res, key=lambda r: (r[0], r[1])):
        for p, v in o.items(): out[kind][p].append(v)
    json.dump(out, open(out_json, 'w')); return out

def report(out):
    from scipy import stats
    for kind in ('nn', 'lr'):
        print(f'\n== {kind.upper()} budget test (macro-F1 after the last stage, mean±sd over seeds)')
        base = np.array([r['final'] for r in out[kind]['replay']])
        for p in out[kind]:
            f = np.array([r['final'] for r in out[kind][p]]); sm = np.mean([r['samples'] for r in out[kind][p]]) / 1e6
            ex = ''
            if p != 'replay' and len(f) > 1 and np.any(f != base): ex = f'  vs replay {np.mean(f-base):+.3f} (t p={stats.ttest_rel(f, base).pvalue:.3f}, W p={stats.wilcoxon(f, base).pvalue:.4f})'
            print(f'  {p:15s} {f.mean():.3f}±{f.std(ddof=1) if len(f)>1 else 0:.3f}  {sm:.2f}M samples{ex}')
        for p in [q for q in out[kind] if q.startswith('budget')]:
            fired = sum(a == 'retrain' for r in out[kind][p] for a in r['actions']); n = len(out[kind][p])
            print(f'  {p}: rule fired {fired} times in {n} runs ({sum(any(a=="retrain" for a in r["actions"]) for r in out[kind][p])} runs at least once)')
