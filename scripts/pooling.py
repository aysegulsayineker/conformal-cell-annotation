"""alpha rejimi tablosu: veri-ici, capraz ve havuzlanmis kalibrasyon.
Esitlikler ontolojiye gore bozulur (recompute.py ile ayni yordam)."""
import json, numpy as np, pandas as pd
exec(open("recompute.py").read().split('if __name__')[0], globals())

MODEL = {"pbmc":"votes/oylar_v2_pbmc_qwen25_7b.json", "panc":"votes/oylar_v2_panc_qwen25_7b.json"}
KL = {"pbmc": pd.read_csv("handoff/clusters_pbmc_v2.csv"),
      "panc": pd.read_csv("handoff/clusters_pancreas_v2.csv")}
for k in KL: KL[k]["cluster"] = KL[k].cluster.astype(str)
TRUTH = {k: dict(zip(d.cluster, d.true_type)) for k, d in KL.items()}

ST = {}
for ds in ("pbmc","panc"):
    labs, lcl, D, lcad = load_space(ds)
    P, order, V = probs(MODEL[ds], labs)
    Y = np.array([labs.index(TRUTH[ds][c]) for c in order])
    ST[ds] = dict(P=P, Y=Y, D=D, lcad=lcad, lcl=lcl, labs=labs, KO=onto_key(P, lcl, lcad))

def scores(ds, rng):
    s = ST[ds]
    key = s["KO"] + rng.random(s["P"].shape)*1e-6
    S = aps_keyed(s["P"], rng.random(len(s["Y"])), key)
    return S, S[np.arange(len(s["Y"])), s["Y"]]

def run(alpha, regime, test_ds, n_rep=2000, seed=11):
    rng = np.random.default_rng(seed); other = "panc" if test_ds=="pbmc" else "pbmc"
    s, so = ST[test_ds], ST[other]
    rows = []
    for _ in range(n_rep):
        S, st = scores(test_ds, rng)
        # Veri-ici rejimde oteki veri setinden hic cekilis yapilmaz; boylece rastgele akis
        # recompute.evaluate ile birebir ayni kalir ve ana izgarayla tam tutarlilik saglanir.
        if regime != "within":
            So, sto = scores(other, rng)
        n = len(s["Y"])
        if regime == "within":
            perm = rng.permutation(n); cal_s, tst = st[perm[:n//2]], perm[n//2:]
        elif regime == "cross":
            tst = rng.permutation(n)[: n - n//2]; cal_s = sto[rng.permutation(len(sto))[: len(sto)//2]]
        else:                                     # pooled
            perm = rng.permutation(n); tst = perm[n//2:]
            cal_s = np.concatenate([st[perm[:n//2]], sto[rng.permutation(len(sto))[: len(sto)//2]]])
        k = int(np.ceil((len(cal_s)+1)*(1-alpha)))
        q = np.inf if k > len(cal_s) else np.sort(cal_s)[k-1]
        M = S[tst] <= q
        e = ~M.any(1)
        if e.any(): M[e, np.argmin(S[tst][e],1)] = True
        hit = M[np.arange(len(tst)), s["Y"][tst]]
        res = np.mean([s["lcad"]([s["lcl"][j] for j in np.flatnonzero(r)])/s["D"][t]
                       for r, t in zip(M, s["Y"][tst])])
        rows.append((hit.mean(), M.sum(1).mean(), 100*res, len(cal_s)))
    v = np.array(rows).mean(0)
    return dict(kapsama=round(float(v[0]),3), kume_boyu=round(float(v[1]),2),
                cozunurluk=round(float(v[2]),1), n_cal=int(round(v[3])))

out = []
for a in (0.05, 0.10, 0.20):
    for reg, lab in (("cross","cross-dataset"), ("pooled","pooled"), ("within","within-dataset")):
        for ds, dlab in (("pbmc","PBMC"), ("panc","Pancreas")):
            r = run(a, reg, ds)
            out.append({"alpha":a, "kalibrasyon":lab, "test":dlab, **r})
            print(a, lab, dlab, r, flush=True)
pd.DataFrame(out).to_csv("out/havuzlama_yeniden.csv", index=False)
print("TAMAM")
