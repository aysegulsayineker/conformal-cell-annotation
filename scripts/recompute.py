"""Butun konformal tablolari TEK ve tutarli yordamla yeniden uretir.
Esitlikler ontolojiye gore bozulur; rastgele esitlik taban cizgisi de raporlanir."""
import json, glob, os, numpy as np, pandas as pd

exec(open("baseline.py").read(), globals())          # cl_aps_scores, smooth, build_onto, ...

ALPHAS = (0.05, 0.10, 0.20)
N_REP, SEED = 2000, 11

def load_space(ds):
    O = json.load(open(f"handoff/cl_{'pbmc' if ds=='pbmc' else 'panc'}.json"))
    labs = json.load(open(f"handoff/labels_{'pbmc' if ds=='pbmc' else 'panc'}.json"))
    lcl = [O["label_cl"][l] for l in labs]
    dmap, lcad, _, _ = build_onto(O["par"], lcl)
    return labs, lcl, np.array([dmap[c] for c in lcl]), lcad

def probs(vote_json, labs):
    V = json.load(open(vote_json)) if isinstance(vote_json, str) else vote_json
    order = [str(c["cluster"]) for c in V["clusters"]]
    C = np.zeros((len(order), len(labs)))
    for i, c in enumerate(V["clusters"]):
        for k, v in c["counts"].items():
            if k in labs: C[i, labs.index(k)] = v
    P = (C + 0.05) / (C.sum(1, keepdims=True) + 0.05*len(labs))
    return P, order, V

def aps_keyed(P, u, key):
    n, m = P.shape
    ordr = np.empty((n, m), dtype=int)
    for i in range(n):
        ordr[i] = np.lexsort((-key[i], -P[i]))
    Ps = np.take_along_axis(P, ordr, 1)
    s = (np.cumsum(Ps, 1) - Ps) + Ps * u[:, None]
    S = np.empty_like(s); np.put_along_axis(S, ordr, s, 1)
    return S

def onto_key(P, lcl, lcad):
    n, m = P.shape; K = np.zeros((n, m))
    for i in range(n):
        top = lcl[int(np.argmax(P[i]))]
        for j in range(m): K[i, j] = lcad([top, lcl[j]])
    return K

def evaluate(P, Y, D, lcad, lcl, alpha, mode, n_rep=N_REP, seed=SEED, true_type=None):
    n, m = P.shape; rng = np.random.default_rng(seed); rows = []; cls = {}
    KO = onto_key(P, lcl, lcad) if mode == "onto" else None
    for _ in range(n_rep):
        key = rng.random((n, m)) if mode == "random" else KO + rng.random((n, m))*1e-6
        S = aps_keyed(P, rng.random(n), key); st = S[np.arange(n), Y]
        perm = rng.permutation(n); cal, tst = perm[:n//2], perm[n//2:]
        k = int(np.ceil((len(cal)+1)*(1-alpha)))
        q = np.inf if k > len(cal) else np.sort(st[cal])[k-1]
        M = S[tst] <= q
        e = ~M.any(1)
        if e.any(): M[e, np.argmin(S[tst][e],1)] = True
        hit = M[np.arange(len(tst)), Y[tst]]
        res = np.mean([lcad([lcl[j] for j in np.flatnonzero(r)])/D[t] for r, t in zip(M, Y[tst])])
        rows.append((hit.mean(), M.sum(1).mean(), 100*res))
        if true_type is not None:
            for lab in set(true_type):
                sel = np.array([true_type[i] == lab for i in tst])
                if sel.any(): cls.setdefault(lab, []).extend(hit[sel].tolist())
    v = np.array(rows).mean(0)
    out = dict(kapsama=round(float(v[0]),3), kume_boyu=round(float(v[1]),2),
               cozunurluk=round(float(v[2]),1))
    cc = {l: round(float(np.mean(x)),3) for l, x in cls.items()} if cls else None
    return out, cc

def top1(V, truth, order):
    """Kati coklu kural: oy maksimumu birden fazla etikette ise kume yanlis sayilir."""
    h = 0
    for c in V["clusters"]:
        if not c["counts"]: continue
        mx = max(c["counts"].values())
        win = [l for l, v in c["counts"].items() if v == mx]
        h += (len(win) == 1 and win[0] == truth[str(c["cluster"])])
    return round(h/len(V["clusters"]), 3)

def collapse_rate(V):
    tot = len(V["clusters"])
    col = sum(1 for c in V["clusters"] if c["counts"] and max(c["counts"].values()) == sum(c["counts"].values()))
    return col, tot, round(col/tot, 3)

if __name__ == "__main__":
    KL = {"pbmc": pd.read_csv("handoff/clusters_pbmc_v2.csv"),
          "panc": pd.read_csv("handoff/clusters_pancreas_v2.csv")}
    for k in KL: KL[k]["cluster"] = KL[k].cluster.astype(str)
    TRUTH = {k: dict(zip(d.cluster, d.true_type)) for k, d in KL.items()}
    SPACE = {k: load_space(k) for k in ("pbmc","panc")}

    recs, coll = [], []
    for f in sorted(glob.glob("votes/*.json")):
        meta = json.load(open(f))
        ds = "pbmc" if meta["dataset"] in ("pbmc","PBMC3k") else "panc"
        labs, lcl, D, lcad = SPACE[ds]
        P, order, V = probs(f, labs)
        Y = np.array([labs.index(TRUTH[ds][c]) for c in order])
        tt = [TRUTH[ds][c] for c in order]
        t1 = top1(V, TRUTH[ds], order)
        c_, n_, cr = collapse_rate(V)
        coll.append({"veri":ds,"model":meta["model"],"cokme":f"{c_}/{n_}","oran":cr,"top1":t1})
        for a in ALPHAS:
            for mode in ("onto","random"):
                r, cc = evaluate(P, Y, D, lcad, lcl, a, mode, true_type=tt if (a==0.10 and mode=="onto") else None)
                recs.append({"veri":ds,"model":meta["model"],"alpha":a,"esitlik":mode,
                             "top1":t1, **r})
                if cc: json.dump(cc, open(f"out/sinif_{ds}_{meta['model']}_a{a}.json","w"))
        print("islendi", f, flush=True)
    os.makedirs("out", exist_ok=True)
    pd.DataFrame(recs).to_csv("out/izgara_yeniden.csv", index=False)
    pd.DataFrame(coll).to_csv("out/cokme_yeniden.csv", index=False)
    print("TAMAM", len(recs), "satir")
