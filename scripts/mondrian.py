"""Sinif-kosullu kapsama: marjinal vs grup bazli (Mondrian) esik + coklu tohum.
Esitlikler ontolojiye gore bozulur (recompute.py ile ayni yordam)."""
import json, glob, numpy as np, pandas as pd
exec(open("recompute.py").read().split('if __name__')[0], globals())

KL = pd.read_csv("handoff/clusters_pancreas_v2.csv"); KL["cluster"] = KL.cluster.astype(str)
TRUTH = dict(zip(KL.cluster, KL.true_type))
labs, lcl, D, lcad = load_space("panc")
O = json.load(open("handoff/cl_panc.json"))
DEP = {}
def _d(c):
    if c in DEP: return DEP[c]
    ps = O["par"].get(c, []); DEP[c] = 0 if not ps else 1+max(_d(p) for p in ps); return DEP[c]
for c in O["par"]: _d(c)
def anc(c, s=None):
    s = s if s is not None else set(); s.add(c)
    for p in O["par"].get(c, []): anc(p, s)
    return s
ANC = {c: anc(c) for c in O["par"]}

# grup = derinlik-4 ata (yoksa en yakin ust)
def group_of(cl_id):
    a = [c for c in ANC[cl_id] if DEP[c] == 4]
    if a: return sorted(a)[0]
    a = [c for c in ANC[cl_id] if DEP[c] <= 4]
    return sorted(a, key=lambda c: -DEP[c])[0] if a else "root"
GRP = {labs[i]: group_of(lcl[i]) for i in range(len(labs))}

def evaluate_cc(P, Y, alpha, mode_thr, n_rep=2000, seed=11):
    """mode_thr: 'marginal' | 'mondrian'. Sinif-kosullu kapsama dondurur."""
    n, m = P.shape; rng = np.random.default_rng(seed)
    KO = onto_key(P, lcl, lcad)
    gid = np.array([list(dict.fromkeys(GRP.values())).index(GRP[labs[y]]) for y in Y])
    need = int(np.ceil(1/alpha)) - 1
    cc = {}; sz = []; rs = []
    for _ in range(n_rep):
        key = KO + rng.random((n, m))*1e-6
        S = aps_keyed(P, rng.random(n), key); st = S[np.arange(n), Y]
        perm = rng.permutation(n); cal, tst = perm[:n//2], perm[n//2:]
        if mode_thr == "marginal":
            k = int(np.ceil((len(cal)+1)*(1-alpha)))
            q = np.full(n, np.inf if k > len(cal) else np.sort(st[cal])[k-1])
        else:
            q = np.full(n, np.inf)
            for g in np.unique(gid):
                cg = cal[gid[cal] == g]
                if len(cg) >= need:
                    k = int(np.ceil((len(cg)+1)*(1-alpha)))
                    if k <= len(cg): q[gid == g] = np.sort(st[cg])[k-1]
        M = S[tst] <= q[tst][:, None]
        e = ~M.any(1)
        if e.any(): M[e, np.argmin(S[tst][e],1)] = True
        hit = M[np.arange(len(tst)), Y[tst]]
        sz.append(M.sum(1).mean())
        rs.append(100*np.mean([lcad([lcl[j] for j in np.flatnonzero(r)])/D[t]
                               for r, t in zip(M, Y[tst])]))
        for i, t in enumerate(tst):
            cc.setdefault(labs[Y[t]], []).append(float(hit[i]))
    return ({k: round(float(np.mean(v)),3) for k, v in cc.items()},
            round(float(np.mean(sz)),2), round(float(np.mean(rs)),1))

rows, summ = [], []
for mdl, f in (("qwen25-7b","votes/oylar_v2_panc_qwen25_7b.json"),
               ("gemma2-2b","votes/oylar_v2_panc_gemma2_2b.json")):
    P, order, _ = probs(f, labs); Y = np.array([labs.index(TRUTH[c]) for c in order])
    for a in (0.10, 0.20):
        for thr in ("marginal","mondrian"):
            cc, s, r = evaluate_cc(P, Y, a, thr)
            summ.append({"model":mdl,"alpha":a,"esik":thr,"kume_boyu":s,"cozunurluk":r})
            for k, v in cc.items():
                rows.append({"model":mdl,"alpha":a,"esik":thr,"sinif":k,
                             "n_kume":int((KL.true_type==k).sum()),"kapsama":v,
                             "grup":GRP[k]})
            print(mdl, a, thr, "kume", s, "coz", r, flush=True)
pd.DataFrame(rows).to_csv("out/mondrian_sinif_yeniden.csv", index=False)
pd.DataFrame(summ).to_csv("out/mondrian_ozet_yeniden.csv", index=False)

# --- coklu tohum ---
sd = []
for mdl in ("qwen25-7b","gemma2-2b"):
    files = [("1234", f"votes/oylar_v2_panc_{mdl.replace('-','_') if mdl=='gemma2-2b' else mdl}.json")]
    files = [("1234", "votes/oylar_v2_panc_qwen25_7b.json" if mdl=="qwen25-7b" else "votes/oylar_v2_panc_gemma2_2b.json"),
             ("7",    f"seeds_oylar_v2_panc_{mdl}_s7.json"),
             ("99",   f"seeds_oylar_v2_panc_{mdl}_s99.json")]
    per = {}
    for tag, f in files:
        P, order, _ = probs(f, labs); Y = np.array([labs.index(TRUTH[c]) for c in order])
        cc, _, _ = evaluate_cc(P, Y, 0.10, "marginal", n_rep=800)
        for k, v in cc.items(): per.setdefault(k, {})[tag] = v
    for k, d in per.items():
        vs = list(d.values())
        sd.append({"model":mdl,"sinif":k,"n_kume":int((KL.true_type==k).sum()),
                   "ort":round(float(np.mean(vs)),3),"min":min(vs),"max":max(vs),
                   "yayilim":round(max(vs)-min(vs),3),
                   "tohum_nominalin_altinda":sum(1 for v in vs if v < 0.90)})
    print("tohumlar", mdl, "bitti", flush=True)
pd.DataFrame(sd).to_csv("out/tohum_yeniden.csv", index=False)
print("TAMAM")
