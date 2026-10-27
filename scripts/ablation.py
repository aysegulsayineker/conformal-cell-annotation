"""Etiket uzayi ablasyonu: tutulan sinif LLM'in listesinden de cikarilir."""
import json, sys, numpy as np, pandas as pd
exec(open("recompute.py").read().split('if __name__')[0], globals())
exec(open("baseline.py").read(), globals())

DS = sys.argv[1]
KL = pd.read_csv({"pbmc":"handoff/clusters_pbmc_v2.csv","panc":"handoff/clusters_pancreas_v2.csv"}[DS])
KL["cluster"] = KL.cluster.astype(str); TRUTH = dict(zip(KL.cluster, KL.true_type))
labs, lcl, D, lcad = load_space(DS)
LLM = {"pbmc":{"gemma2-2b":"votes/oylar_v2_pbmc_gemma2_2b.json",
               "qwen25-7b":"votes/oylar_v2_pbmc_qwen25_7b.json"},
       "panc":{"gemma2-2b":"votes/oylar_v2_panc_gemma2_2b.json",
               "qwen25-7b":"votes/oylar_v2_panc_qwen25_7b.json"}}[DS]
order = [str(c) for c in KL.cluster]
O = json.load(open(f"handoff/cl_{DS}.json"))
def anc(c, s=None):
    s = s if s is not None else set(); s.add(c)
    for p in O["par"].get(c, []): anc(p, s)
    return s
ANC = {c: anc(c) for c in O["par"]}
DEP = {}
def _d(c):
    if c in DEP: return DEP[c]
    ps = O["par"].get(c, []); DEP[c] = 0 if not ps else 1+max(_d(p) for p in ps); return DEP[c]
for c in O["par"]: _d(c)

counts = KL.true_type.value_counts()
rows = []
for cls in [c for c in counts.index if counts[c] >= 2]:
    di = labs.index(cls)
    keep = [i for i in range(len(labs)) if i != di]
    labs2 = [labs[i] for i in keep]; lcl2 = [lcl[i] for i in keep]
    held = np.array([TRUTH[c] == cls for c in order])
    idx_in = np.flatnonzero(~held); idx_out = np.flatnonzero(held)
    Y2 = np.array([labs2.index(TRUTH[order[i]]) for i in idx_in])
    for k, v in LLM.items():
        P0 = probs(v, labs)[0][:, keep]
        P2 = P0 / P0.sum(1, keepdims=True)
        rng = np.random.default_rng(11)
        KO = onto_key(P2, lcl2, lcad)
        hits = []; szs = []; rss = []
        for _ in range(500):
            key = KO + rng.random(P2.shape)*1e-6
            S = aps_keyed(P2, rng.random(len(order)), key)
            st = S[idx_in, Y2]
            perm = rng.permutation(len(idx_in)); cal = idx_in[perm[:len(idx_in)//2]]
            stc = S[cal, np.array([labs2.index(TRUTH[order[i]]) for i in cal])]
            kq = int(np.ceil((len(cal)+1)*0.9))
            q = np.inf if kq > len(cal) else np.sort(stc)[kq-1]
            M = S[idx_out] <= q
            e = ~M.any(1)
            if e.any(): M[e, np.argmin(S[idx_out][e], 1)] = True
            for r_ in M:
                mem = [lcl2[j] for j in np.flatnonzero(r_)]
                cand = set.intersection(*[ANC[c] for c in mem]) if mem else set()
                node = max(cand, key=lambda c: DEP[c]) if cand else None
                hits.append(float(node is not None and node in ANC[lcl[di]]))
                szs.append(r_.sum()); rss.append(100*lcad(mem)/D[di])
        rows.append({"veri":DS, "tutulan":cls, "n_kume":int(counts[cls]), "model":k,
                     "ont_kapsama":round(float(np.mean(hits)),3),
                     "kume_boyu":round(float(np.mean(szs)),2),
                     "cozunurluk":round(float(np.mean(rss)),1)})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"out/ablasyon_{DS}.csv", index=False)
