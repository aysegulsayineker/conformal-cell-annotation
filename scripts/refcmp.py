"""Temel karsilastirma + yeni-tip deneyi + etiket uzayi ablasyonu, ontolojik esitlik cozumuyle."""
import json, sys, numpy as np, pandas as pd
exec(open("recompute.py").read().split('if __name__')[0], globals())
exec(open("baseline.py").read(), globals())          # ref_probs

DS = sys.argv[1]                                      # pbmc | panc
H5 = {"pbmc":"handoff/cells_pbmc.h5ad","panc":"handoff/cells_panc.h5ad"}[DS]
KL = pd.read_csv({"pbmc":"handoff/clusters_pbmc_v2.csv","panc":"handoff/clusters_pancreas_v2.csv"}[DS])
KL["cluster"] = KL.cluster.astype(str); TRUTH = dict(zip(KL.cluster, KL.true_type))
labs, lcl, D, lcad = load_space(DS)
LLM = {"pbmc":{"gemma2-2b":"votes/oylar_v2_pbmc_gemma2_2b.json",
               "qwen25-7b":"votes/oylar_v2_pbmc_qwen25_7b.json"},
       "panc":{"gemma2-2b":"votes/oylar_v2_panc_gemma2_2b.json",
               "qwen25-7b":"votes/oylar_v2_panc_qwen25_7b.json"}}[DS]
order = [str(c) for c in KL.cluster]
Y = np.array([labs.index(TRUTH[c]) for c in order])
ANC = {c: set() for c in lcl}
O = json.load(open(f"handoff/cl_{DS}.json"))
def anc(c, s=None):
    s = s if s is not None else set(); s.add(c)
    for p in O["par"].get(c, []): anc(p, s)
    return s
ANC = {}
for c in O["par"]:
    ANC[c] = anc(c)
DEP = {}
def _d(c):
    if c in DEP: return DEP[c]
    ps = O["par"].get(c, [])
    DEP[c] = 0 if not ps else 1 + max(_d(p) for p in ps)
    return DEP[c]
for c in O["par"]: _d(c)

rows = []
# --- 1) temel karsilastirma ---
Pref = ref_probs(H5, order, labs)
for src, P in [(f"LLM {k}", probs(v, labs)[0]) for k, v in LLM.items()] + [("reference classifier", smooth(Pref))]:
    for a in (0.05, 0.10, 0.20):
        r, _ = evaluate(P, Y, D, lcad, lcl, a, "onto")
        rows.append({"veri":DS, "kaynak":src, "alpha":a, **r})
        print(DS, src, a, r, flush=True)
pd.DataFrame(rows).to_csv(f"out/temel_{DS}.csv", index=False)

# --- 2) yeni tip: sinif referanstan cikarilir ---
nv = []
counts = KL.true_type.value_counts()
for cls in [c for c in counts.index if counts[c] >= 2]:
    di = labs.index(cls)
    Pr = smooth(ref_probs(H5, order, labs, drop_class=cls))
    for k, v in LLM.items():
        Pl = probs(v, labs)[0]
        for nm, P in (("reference classifier", Pr), (f"LLM {k}", Pl)):
            sel = np.array([TRUTH[c] == cls for c in order])
            r, _ = evaluate(P, Y, D, lcad, lcl, 0.10, "onto")
            # sinifa ozel kapsama: ontolojik olcut
            rng = np.random.default_rng(11); KO = onto_key(P, lcl, lcad); hits=[]; szs=[]; rss=[]
            for _ in range(500):
                key = KO + rng.random(P.shape)*1e-6
                S = aps_keyed(P, rng.random(len(Y)), key); st = S[np.arange(len(Y)), Y]
                perm = rng.permutation(len(Y)); cal, tst = perm[:len(Y)//2], perm[len(Y)//2:]
                kq = int(np.ceil((len(cal)+1)*0.9)); q = np.inf if kq>len(cal) else np.sort(st[cal])[kq-1]
                M = S[tst] <= q
                e = ~M.any(1)
                if e.any(): M[e, np.argmin(S[tst][e],1)] = True
                for r_, t_ in zip(M, tst):
                    if not sel[t_]: continue
                    mem = [lcl[j] for j in np.flatnonzero(r_)]
                    node_d = lcad(mem)
                    # iddia edilen dugum: uyelerin en dusuk ortak atasi
                    cand = set.intersection(*[ANC[c] for c in mem]) if mem else set()
                    node = max(cand, key=lambda c: DEP[c]) if cand else None
                    # ontolojik olcut: iddia edilen dugum gercek tipin atasi ya da kendisi mi
                    hits.append(float(node is not None and node in ANC[lcl[di]]))
                    szs.append(r_.sum()); rss.append(100*node_d/D[di])
            nv.append({"veri":DS,"tutulan":cls,"n_kume":int(counts[cls]),"kaynak":nm,
                       "ont_kapsama":round(float(np.mean(hits)),3),
                       "kume_boyu":round(float(np.mean(szs)),2),
                       "cozunurluk":round(float(np.mean(rss)),1)})
            print("  yeni-tip", cls, nm, nv[-1]["ont_kapsama"], flush=True)
pd.DataFrame(nv).to_csv(f"out/yenitip_{DS}.csv", index=False)
print("TAMAM", DS)
