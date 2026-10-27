"""Leiden cozunurluk duyarliligi: her cozunurlukte ayni konformal yordam."""
import json, glob, numpy as np, pandas as pd
exec(open("recompute.py").read().split('if __name__')[0], globals())

rows = []
for f in sorted(glob.glob("votes_res/*_r*.json")):
    V = json.load(open(f))
    ds, res, mdl = V["dataset"], V["resolution"], V["model"]
    KL = pd.read_csv(f"handoff/clusters_{ds}_r{res}.csv"); KL["cluster"] = KL.cluster.astype(str)
    TRUTH = dict(zip(KL.cluster, KL.true_type))
    labs, lcl, D, lcad = load_space(ds)
    P, order, _ = probs(f, labs)
    Y = np.array([labs.index(TRUTH[c]) for c in order])
    n = len(Y); ncal = n // 2
    t1 = top1(V, TRUTH, order); col, tot, rate = collapse_rate(V)
    for a in (0.10, 0.20):
        need = int(np.ceil(1 / a)) - 1
        r, _ = evaluate(P, Y, D, lcad, lcl, a, "onto")
        rows.append({"model": mdl, "veri": ds, "leiden": res, "kume": n, "n_cal": ncal,
                     "alpha": a, "uygun": bool(ncal >= need),
                     "saflik": round(float(KL.purity.mean()), 3),
                     "top1": t1, "cokme": rate, **r})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("out/cozunurluk_duyarlilik.csv", index=False)
print("TAMAM", len(rows), "satir")
