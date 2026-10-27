"""Cozunurluk taramasi, ana analizin komsuluk grafigi sabit: ayni konformal yordam."""
import json, glob, numpy as np, pandas as pd
exec(open("recompute.py").read().split('if __name__')[0], globals())

JOBS = [(f, f"handoff/fixg_{json.load(open(f))['dataset']}_r{json.load(open(f))['resolution']}.csv")
        for f in sorted(glob.glob("votes_fixg/*_r*.json"))]
JOBS.append(("votes_paper/gemma2-2b_pbmc_PAPER.json", "handoff/clusters_pbmc_v2.csv"))

rows = []
for f, clf in JOBS:
    V = json.load(open(f)); ds = V["dataset"]; res = V.get("resolution", 3.0)
    KL = pd.read_csv(clf); KL["cluster"] = KL.cluster.astype(str)
    TRUTH = dict(zip(KL.cluster, KL.true_type))
    labs, lcl, D, lcad = load_space(ds)
    P, order, _ = probs(f, labs)
    Y = np.array([labs.index(TRUTH[c]) for c in order])
    n = len(Y); ncal = n // 2
    t1 = top1(V, TRUTH, order); col, tot, rate = collapse_rate(V)
    for a in (0.10, 0.20):
        need = int(np.ceil(1 / a)) - 1
        r, _ = evaluate(P, Y, D, lcad, lcl, a, "onto")
        rows.append({"model": V["model"], "veri": ds, "leiden": res, "kume": n, "n_cal": ncal,
                     "alpha": a, "uygun": bool(ncal >= need),
                     "saflik": round(float(KL.purity.mean()), 3),
                     "ortanca_boy": int(KL.n_cells.median()),
                     "top1": t1, "cokme": rate, **r})
        print(rows[-1], flush=True)
d = pd.DataFrame(rows).sort_values(["alpha", "veri", "leiden"])
d.to_csv("out/sabitgraf_duyarlilik.csv", index=False)
print("TAMAM", len(d), "satir")
