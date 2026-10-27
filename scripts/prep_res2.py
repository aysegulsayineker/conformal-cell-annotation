"""Cozunurluk taramasi, ANA ANALIZIN komsuluk grafigi sabit tutularak.

prep_res.py gommeyi ve grafigi yeniden kuruyordu; bu betik ana analizin grafigini kullanir,
boylece ayarlar arasinda degisen tek sey Leiden cozunurlugudur. r=3.0'da yayinlanan bolunmeyi
hucre-kume uyelikleri duzeyinde yeniden uretmesi gerekir; betik bunu dogrular ve
uretemezse durur.
"""
import sys, warnings, numpy as np, pandas as pd, scanpy as sc, anndata as ad
from sklearn.metrics import adjusted_rand_score
warnings.simplefilter("ignore"); sc.settings.verbosity = 0

DS = sys.argv[1]
RES = [float(x) for x in sys.argv[2].split(",")]
A = ad.read_h5ad(f"handoff/cells_{DS}.h5ad")          # tum genler, log1p, tutulan hucreler
PUB = {"pbmc": "handoff/clusters_pbmc_v2.csv", "panc": "handoff/clusters_pancreas_v2.csv"}[DS]

if DS == "pbmc":
    # ana analiz: islenmis nesnenin kendi komsuluk grafigi
    G = sc.datasets.pbmc3k_processed()
    G.obs["true_type"] = G.obs["louvain"].astype(str)
    if "neighbors" not in G.uns: sc.pp.neighbors(G, n_neighbors=15)
else:
    # ana analiz: olceklenmis HVG matrisi -> 40 bilesenli PCA -> 15 komsu
    G = ad.AnnData(X=np.asarray(A.obsm["hvg"]), obs=A.obs.copy())
    sc.tl.pca(G, n_comps=40, svd_solver="arpack")
    sc.pp.neighbors(G, n_neighbors=15)

# --- r=3.0 kontrolu: yayinlanan bolunme hucre duzeyinde yeniden uretiliyor mu ---
sc.tl.leiden(G, resolution=3.0, key_added="chk", flavor="igraph", n_iterations=2, directed=False)
common = A.obs_names.intersection(G.obs_names)
ari = adjusted_rand_score(A.obs.loc[common, "sub"].astype(str).values,
                          G.obs.loc[common, "chk"].astype(str).values)
print(f"{DS}: r=3.0 kontrolu ortak hucre {len(common)}/{A.n_obs} ARI {ari:.4f}", flush=True)
assert ari > 0.999, f"{DS}: ana analizin grafigi yeniden uretilemedi (ARI {ari:.4f})"

rows = []
for r in RES:
    key = f"L{r}"
    sc.tl.leiden(G, resolution=r, key_added=key, flavor="igraph", n_iterations=2, directed=False)
    lab = G.obs.loc[common, key].astype(str)
    keep = [c for c in lab.unique() if (lab == c).sum() >= 15]
    sel = lab.isin(keep)
    tt = A.obs.loc[common, "true_type"]
    pur = [(tt[(lab == c).values].value_counts().iloc[0]) / (lab == c).sum() for c in keep]
    sizes = [int((lab == c).sum()) for c in keep]
    rows.append(dict(cozunurluk=r, kume=len(keep), hucre=int(sum(sizes)),
                     saflik=round(float(np.mean(pur)), 3),
                     ortanca_boy=int(np.median(sizes)),
                     tip_hucre=int(tt[sel.values].nunique())))
    print(rows[-1], flush=True)

    sub = A[common][sel.values].copy()
    sub.obs["cl"] = lab[sel].values
    sc.tl.rank_genes_groups(sub, "cl", method="wilcoxon")
    de = {c: list(sub.uns["rank_genes_groups"]["names"][c][:8]) for c in keep}
    top = {}
    if DS == "panc":
        Xd = np.asarray(sub.X.todense()) if hasattr(sub.X, "todense") else np.asarray(sub.X)
        for c in keep:
            mu = Xd[(sub.obs.cl == c).values].mean(0)
            top[c] = [sub.var_names[i] for i in np.argsort(-mu)[:8]]
    out = []
    for c in keep:
        m = (lab == c).values
        vc = tt[m].value_counts()
        out.append(dict(cluster=c, n_cells=int(m.sum()), true_type=vc.index[0],
                        purity=round(float(vc.iloc[0] / m.sum()), 3),
                        de_markers=",".join(de[c]), top_expressed=",".join(top.get(c, []))))
    pd.DataFrame(out).to_csv(f"handoff/fixg_{DS}_r{r}.csv", index=False)

pd.DataFrame(rows).to_csv(f"out/sabitgraf_ozet_{DS}.csv", index=False)
print("TAMAM", DS, flush=True)
