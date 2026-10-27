"""Leiden cozunurluk duyarliligi: ayni veriden birkac cozunurlukte kume + istem tablosu.
Protokol v2 ile ayni: PBMC icin yalniz DE genleri, pankreas icin DE + en cok ifade edilenler.
"""
import sys, json, numpy as np, pandas as pd, scanpy as sc, anndata as ad

DS = sys.argv[1]                      # pbmc | panc
RES = [float(x) for x in sys.argv[2].split(",")]
A = ad.read_h5ad(f"handoff/cells_{DS}.h5ad")
HV = A.obsm["hvg"]                    # hvg matrisi onceden kaydedilmis
H = ad.AnnData(X=np.asarray(HV), obs=A.obs.copy())
sc.pp.scale(H, max_value=10)
sc.tl.pca(H, n_comps=50, svd_solver="arpack")
sc.pp.neighbors(H, n_neighbors=15, n_pcs=40)

rows = []
for r in RES:
    key = f"leiden_{r}"
    sc.tl.leiden(H, resolution=r, key_added=key, flavor="igraph", n_iterations=2, directed=False)
    A.obs[key] = H.obs[key].values
    lab = H.obs[key].astype(str)
    keep = [c for c in lab.unique() if (lab == c).sum() >= 15]
    pur, sizes = [], []
    for c in keep:
        m = lab == c
        vc = A.obs.true_type[m.values].value_counts()
        pur.append(vc.iloc[0] / m.sum()); sizes.append(int(m.sum()))
    rows.append(dict(cozunurluk=r, kume=len(keep), hucre=int(sum(sizes)),
                     saflik=round(float(np.mean(pur)), 3),
                     ortanca_boy=int(np.median(sizes)),
                     tip=int(A.obs.true_type[lab.isin(keep).values].nunique())))
    print(rows[-1], flush=True)

    # istem tablosu
    sub = A[lab.isin(keep).values].copy()
    sub.obs["cl"] = lab[lab.isin(keep)].values
    sc.tl.rank_genes_groups(sub, "cl", method="wilcoxon")
    de = {c: [g for g in sub.uns["rank_genes_groups"]["names"][c][:8]] for c in keep}
    top = {}
    if DS == "panc":
        Xd = np.asarray(sub.X.todense()) if hasattr(sub.X, "todense") else np.asarray(sub.X)
        for c in keep:
            m = (sub.obs.cl == c).values
            mu = Xd[m].mean(0)
            top[c] = [sub.var_names[i] for i in np.argsort(-mu)[:8]]
    out = []
    for c in keep:
        m = (lab == c).values
        vc = A.obs.true_type[m].value_counts()
        out.append(dict(cluster=c, n_cells=int(m.sum()), true_type=vc.index[0],
                        purity=round(float(vc.iloc[0]/m.sum()), 3),
                        de_markers=",".join(de[c]),
                        top_expressed=",".join(top.get(c, []))))
    pd.DataFrame(out).to_csv(f"handoff/clusters_{DS}_r{r}.csv", index=False)

pd.DataFrame(rows).to_csv(f"out/cozunurluk_ozet_{DS}.csv", index=False)
print("TAMAM", DS)
