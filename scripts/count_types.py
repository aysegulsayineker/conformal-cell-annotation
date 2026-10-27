"""Her cozunurlukte KORUNAN kumelerdeki HUCRELER uzerinden gercek tip sayisi.
Kume CSV'lerine dokunmaz; yalnizca sayar ve mevcut CSV'lerle tutarliligi dogrular."""
import sys, numpy as np, pandas as pd, scanpy as sc, anndata as ad
DS = sys.argv[1]; RES = [float(x) for x in sys.argv[2].split(",")]
A = ad.read_h5ad(f"handoff/cells_{DS}.h5ad")
H = ad.AnnData(X=np.asarray(A.obsm["hvg"]), obs=A.obs.copy())
sc.pp.scale(H, max_value=10)
sc.tl.pca(H, n_comps=50, svd_solver="arpack")
sc.pp.neighbors(H, n_neighbors=15, n_pcs=40)
rows = []
for r in RES:
    sc.tl.leiden(H, resolution=r, key_added="k", flavor="igraph", n_iterations=2, directed=False)
    lab = H.obs["k"].astype(str)
    keep = [c for c in lab.unique() if (lab == c).sum() >= 15]
    sel = lab.isin(keep).values
    tip_hucre = int(A.obs.true_type[sel].nunique())              # DOGRU olcut
    maj = [A.obs.true_type[(lab == c).values].value_counts().index[0] for c in keep]
    tip_cogunluk = len(set(maj))                                  # farkli, daha dar olcut
    ref = pd.read_csv(f"handoff/clusters_{DS}_r{r}.csv")
    rows.append(dict(veri=DS, cozunurluk=r, kume=len(keep), hucre=int(sel.sum()),
                     tip_hucre=tip_hucre, tip_cogunluk=tip_cogunluk,
                     kume_eslesti=(len(keep) == len(ref)),
                     hucre_eslesti=(int(sel.sum()) == int(ref.n_cells.sum()))))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"out/tip_sayimi_{DS}.csv", index=False)
