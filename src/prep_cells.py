import warnings, sys, os, numpy as np, pandas as pd
warnings.simplefilter("ignore")
import scanpy as sc
sc.settings.verbosity = 0

def finish(ad, hvg_mat, out, ref_csv):
    import anndata as adata_mod
    ad.obs["sub"] = ad.obs["sub"].astype(str)
    ad.obsm["hvg"] = np.asarray(hvg_mat, dtype=np.float32)
    ad.write_h5ad(out)
    ref = pd.read_csv(ref_csv)
    got = ad.obs.groupby("sub").size()
    chk = ref.assign(now=ref.cluster.astype(str).map(got))
    print(out, ad.shape, "hvg", ad.obsm["hvg"].shape,
          "| kume eslesme:", int((chk.n_cells == chk.now).sum()), "/", len(chk), flush=True)

w = sys.argv[1]
if w == "pbmc":
    ad = sc.datasets.pbmc3k_processed()
    ad.obs["true_type"] = ad.obs["louvain"].astype(str)
    if "neighbors" not in ad.uns: sc.pp.neighbors(ad, n_neighbors=15)
    sc.tl.leiden(ad, resolution=3.0, key_added="sub", flavor="igraph", n_iterations=2, directed=False)
    k = ad.obs["sub"].value_counts(); k = k[k >= 15].index
    ad = ad[ad.obs["sub"].isin(k)].copy()
    hvg = np.asarray(ad.X)                      # zaten HVG + olceklenmis
    raw = ad.raw.to_adata()[ad.obs_names].copy()  # tum genler, log1p
    raw.obs["sub"] = ad.obs["sub"].values; raw.obs["true_type"] = ad.obs["true_type"].values
    finish(raw, hvg, "handoff/cells_pbmc.h5ad", "handoff/clusters_pbmc_v2.csv")
else:
    ad = sc.datasets.ebi_expression_atlas("E-MTAB-5061")
    ad.obs["true_type"] = ad.obs["Factor Value[inferred cell type - ontology labels]"].astype(str)
    KEEP = ["pancreatic A cell","pancreatic ductal cell","type B pancreatic cell","pancreatic PP cell",
            "acinar cell","pancreatic D cell","pancreatic stellate cell","endothelial cell"]
    ad = ad[ad.obs.true_type.isin(KEEP)].copy(); ad.var_names_make_unique()
    sc.pp.filter_cells(ad, min_genes=500); sc.pp.filter_genes(ad, min_cells=5)
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad); ad.raw = ad
    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    sub = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(sub, max_value=10); sc.tl.pca(sub, n_comps=40); sc.pp.neighbors(sub, n_neighbors=15)
    sc.tl.leiden(sub, resolution=3.0, key_added="sub", flavor="igraph", n_iterations=2, directed=False)
    ad.obs["sub"] = sub.obs["sub"].values; sub.obs["sub"] = sub.obs["sub"].astype(str)
    k = ad.obs["sub"].value_counts(); k = k[k >= 15].index
    m = ad.obs["sub"].isin(k).values
    ad = ad[m].copy(); sub = sub[m].copy()
    raw = ad.raw.to_adata()[ad.obs_names].copy()
    raw.obs["sub"] = ad.obs["sub"].values; raw.obs["true_type"] = ad.obs["true_type"].values
    finish(raw, np.asarray(sub.X), "handoff/cells_panc.h5ad", "handoff/clusters_pancreas_v2.csv")
