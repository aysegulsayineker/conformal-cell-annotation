import warnings, sys, re, numpy as np, pandas as pd
warnings.simplefilter("ignore")
import scanpy as sc
sc.settings.verbosity = 0
NDE, NTOP = 30, 15

def markers_table(ad, label_col, out):
    raw = ad.raw.to_adata()
    raw.obs["sub"] = ad.obs["sub"].values
    sc.tl.rank_genes_groups(raw, "sub", method="wilcoxon", n_genes=NDE)
    de = pd.DataFrame(raw.uns["rank_genes_groups"]["names"])
    X = raw.X
    rows = []
    for c in de.columns:
        mask = (raw.obs["sub"] == c).values
        sub = X[mask]
        mean = np.asarray(sub.mean(axis=0)).ravel()
        order = np.argsort(-mean)
        top = [str(g) for g in raw.var_names[order][:150]]
        s = ad.obs[ad.obs["sub"] == c]
        vc = s[label_col].value_counts()
        rows.append({"cluster": c, "n_cells": int(len(s)), "true_type": vc.index[0],
                     "purity": round(float(vc.iloc[0]/len(s)), 3),
                     "de_markers": ",".join(de[c].tolist()),
                     "top_expressed": ",".join(top)})
    df = pd.DataFrame(rows).sort_values("cluster")
    df.to_csv(out, index=False)
    print(out, len(df), "kume | ort saflik", round(df.purity.mean(), 3), flush=True)

which = sys.argv[1]
if which == "pbmc":
    ad = sc.datasets.pbmc3k_processed()
    ad.obs["true_type"] = ad.obs["louvain"].astype(str)
    if "neighbors" not in ad.uns: sc.pp.neighbors(ad, n_neighbors=15)
    sc.tl.leiden(ad, resolution=3.0, key_added="sub", flavor="igraph", n_iterations=2, directed=False)
    k = ad.obs["sub"].value_counts(); k = k[k >= 15].index
    ad = ad[ad.obs["sub"].isin(k)].copy(); ad.obs["sub"] = ad.obs["sub"].astype(str)
    markers_table(ad, "true_type", "handoff/clusters_pbmc_v2.csv")
else:
    import os
    if os.path.exists("handoff/pancreas_proc.h5ad"):
        ad = sc.read_h5ad("handoff/pancreas_proc.h5ad")
        markers_table(ad, "true_type", "handoff/clusters_pancreas_v2.csv"); sys.exit()
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
    ad.obs["sub"] = sub.obs["sub"].values
    k = ad.obs["sub"].value_counts(); k = k[k >= 15].index
    ad = ad[ad.obs["sub"].isin(k)].copy(); ad.obs["sub"] = ad.obs["sub"].astype(str)
    ad.write_h5ad("handoff/pancreas_proc.h5ad")
    markers_table(ad, "true_type", "handoff/clusters_pancreas_v2.csv")
