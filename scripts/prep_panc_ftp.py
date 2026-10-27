"""Pankreas verisini EBI FTP arsivinden kurar (gxa API 500 veriyor).
Scanpy yukleyicisinin urettigi AnnData'yi birebir taklit eder."""
import warnings, numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sp
warnings.simplefilter("ignore")
import anndata as ad_mod, scanpy as sc
sc.settings.verbosity = 0
D = "dl/emtab/E-MTAB-5061.aggregated_filtered_counts"
M = sio.mmread(D + ".mtx").tocsr()                      # genler x hucreler
genes = pd.read_csv(D + ".mtx_rows", sep="\t", header=None)[0].values
cells = pd.read_csv(D + ".mtx_cols", sep="\t", header=None)[0].values
A = ad_mod.AnnData(X=sp.csr_matrix(M.T), obs=pd.DataFrame(index=cells),
                   var=pd.DataFrame(index=genes))
md = pd.read_csv("dl/emtab/cell_metadata.tsv", sep="\t", low_memory=False).set_index("id")
A.obs["Factor Value[inferred cell type - ontology labels]"] = \
    md.loc[A.obs_names, "inferred_cell_type_-_ontology_labels"].astype(str).values
A.obs["true_type"] = A.obs["Factor Value[inferred cell type - ontology labels]"].astype(str)
KEEP = ["pancreatic A cell","pancreatic ductal cell","type B pancreatic cell","pancreatic PP cell",
        "acinar cell","pancreatic D cell","pancreatic stellate cell","endothelial cell"]
A = A[A.obs.true_type.isin(KEEP)].copy(); A.var_names_make_unique()
sc.pp.filter_cells(A, min_genes=500); sc.pp.filter_genes(A, min_cells=5)
sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A); A.raw = A
sc.pp.highly_variable_genes(A, n_top_genes=2000)
sub = A[:, A.var.highly_variable].copy()
sc.pp.scale(sub, max_value=10); sc.tl.pca(sub, n_comps=40); sc.pp.neighbors(sub, n_neighbors=15)
sc.tl.leiden(sub, resolution=3.0, key_added="sub", flavor="igraph", n_iterations=2, directed=False)
A.obs["sub"] = sub.obs["sub"].values
k = A.obs["sub"].value_counts(); k = k[k >= 15].index
m = A.obs["sub"].isin(k).values
A = A[m].copy(); sub = sub[m].copy()
raw = A.raw.to_adata()[A.obs_names].copy()
raw.obs["sub"] = A.obs["sub"].astype(str).values
raw.obs["true_type"] = A.obs["true_type"].values
raw.obsm["hvg"] = np.asarray(sub.X, dtype=np.float32)
raw.write_h5ad("handoff/cells_panc.h5ad")
ref = pd.read_csv("handoff/clusters_pancreas_v2.csv")
got = raw.obs.groupby("sub").size()
chk = ref.assign(now=ref.cluster.astype(str).map(got))
print("cells_panc.h5ad", raw.shape, "hvg", raw.obsm["hvg"].shape,
      "| kume eslesme:", int((chk.n_cells == chk.now).sum()), "/", len(chk),
      "| kume sayisi:", raw.obs['sub'].nunique())
