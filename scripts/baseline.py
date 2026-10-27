"""Referans siniflandirici, ontoloji kurulumu ve Laplace yumusatma.
Uygunsuzluk skoru ve esitlik cozumu BURADA DEGIL, recompute.py icindedir (aps_keyed);
bu dosya yalnizca onun ihtiyac duydugu yardimcilari saglar.
Derinlik DAG uzerinde EN UZUN yoldan, ortak ata tam ata kapanimlarinin kesisiminden.
"""
import json, numpy as np, pandas as pd, anndata as ad
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

def smooth(P, K=10, s=0.05):
    """LLM oylariyla ayni son-islem: K sozde-oy + Laplace duzeltmesi."""
    C = np.asarray(P, dtype=np.float64) * K
    return (C + s) / (C.sum(1, keepdims=True) + s * P.shape[1])

def build_onto(par, label_cl):
    """par: CL id -> ebeveynler. label_cl: etiket metni -> CL id."""
    def ancs(c, seen=None):
        seen = seen or set()
        if c in seen: return seen
        seen.add(c)
        for p in par.get(c, []): ancs(p, seen)
        return seen
    A = {c: ancs(c) for c in par}
    dep = {}
    def depth(c):
        if c in dep: return dep[c]
        ps = [p for p in par.get(c, []) if p in par and p != c]
        dep[c] = 0 if not ps else 1 + max(depth(p) for p in ps)
        return dep[c]
    for c in par: depth(c)
    def _common(cl_ids):
        return set.intersection(*[A[c] for c in cl_ids]) if cl_ids else {"CL:0000000"}
    def lca_depth(cl_ids):
        common = _common(cl_ids)
        return max(dep[c] for c in common) if common else 0
    def lca_node(cl_ids):
        common = _common(cl_ids)
        return max(common, key=lambda c: dep[c]) if common else "CL:0000000"
    return dep, lca_depth, lca_node, A

def ref_probs(h5ad_path, cluster_order, labels, drop_class=None, seeds=(0,1,2), C=1.0):
    """Hucre duzeyinde %50/%50 referans/sorgu bolunmesi; kume basina ortalama olasilik."""
    a = ad.read_h5ad(h5ad_path)
    X = np.asarray(a.obsm["hvg"], dtype=np.float32)
    y = a.obs["true_type"].astype(str).values
    g = a.obs["sub"].astype(str).values
    L = {l: i for i, l in enumerate(labels)}
    acc = np.zeros((len(cluster_order), len(labels)), dtype=np.float64)
    for s in seeds:
        tr, te = next(StratifiedShuffleSplit(1, train_size=0.5, random_state=s).split(X, y))
        if drop_class is not None:
            tr = tr[y[tr] != drop_class]
        m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        Pc = m.predict_proba(X[te])
        cols = [L[c] for c in m.classes_]
        for i, cid in enumerate(cluster_order):
            sel = g[te] == str(cid)
            if sel.sum() == 0: continue
            acc[i, cols] += Pc[sel].mean(0)
    return acc / len(seeds)

