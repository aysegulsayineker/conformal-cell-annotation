"""Referans-tabanli siniflandirici + konformal: LLM oylariyla ayni izgara."""
import json, numpy as np, pandas as pd, anndata as ad
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

def cl_aps_scores(P, u=None):
    order = np.argsort(-P, axis=1)
    Ps = np.take_along_axis(P, order, 1)
    cum_before = np.cumsum(Ps, 1) - Ps
    s_sorted = cum_before + (Ps if u is None else Ps * u[:, None])
    S = np.empty_like(s_sorted)
    np.put_along_axis(S, order, s_sorted, 1)
    return S

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

def conf_eval(P, Y, D, lca_depth, label_cl, alpha, n_rep=2000, seed=11, true_type=None):
    """APS + split-konformal; EK 8/EK 9 ile ayni metrik (birim basina LCA/gercek derinlik)."""
    n = len(Y); rng = np.random.default_rng(seed)
    rows, cls = [], {}
    for _ in range(n_rep):
        S = cl_aps_scores(P, rng.random(n)); st = S[np.arange(n), Y]
        perm = rng.permutation(n); cal, tst = perm[:n//2], perm[n//2:]
        k = int(np.ceil((len(cal)+1)*(1-alpha)))
        q = np.inf if k > len(cal) else np.sort(st[cal])[k-1]
        M = S[tst] <= q
        e = ~M.any(1)
        if e.any(): M[e, np.argmin(S[tst][e],1)] = True
        hit = M[np.arange(len(tst)), Y[tst]]
        res = np.mean([lca_depth([label_cl[j] for j in np.flatnonzero(r)])/D[t]
                       for r, t in zip(M, Y[tst])])
        rows.append((hit.mean(), M.sum(1).mean(), 100*res))
        if true_type is not None:
            for lab in set(true_type):
                sel = np.array([true_type[i] == lab for i in tst])
                if sel.any(): cls.setdefault(lab, []).extend(hit[sel].tolist())
    v = np.array(rows).mean(0)
    cc = pd.DataFrame([{"sinif": l, "kapsama": round(float(np.mean(x)), 3)} for l, x in cls.items()]) \
         if cls else None
    return {"kapsama": round(float(v[0]),3), "kume_boyu": round(float(v[1]),2),
            "cozunurluk": round(float(v[2]),1)}, cc

def novel_eval(P, Y_, D_, lca_, clids, alpha, drop_idx, n_rep=2000, seed=7,
               cal_has_novel=True, ANC=None, true_cl=None, lca_n=None):
    """Yeni-sinif deneyi. ANC+true_cl verilirse kapsama ontolojik olcutle hesaplanir:
    one surulen dugum gercek tipin atasi-veya-kendisi mi."""
    n = len(Y_); rng = np.random.default_rng(seed)
    is_novel = (Y_ == drop_idx); pool = np.flatnonzero(~is_novel)
    cov, res, sz = [], [], []
    for _ in range(n_rep):
        S = cl_aps_scores(P, rng.random(n)); st = S[np.arange(n), Y_]
        if cal_has_novel:
            perm = rng.permutation(n); cal, tst = perm[:n//2], perm[n//2:]
            if not is_novel[tst].any(): continue
        else:
            cal = rng.permutation(pool)[:n//2]
            tst = np.setdiff1d(np.arange(n), cal)
        k = int(np.ceil((len(cal)+1)*(1-alpha)))
        q = np.inf if k > len(cal) else np.sort(st[cal])[k-1]
        M = S[tst] <= q
        e = ~M.any(1)
        if e.any(): M[e, np.argmin(S[tst][e],1)] = True
        nv = is_novel[tst]
        if ANC is None:
            hit = M[np.arange(len(tst)), Y_[tst]][nv]
        else:
            hit = np.array([lca_n([clids[j] for j in np.flatnonzero(r)]) in ANC[true_cl[t]]
                            for r, t in zip(M[nv], Y_[tst][nv])])
        cov.append(hit.mean()); sz.append(M[nv].sum(1).mean())
        res.append(100*np.mean([lca_([clids[j] for j in np.flatnonzero(r)])/D_[t]
                                for r, t in zip(M[nv], Y_[tst][nv])]))
    return round(float(np.mean(cov)),3), round(float(np.mean(sz)),2), round(float(np.mean(res)),1)
