import numpy as np

def cl_onto_depths(parent_map):
    """Depth of every node in a parent->parent mapping whose root is 'root'."""
    depths = {}
    for node in list(parent_map) + ["root"]:
        chain, cur = [node], node
        while cur != "root":
            cur = parent_map[cur]
            chain.append(cur)
        depths[node] = len(chain) - 1
    return depths

def cl_onto_lca(parent_map, labels):
    """Lowest common ancestor of `labels` in the ontology."""
    chains = []
    for lab in labels:
        chain, cur = [lab], lab
        while cur != "root":
            cur = parent_map[cur]
            chain.append(cur)
        chains.append(list(reversed(chain)))
    out = "root"
    for tier in zip(*chains):
        if len(set(tier)) == 1:
            out = tier[0]
        else:
            break
    return out

def cl_probs_from_counts(count_dicts, labels, smoothing=None):
    """Vote counters -> smoothed probability matrix (n_units x n_labels)."""
    if smoothing is None:
        smoothing = 0.05
    P = np.zeros((len(count_dicts), len(labels)))
    for i, c in enumerate(count_dicts):
        tot = sum(c.values())
        for j, lab in enumerate(labels):
            P[i, j] = (c.get(lab, 0) + smoothing) / (tot + smoothing * len(labels))
    return P

def cl_aps_scores(P, u=None):
    """Adaptive-prediction-set nonconformity scores; `u` enables randomization."""
    order = np.argsort(-P, axis=1)
    Ps = np.take_along_axis(P, order, 1)
    cum_before = np.cumsum(Ps, 1) - Ps
    s_sorted = cum_before + (Ps if u is None else Ps * u[:, None])
    S = np.empty_like(s_sorted)
    np.put_along_axis(S, order, s_sorted, 1)
    return S

def cl_conformal_grid(P, y_idx, labels, parent_map, alphas=None, n_cal=None,
                      n_rep=600, seed=0, randomize=True):
    """Split-conformal over units: coverage, set size, asserted ontology depth.

    Returns one dict per alpha. `cozunurluk_pct` is the asserted node's depth as
    a percentage of leaf depth - the informativeness metric.
    """
    if alphas is None:
        alphas = (0.05, 0.10, 0.20)
    n = P.shape[0]
    if n_cal is None:
        n_cal = n // 2
    depths = cl_onto_depths(parent_map)
    maxd = max(depths[lab] for lab in labels)
    cache, rng, acc = {}, np.random.default_rng(seed), {a: [] for a in alphas}
    for _ in range(n_rep):
        u = rng.random(n) if randomize else None
        S = cl_aps_scores(P, u)
        s_true = S[np.arange(n), y_idx]
        perm = rng.permutation(n)
        cal, tst = perm[:n_cal], perm[n_cal:]
        s_sorted = np.sort(s_true[cal])
        for a in alphas:
            k = int(np.ceil((n_cal + 1) * (1 - a)))
            q = np.inf if k > n_cal else s_sorted[k - 1]
            M = S[tst] <= q
            empty = ~M.any(1)
            if empty.any():
                M[empty, np.argmin(S[tst][empty], axis=1)] = True
            dep = []
            for row in M:
                key = row.tobytes()
                if key not in cache:
                    cache[key] = depths[cl_onto_lca(parent_map,
                                   [labels[j] for j in np.flatnonzero(row)])]
                dep.append(cache[key])
            acc[a].append((M[np.arange(len(tst)), y_idx[tst]].mean(),
                           M.sum(1).mean(), float(np.mean(dep))))
    out = []
    for a in alphas:
        v = np.array(acc[a]).mean(0)
        out.append({"alpha": a, "hedef_kapsama": round(1 - a, 3),
                    "kapsama": round(float(v[0]), 3),
                    "kume_boyu": round(float(v[1]), 2),
                    "cozunurluk_pct": round(100 * float(v[2]) / maxd, 1),
                    "n_cal": n_cal, "n_unit": n})
    return out

def cl_min_calibration_units(alpha):
    """Smallest calibration set for which split conformal can reach 1-alpha."""
    import math
    return int(math.ceil(1.0 / alpha)) - 1
