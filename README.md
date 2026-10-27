# Conformal cell type annotation with ontology back-off

Distribution-free guarantees for LLM-based scRNA-seq cell type annotation. Instead of
abstaining when uncertain, the method backs off up the Cell Ontology: it asserts
*T cell* at 1 − α confidence rather than returning the set {CD4 T, CD8 T, NK}.

Two quantities are reported separately because they dissociate:

- **validity** — empirical coverage against the nominal level 1 − α
- **resolution** — depth of the asserted ontology node as a fraction of the true label's
  depth (informativeness)

This repository contains the code, the raw model votes, and the result tables behind the
manuscript.

## Headline results

| | |
|---|---|
| Datasets | PBMC3k (10x, 36 clusters) and E-MTAB-5061 Segerstolpe pancreas (Smart-seq2, 35 clusters) |
| Models | 6 free local models (llama.cpp, CPU) + Claude Haiku 4.5 + Claude Sonnet 4.5 |
| Validity | holds in all 16 model-by-dataset configurations at every level tested under **within-dataset** calibration (lowest 0.944 against 0.90, 0.832 against 0.80); pooled calibration under-covers on the pancreas at α = 0.10 (0.891), and at α = 0.20 both cross-dataset (0.655) and pooled (0.709) under-cover there |
| Resolution | varies more than fourfold across models (23.8% – 100.0% at α = 0.10) |
| Clustering resolution | coverage holds at every Leiden setting whose calibration half clears the feasibility bar; resolution has an interior optimum, and the coarsest feasible setting is the most informative at α = 0.10 on both datasets |
| Structural limit | α ≤ 0.05 needs n_cal ≥ ⌈1/α⌉ − 1 calibration units; pooling across datasets unlocks it |
| Tie resolution | most of the label vector sits at the smoothing floor, so the ontology supplies the informativeness: breaking those ties at random leaves resolution at 13.9% – 18.6% in thirteen of sixteen cells |
| Class-conditional | group-wise (Mondrian) thresholds relocate the violation rather than removing it; the feasibility bar applies to a group's units in the **calibration half** of each split, so at α = 0.10 only the endocrine group clears it, in 86.9% of splits |
| Baseline | a reference-based classifier is far more informative *when* an in-tissue reference exists (95.3% and 97.5% against 37.9% and 53.2%); when a class is removed from the reference it stays ontologically valid by widening its sets, and whether an LLM asserts a deeper node for the same coverage depends on the model and the dataset |

## Layout

```
scripts/
  recompute.py            the single canonical procedure: APS with ontology-aware tie-breaking
  baseline.py             reference classifier, ontology helpers, conformal evaluation
  refcmp.py               baseline comparison and the held-out-type experiment
  ablation.py             held-out type also removed from the prompted label list
  mondrian.py             group-wise thresholds, class-conditional coverage, multi-seed
  pooling.py              cross-dataset and pooled calibration regimes
  prep_cells.py           cell-level matrices for the reference classifier
  prep_panc_ftp.py        pancreas rebuild from the EBI FTP mirror
  run_local_v4.py         grammar-constrained voting, local GGUF models
  api_run.py              API model runs
labels/                   the 14-label spaces and their Cell Ontology mappings
votes/                    raw per-cluster vote counts for every run (K = 10)
  resolution_sweep/       votes for the clustering-resolution sweep, one file per setting
  paper_clustering/       the published clustering re-voted on the current build
results/                  every table reported in the manuscript
figures/                  the six manuscript figures
```

## Reproducing

Scripts read and write under a `handoff/` working directory; create it first.

```bash
pip install -r requirements.txt
mkdir -p handoff && cp labels/labels_*.json handoff/

# 1. clusters and marker tables (needs network: Scanpy datasets / EBI Atlas)
python scripts/prep_cells.py pbmc
python scripts/prep_panc_ftp.py

# 2. LLM votes (a GGUF model file is required; see note below)
python scripts/run_local_v4.py <model.gguf> <chat_format> <tag> pbmc

# 3. cell-level matrices, then the baseline and the conformal evaluation
python scripts/recompute.py
python scripts/pooling.py
python scripts/mondrian.py
```

Analysis entry points live in `scripts/recompute.py` (the procedure) and
`scripts/baseline.py` (`ref_probs` fits the reference classifier, `build_onto` builds the
ontology, `conf_eval` runs the conformal evaluation).

Notes that matter for reproduction:

- **Scanpy and numba.** Importing Scanpy fails if numba cannot write its cache. Run the
  preparation scripts in a separate process with `NUMBA_DISABLE_JIT=1` and a writable
  `NUMBA_CACHE_DIR`.
- **Cell Ontology.** Terms and ancestor closures are fetched from the EBI OLS4 API at run
  time. Depth is taken along the *longest* path to the root, because the ontology is a
  directed acyclic graph with leaf depths from 3 to 8.
- **Model weights are not included.** Local runs used GGUF quantisations; the
  `chat_format` argument must match the model (`chatml`, `gemma`, `llama-3`, …).
- **Prompt protocol is tissue-dependent.** Differential-expression genes alone are
  sufficient for 10x immune data but not for pancreas, where over-clustering removes the
  dominant type's marker from the DE list. See the manuscript's Methods section;
  `results/protokol_bilgilendiricilik.csv` and
  `results/marker_bilgilendiricilik_teshisi.csv` hold the diagnostic.


## Which script produced which table

Every number in the manuscript comes from the scripts in `scripts/`, run on the vote files in
`votes/`. There is **one** implementation of the procedure: `scripts/recompute.py`. The other
scripts import it (`exec(open("recompute.py").read()...)`) so that the nonconformity score,
the tie-breaking rule, the ontology depth and the resolution metric are shared rather than
re-implemented.

| Manuscript item | Script | Output in `results/` |
|---|---|---|
| Table 3, Figure 2, Figure 3 (grid, tie ablation) | `recompute.py` | `izgara_yeniden.csv`, `aps_kuyruk.csv` |
| Table 4 (calibration regimes) | `pooling.py` | `havuzlama_yeniden.csv` |
| Table 5 (predictor correlations) | `recompute.py` | `ongorucu_kati.csv` |
| Table 6, Figure 4 (class-conditional, Mondrian) | `mondrian.py` | `mondrian_sinif_yeniden.csv`, `mondrian_ozet_yeniden.csv`, `mondrian_esik_uygunluk.csv` |
| Table 7, Figure 5 (baseline, held-out type) | `refcmp.py` | `temel_birlesik.csv`, `yenitip_birlesik.csv` |
| Table 8 (label-space ablation) | `ablation.py` | `ablasyon_birlesik.csv` |
| Table 9 (seed replication) | `mondrian.py` | `tohum_yeniden.csv` |
| Table 10 (vote collapse, top-1) | `recompute.py` | `cokme_kati.csv`, `top1_beraberlik_16hucre.csv` |
| Table 11, Figure 7 (clustering resolution sweep) | `prep_res.py` → `run_local_v4r.py` → `ressens.py` | `cozunurluk_kumeleme.csv`, `cozunurluk_duyarlilik.csv` |
| Table 12 (build-to-build reproducibility) | `ressens.py` | `ikili_tekrar_kontrolu.csv` |

## Three definitions, and where they live in the code

`scripts/` is the only implementation of the procedure. These are the three choices a reader
is most likely to want to check against the manuscript, with the line that makes each one:

1. **Resolution.** The denominator is the depth of *that unit's own true label*, not the
   maximum depth in the candidate space: `recompute.py` line with `lcad(...)/D[t]`, where `D`
   is a per-label depth array. `refcmp.py` and `ablation.py` use `D[di]` for the held-out
   class, which is the same quantity.
2. **Tie-breaking.** `aps_keyed` orders labels with
   `np.lexsort((-key, -P))`, so equal probabilities are resolved by `key`. With
   `key = onto_key(P, lcl, lcad)` this is the ontology-aware rule reported in the paper; with
   a uniform random key it is the random-tie baseline also reported. Plain `np.argsort(-P)`
   appears nowhere in the reporting path.
3. **Ontology depth and LCA on a DAG.** The Cell Ontology is a directed acyclic graph, so a
   term can have several parents. `baseline.py:build_onto` follows *every* parent
   (`for p in par.get(c, [])`), takes depth as the **longest** path to the root
   (`1 + max(depth(p) for p in ps)`), and computes the least common ancestor as the deepest
   member of the intersection of full ancestor closures
   (`set.intersection(*[A[c] for c in cl_ids])`). Where several candidates share that maximal
   depth the first in sorted CL-id order is taken, which is deterministic.

Ancestor closures were retrieved from the EBI OLS4 API in October 2026 and are cached in
`labels/` so that the tables can be reproduced without re-querying the service.

## Data

- PBMC3k — distributed with Scanpy (`sc.datasets.pbmc3k_processed`)
- Pancreas — ArrayExpress/BioStudies accession **E-MTAB-5061**, via the EBI Single Cell
  Expression Atlas
- Cell Ontology — EBI OLS4 API

Raw votes in `votes/` let every table in `results/` be recomputed without re-running any
model. This matters for more than convenience: decoder sampling at temperature 1.0 is
reproducible only within one llama.cpp build, and `results/ikili_tekrar_kontrolu.csv` measures
how far one configuration moves between two builds (coverage is reproduced; set size,
resolution, top-1 accuracy and the vote collapse rate move more). The released votes, not the
binary, are what make these tables re-derivable.

The resolution sweep in `votes/resolution_sweep/` was produced entirely within one build and is
read within itself; `votes/paper_clustering/gemma2-2b_pbmc_PAPER.json` is the published
clustering re-voted on that same build, and is the paired measurement behind the
reproducibility table.

One scope note on the sweep. `prep_res.py` rebuilds the embedding and the nearest-neighbour
graph from the stored highly-variable-gene matrix, while `prep_cells.py` — which produced the
clustering used everywhere else — leidens the graph that ships with the processed PBMC3k object
and, for the pancreas, a 40-component decomposition. The resolution parameter agrees at 3.0 but
the partition does not: 37 units at mean purity 0.934 against 36 at 0.953 on PBMC3k, and 35 at
0.992 against 35 at 0.991 on the pancreas. Resolution 3.0 in `cozunurluk_duyarlilik.csv` is a
second clustering at the same parameter value, not the published one.

## Not included

- A packaged command-line tool. This repository reproduces the study; it is not yet an
  installable annotator.
- A free-text to Cell Ontology projection layer. All runs use a closed 14-label space.
- CellTypist's own pretrained model and SingleR. The baseline here is an L2-regularised
  logistic regression — CellTypist's model class — trained on an in-dataset reference,
  which is an upper bound on what those tools could achieve on these data.

## Citation

A manuscript describing this work is in preparation; this section will be updated with the
reference once it is available.

Repository: <https://github.com/aysegulsayineker/conformal-cell-annotation>

## License

MIT, see `LICENSE`.
