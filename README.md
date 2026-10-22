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
| Validity | holds in all 45 model configurations under **within-dataset** calibration; cross-dataset and pooled calibration under-cover on the pancreas at α = 0.20 (0.743 and 0.771 against a nominal 0.80) |
| Resolution | varies more than fourfold across models (23% – 100% at α = 0.10) |
| Structural limit | α ≤ 0.05 needs n_cal ≥ ⌈1/α⌉ − 1 calibration units; pooling across datasets unlocks it |
| Class-conditional | group-wise (Mondrian) thresholds relocate the violation rather than removing it |
| Baseline | a reference-based classifier is far more informative *when* an in-tissue reference exists, and loses validity when no ancestor of the queried type is in the reference |

## Layout

```
src/
  conformal.py            APS scores, split-conformal grid, ontology depth and LCA
  prep_clusters.py        over-clustering + marker tables (both datasets)
  prep_cells.py           cell-level matrices for the reference-classifier baseline
  run_local_llm.py        grammar-constrained voting, local GGUF models
  run_local_llm_seeded.py same, with an explicit sampling seed
  baseline.py             reference classifier, novel-type experiment, conformal evaluation
scripts/
  run_api_haiku_pbmc.py   API model run (one missing cell of the grid)
labels/                   the 14-label spaces and their Cell Ontology mappings
votes/                    raw per-cluster vote counts for every run (K = 10)
results/                  every table reported in the manuscript
figures/                  the four manuscript figures
```

## Reproducing

Scripts read and write under a `handoff/` working directory; create it first.

```bash
pip install -r requirements.txt
mkdir -p handoff && cp labels/labels_*.json handoff/

# 1. clusters and marker tables (needs network: Scanpy datasets / EBI Atlas)
python src/prep_clusters.py pbmc
python src/prep_clusters.py panc

# 2. LLM votes (a GGUF model file is required; see note below)
python src/run_local_llm.py <model.gguf> <chat_format> <tag> pbmc

# 3. cell-level matrices, then the baseline and the conformal evaluation
python src/prep_cells.py pbmc
python src/prep_cells.py panc
```

Analysis entry points live in `src/baseline.py`: `ref_probs` fits the reference
classifier, `conf_eval` runs the conformal evaluation, and `novel_eval` runs the
held-out-type experiment. `src/conformal.py` holds the scoring and ontology helpers.

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
  dominant type's marker from the DE list. See the manuscript's Implementation section;
  `results/protokol_bilgilendiricilik.csv` and
  `results/marker_bilgilendiricilik_teshisi.csv` hold the diagnostic.

## Data

- PBMC3k — distributed with Scanpy (`sc.datasets.pbmc3k_processed`)
- Pancreas — ArrayExpress/BioStudies accession **E-MTAB-5061**, via the EBI Single Cell
  Expression Atlas
- Cell Ontology — EBI OLS4 API

Raw votes in `votes/` let every table in `results/` be recomputed without re-running any
model.

## Not included

- A packaged command-line tool. This repository reproduces the study; it is not yet an
  installable annotator.
- A free-text to Cell Ontology projection layer. All runs use a closed 14-label space.
- CellTypist's own pretrained model and SingleR. The baseline here is an L2-regularised
  logistic regression — CellTypist's model class — trained on an in-dataset reference,
  which is an upper bound on what those tools could achieve on these data.

## Citation

TODO — add the manuscript reference once available.

Repository: <https://github.com/aysegulsayineker/conformal-cell-annotation>

## License

MIT, see `LICENSE`.
