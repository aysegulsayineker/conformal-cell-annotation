"""run_local_v4.py ile BIREBIR ayni istem ve kisitlama; yalniz kume dosyasi parametrik.
Fark yok: ayni SYS metni, ayni prompt() (her iki veri setinde de top_expressed),
ayni response_format JSON semasi, ayni ornekleme ayarlari."""
import json, sys, time, collections, os, re
import pandas as pd
from llama_cpp import Llama
model_path, chat_format, tag, ds, clpath, outpath = sys.argv[1:7]
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 1234
K = 10
TISSUE = {"pbmc": "PBMC", "panc": "pancreas"}[ds]
cl = pd.read_csv(clpath)
LABELS = json.load(open({"pbmc": "handoff/labels_pbmc.json", "panc": "handoff/labels_panc.json"}[ds]))
SYS = (f"You are an expert in single-cell transcriptomics annotating human {TISSUE} clusters. "
       "Choose exactly one label from the provided list. Answer with the label text only, nothing else.")
def prompt(de, te):
    return (f"Human {TISSUE} scRNA-seq cluster.\n"
            f"Top differentially expressed marker genes (ranked):\n{de}\n\n"
            f"Most highly expressed genes in this cluster (ranked; mitochondrial and ribosomal removed):\n{te}\n\n"
            "Allowed labels:\n" + "\n".join(f"- {l}" for l in LABELS) + "\n\nLabel:")
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS}},
          "required": ["label"], "additionalProperties": False}
llm = Llama(model_path=model_path, n_ctx=3072, n_gpu_layers=0, n_threads=6,
            chat_format=chat_format, verbose=False, seed=SEED)
out, t0, bad = [], time.time(), 0
for r in cl.itertuples():
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": prompt(r.de_markers, r.top_expressed)}]
    c = collections.Counter()
    for _ in range(K):
        try:
            res = llm.create_chat_completion(messages=msgs, temperature=1.0, top_p=0.95,
                    max_tokens=32, response_format={"type": "json_object", "schema": SCHEMA})
            lab = json.loads(res["choices"][0]["message"]["content"]).get("label")
            if lab in LABELS: c[lab] += 1
            else: bad += 1
        except Exception: bad += 1
    out.append({"cluster": str(r.cluster), "counts": dict(c)})
meta = json.load(open(outpath + ".meta")) if os.path.exists(outpath + ".meta") else {}
m = re.search(r"_r([0-9.]+)\.csv$", clpath)          # cozunurluk kume dosyasi adindan
if m: meta.setdefault("resolution", float(m.group(1)))
json.dump({"model": tag, "dataset": ds, "K": K, "bad": bad, "seconds": round(time.time() - t0, 1),
           "clusters": out, **meta}, open(outpath, "w"))
print(f"BITTI {ds}/{tag} {clpath} bad={bad} sure={time.time()-t0:.0f}s", flush=True)
