# Yeni oturumda tek hucrede calistir: Haiku 4.5 / PBMC v2 kosusu
import json, collections, numpy as np, pandas as pd
clu = pd.read_csv("handoff/clusters_pbmc_v2.csv"); labs = json.load(open("handoff/labels_pbmc.json"))
sys_ = ("You are an expert in single-cell transcriptomics annotating human PBMC clusters. "
        "Choose exactly one label from the provided list. Answer with the label text only, nothing else.")
tool = {"name":"annotate","description":"Return the cell type label.",
        "input_schema":{"type":"object","properties":{"label":{"type":"string","enum":labs}},
                        "required":["label"],"additionalProperties":False}}
reqs, keys = [], []
for r in clu.itertuples():
    u = (f"Human PBMC scRNA-seq cluster.\nTop differentially expressed marker genes (ranked):\n{r.de_markers}\n\n"
         f"Most highly expressed genes in this cluster (ranked; mitochondrial and ribosomal removed):\n{r.top_expressed}\n\n"
         "Allowed labels:\n" + "\n".join(f"- {l}" for l in labs) + "\n\nLabel:")
    for _ in range(10):
        reqs.append({"messages":[{"role":"user","content":u}], "system":sys_, "tools":[tool],
                     "tool_choice":{"type":"tool","name":"annotate"}, "temperature":1.0,
                     "max_tokens":64, "model":"claude-haiku-4-5-20251001"})
        keys.append(str(r.cluster))
res = []
for i in range(0, len(reqs), 180):
    res += host.llm(reqs[i:i+180], max_concurrency=8)
cnt = collections.defaultdict(collections.Counter); bad = 0
for k, o in zip(keys, res):
    lab = (o.get("tool_use") or {}).get("input", {}).get("label") if isinstance(o, dict) else None
    if lab in labs: cnt[k][lab] += 1
    else: bad += 1
json.dump({"model":"haiku45","dataset":"pbmc","K":10,"bad":bad,
           "clusters":[{"cluster":c,"counts":dict(cnt[c])} for c in clu.cluster.astype(str)]},
          open("handoff/v2_pbmc_haiku45.json","w"))
print("bad:", bad, "kume:", len(cnt))
