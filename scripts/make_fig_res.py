
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.lines import Line2D
d = pd.read_csv("out/cozunurluk_duyarlilik.csv")
apply_figure_style(sizes=(8,7,6))
COL = {"pbmc":"#1f4e79","panc":"#b45f06"}; NM = {"pbmc":"PBMC3k","panc":"Pancreas"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, a in zip(axes, (0.10, 0.20)):
    s = d[d.alpha == a]
    for ds in ["pbmc", "panc"]:
        g = s[s.veri == ds].sort_values("kume")
        ok = g[g.uygun]; no = g[~g.uygun]
        ax.plot(g.kume, g.cozunurluk, color=COL[ds], lw=0.9, alpha=0.5, zorder=2)
        ax.scatter(ok.kume, ok.cozunurluk, s=30, color=COL[ds], zorder=3)
        ax.scatter(no.kume, no.cozunurluk, s=30, facecolors="white", edgecolors=COL[ds],
                   linewidths=1.0, zorder=3)
        for _, r in g.iterrows():
            ax.annotate(f"{r.leiden:g}", (r.kume, r.cozunurluk), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=6, color=COL[ds])
    ax.axvline(int(np.ceil(1/a)) * 2 - 2, color=META_GREY, ls=":", lw=0.9, zorder=1)
    ax.set_xlabel("Calibration units (clusters)")
    ax.set_title(f"$\\alpha={a:.2f}$")
    ax.margins(0.14)
axes[0].set_ylabel("Resolution (% of true-label depth)")
axes[0].legend(handles=[Line2D([],[],marker="o",ls="",color=COL["pbmc"],label="PBMC3k"),
                        Line2D([],[],marker="o",ls="",color=COL["panc"],label="Pancreas"),
                        Line2D([],[],marker="o",ls="",mfc="white",mec="0.3",label="calibration-size limited")],
               frameon=False, loc="lower right", fontsize=6)
axes[1].text(0.98, 0.03, "dotted line: smallest unit count\nthat admits a non-vacuous threshold",
             transform=axes[1].transAxes, fontsize=6, color=META_GREY, va="bottom", ha="right", linespacing=1.3)
panel_letter(axes[0], "a"); panel_letter(axes[1], "b")
fig.suptitle("Informativeness against calibration budget across Leiden resolutions (Gemma 2 2B)",
             fontsize=8, y=1.02)
fig.tight_layout(); fig.savefig("fig_resolution.png", dpi=300, bbox_inches="tight")
print("kaydedildi")
