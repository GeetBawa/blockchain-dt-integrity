"""Tables (CSV + markdown) and print-ready figures from the measured results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"
F = Path(__file__).resolve().parent.parent / "figures"
F.mkdir(exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300})


def save(fig, name):
    fig.savefig(F / f"{name}.png", bbox_inches="tight"); plt.close(fig)


def md(df, path, floatfmt=3):
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(df.round(floatfmt).to_markdown(index=False) if hasattr(df, "to_markdown") else df.round(floatfmt).to_string(index=False))


# ------------------------------------------------------------------ Table 1: twin model
mm = json.loads((R / "model_metrics.json").read_text())
t1 = pd.DataFrame(mm).T.reset_index().rename(columns={"index": "seed"})
summ = pd.DataFrame([{"seed": "mean", **t1[["rmse", "mae", "phm_score"]].mean().to_dict()}, {"seed": "sd", **t1[["rmse", "mae", "phm_score"]].std().to_dict()}])
md(pd.concat([t1, summ]), R / "table_twin_model")

# ------------------------------------------------------------------ E1 detection
d = pd.read_csv(R / "e1_detection.csv")
d = d[d.attack != "clean"]
M = {"Merkle-on-chain (proposed)": ("Merkle-on-chain (proposed)", C["blue"], "o", "-"),
     "Statistical + sequence rules (no ledger)": ("Statistical + sequence rules", C["orange"], "s", "--")}
g = d[d.attack == "modify_noise_units"]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for k, (lab, col, mk, ls) in M.items():
    s = g[g.method == k].groupby("param").record_recall.agg(["mean", "std"]).reset_index()
    ax.errorbar(s.param, s["mean"], yerr=s["std"], color=col, marker=mk, ls=ls, lw=1.6, ms=5, capsize=2, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.set_xlabel("Size of the edit (multiples of sensor noise level)"); ax.set_ylabel("Record-level recall"); ax.set_ylim(-0.03, 1.05)
ax.legend(loc="center right", fontsize=8)
save(fig, "fig_detection_vs_edit_size")

rows = []
for att, p in [("modify_1sigma", 0.01), ("delete", 0.01), ("insert", 0.01), ("replay", 0.01)]:
    for k, lab in [("Merkle-on-chain (proposed)", "Merkle-on-chain"), ("Statistical + sequence rules (no ledger)", "Statistical + rules")]:
        x = d[(d.attack == att) & (d.p == p) & (d.method == k)]
        rows.append(dict(attack=att, method=lab, recall=x.record_recall.mean(), recall_sd=x.record_recall.std(),
                         precision=x.record_precision.mean(), fpr=x.record_fpr.mean()))
    for k, lab in [("Central checksum log - outsider", "Central checksum (outsider)"), ("Central checksum log - insider", "Central checksum (insider)")]:
        x = d[(d.attack == att) & (d.p == p) & (d.method == k)]
        rows.append(dict(attack=att, method=lab, recall=x.batch_recall.mean(), recall_sd=x.batch_recall.std(), precision=np.nan, fpr=np.nan))
t_det = pd.DataFrame(rows); md(t_det, R / "table_detection_p1pct")

# ------------------------------------------------------------------ E2-E4 cost
c = pd.read_csv(R / "e2_e3_e4_cost.csv").sort_values("B")
md(c[["B", "n_batches", "full_gas_mean", "full_gas_per_record", "lean_gas_mean", "lean_gas_per_record", "full_tx_ms_mean",
      "offchain_us_per_record", "verify_us_per_record", "proof_verify_offchain_us", "proof_verify_onchain_ms", "proof_verify_onchain_gas_mean",
      "anchor_delay_worst_cycles", "onchain_bytes_full_per_record", "onchain_bytes_lean_per_record"]], R / "table_cost")
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(c.B, c.full_gas_per_record, color=C["blue"], marker="o", lw=1.6, ms=5, label="Full registry")
ax.plot(c.B, c.lean_gas_per_record, color=C["orange"], marker="s", ls="--", lw=1.6, ms=5, label="Lean registry")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(c.B); ax.set_xticklabels(c.B.astype(int))
ax.set_xlabel("Batch size B (records per anchor)"); ax.set_ylabel("Gas per sensor record"); ax.legend(fontsize=8)
save(fig, "fig_gas_vs_batch")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(c.chain_blocks_final, c.query_storage_index_ms, color=C["blue"], marker="o", lw=1.6, ms=5, label="Storage index (per twin)")
ls = c.dropna(subset=["query_log_scan_ms"])
ax.plot(ls.chain_blocks_final, ls.query_log_scan_ms, color=C["orange"], marker="s", ls="--", lw=1.6, ms=5, label="Event-log scan")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Chain length (blocks)"); ax.set_ylabel("Provenance query time (ms)"); ax.legend(fontsize=8)
save(fig, "fig_query_vs_chain_length")

# ------------------------------------------------------------------ E5 impact
if (R / "e5_impact.csv").exists():
    e = pd.read_csv(R / "e5_impact.csv")
    ev = e[e.policy != "engine_detection"]
    clean = ev[ev.attack == "clean"].groupby("policy").rmse.mean()
    names = {"replay_healthy": "Replay of\nhealthy data", "drift_mask": "Degradation\nmasked (drift)", "random_edits": "Random edits\n(3 sigma)"}
    pol = [("none", "No verification", C["orange"]), ("stat", "Statistical detector", C["aqua"]), ("ledger", "Ledger verification", C["blue"])]
    rows = []
    fig, ax = plt.subplots(figsize=(5.2, 3.1)); w = 0.26
    base = clean["none"]
    for j, (pk, pl, col) in enumerate(pol):
        mu, sd = [], []
        for a in names:
            x = ev[(ev.attack == a) & (ev.policy == pk)].rmse
            mu.append(x.mean()); sd.append(x.std()); rows.append(dict(attack=names[a], policy=pl, rmse_mean=x.mean(), rmse_sd=x.std()))
        ax.bar(np.arange(3) + (j - 1) * w, mu, w * 0.92, yerr=sd, color=col, capsize=2, label=pl, error_kw=dict(lw=0.8))
    ax.axhline(base, color=C["muted"], lw=1, ls=":"); ax.text(2.62, base + 0.4, "clean", color=C["muted"], fontsize=8, ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels([names[a] for a in names], fontsize=8)
    ax.set_ylabel("RUL RMSE (cycles), all 100 engines"); ax.legend(fontsize=8, loc="upper right")
    save(fig, "fig_twin_rmse_under_attack")
    t5 = pd.DataFrame(rows); t5.loc[len(t5)] = dict(attack="Clean (no attack)", policy="No verification", rmse_mean=base, rmse_sd=ev[(ev.attack == "clean") & (ev.policy == "none")].rmse.std())
    md(t5, R / "table_twin_rmse_under_attack")
    det = e[e.policy == "engine_detection"].groupby("attack")[["fallback_engines", "no_window_engines"]].mean().rename(
        columns={"fallback_engines": "ledger_engine_detection", "no_window_engines": "stat_engine_detection"}).reset_index()
    md(det, R / "table_engine_detection")
print("report done")
