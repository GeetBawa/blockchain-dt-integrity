"""E1 tamper-detection experiments and E6 security / limitation tests (B = 20, FULL contract, real local chain)."""
import json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from chain import Chain
from data import serialise, MODEL_IDX
from pipeline import *

ROOT = Path(__file__).resolve().parent.parent
B = 20
SEEDS = list(range(10))


def setup_chain(store):
    ch = Chain()
    reg, _ = ch.deploy("TwinProvenanceRegistry")
    ch.send(reg.functions.setGateway(ch.gateway, True), ch.admin)
    ing = ingest(store, B)
    costs = anchor_all(ch, reg, ing, ch.gateway)
    meta = read_onchain_anchors(reg, ch.w3)
    return ch, reg, ing, meta, costs


def sha_table(store, B, n_by_unit):
    """Centralised baseline: per-batch SHA-256 checksum table kept in the same database as the data.
    Batches are defined by cycle range (same layout as the ledger), using the ORIGINAL engine lengths."""
    tab = {}
    for u in store.cycles:
        order = np.argsort(store.cycles[u], kind="stable")
        cyc, val = store.cycles[u][order], store.values[u][order]
        for (s, e) in batch_ranges(n_by_unit[u], B):
            h = hashlib.sha256()
            lo, hi = np.searchsorted(cyc, s, "left"), np.searchsorted(cyc, e, "right")
            for i in range(lo, hi):
                h.update(serialise(u, cyc[i], val[i]))
            tab[(u, s)] = h.digest()
    return tab


def central_flags(store_t, table, B, n_by_unit):
    """Recompute checksums over the (possibly tampered) store and compare to the stored table."""
    now = sha_table(store_t, B, n_by_unit)
    return {k for k in table if now[k] != table[k]}


def run_e1():
    store = load_store("test")
    ch, reg, ing, meta, costs = setup_chain(store)
    det = AnomalyDetector()
    n_by_unit = {u: len(c) for u, c in store.cycles.items()}
    table = sha_table(store, B, n_by_unit)
    rows = []

    def record(attack, param, p, seed, s2, tr):
        r = verify_store(s2, meta, ing.proofs)
        m = metrics_record_level(s2, tr, r)
        tb, fb = truth_batches(tr, s2, B), flagged_batches(r)
        a = anomaly_metrics(det.flag_store(s2), tr)
        # centralised checksum: outsider edits data only; insider also rewrites the checksum table
        out_bad = central_flags(s2, table, B, n_by_unit)
        ins_bad = central_flags(s2, sha_table(s2, B, n_by_unit), B, n_by_unit)   # insider rewrites the table after tampering
        base = dict(attack=attack, param=param, p=p, seed=seed, n_truth_records=m["tp"] + m["fn"], n_truth_batches=len(tb))
        rows.append({**base, "method": "Merkle-on-chain (proposed)", "record_recall": m["recall"], "record_precision": m["precision"],
                     "record_fpr": m["fpr"], "batch_recall": len(tb & fb) / len(tb) if tb else np.nan, "batch_fp": len(fb - tb)})
        rows.append({**base, "method": "Statistical + sequence rules (no ledger)", "record_recall": a["recall"], "record_precision": a["precision"],
                     "record_fpr": a["fpr"], "batch_recall": np.nan, "batch_fp": np.nan})
        rows.append({**base, "method": "Central checksum log - outsider", "record_recall": np.nan, "record_precision": np.nan, "record_fpr": np.nan,
                     "batch_recall": len(tb & out_bad) / len(tb) if tb else np.nan, "batch_fp": len(out_bad - tb)})
        rows.append({**base, "method": "Central checksum log - insider", "record_recall": np.nan, "record_precision": np.nan, "record_fpr": np.nan,
                     "batch_recall": len(tb & ins_bad) / len(tb) if tb else np.nan, "batch_fp": len(ins_bad - tb)})

    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        for m in [0.25, 0.5, 1.0, 2.0, 4.0]:                         # stealth sweep: change size in sensor-noise units
            s2, tr = attack_modify(store, 0.01, m, rng, sd=det.sd)
            record("modify_noise_units", m, 0.01, seed, s2, tr)
        for p in [0.001, 0.01, 0.05]:                                  # intensity sweep
            s2, tr = attack_modify(store, p, 1.0, rng, sd=det.sd); record("modify_1sigma", 1.0, p, seed, s2, tr)
            s2, tr = attack_delete(store, p, rng); record("delete", np.nan, p, seed, s2, tr)
            s2, tr = attack_insert(store, p, rng); record("insert", np.nan, p, seed, s2, tr)
            s2, tr = attack_replay(store, p, rng); record("replay", np.nan, p, seed, s2, tr)
        # clean run (false positives)
        tr0 = Truth({u: np.zeros(len(c), bool) for u, c in store.cycles.items()}, {u: [] for u in store.cycles})
        rr = verify_store(store, meta, ing.proofs); mm = metrics_record_level(store, tr0, rr)
        aa = anomaly_metrics(det.flag_store(store), tr0)
        rows.append(dict(attack="clean", param=np.nan, p=0.0, seed=seed, n_truth_records=0, n_truth_batches=0, method="Merkle-on-chain (proposed)",
                         record_fpr=mm["fpr"], batch_fp=len(flagged_batches(rr))))
        rows.append(dict(attack="clean", param=np.nan, p=0.0, seed=seed, n_truth_records=0, n_truth_batches=0,
                         method="Statistical + sequence rules (no ledger)", record_fpr=aa["fpr"]))
        print("E1 seed", seed, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "e1_detection.csv", index=False)
    return df


def run_e6():
    """Security / limitation tests. Everything is executed against the real contracts."""
    store = load_store("test")
    ch, reg, ing, meta, costs = setup_chain(store)
    det = AnomalyDetector()
    res = {}

    # (a) unauthorised account tries to anchor (e.g. insider without gateway key)
    reverted = 0
    for k in range(20):
        try:
            ch.send(reg.functions.anchor(1, 1, 20, b"\x11" * 32), ch.attacker)
        except Exception:
            reverted += 1
    res["unauthorised_anchor_attempts"] = 20
    res["unauthorised_anchor_reverted"] = reverted

    # (b) insider tampers a batch, recomputes the Merkle root and tries to replace the on-chain root
    rng = np.random.default_rng(7)
    s2, tr = attack_modify(store, 0.01, 1.0, rng, sd=det.sd)
    forged = ingest(s2, B)
    ok_forge = 0
    for u in list(forged.ranges)[:20]:
        s0, e0 = forged.ranges[u][0]
        try:
            ch.send(reg.functions.anchor(int(u), int(s0), int(e0), forged.roots[(u, s0)]), ch.attacker); ok_forge += 1
        except Exception:
            pass
    meta_after = read_onchain_anchors(reg, ch.w3)
    r = verify_store(s2, meta_after, ing.proofs); m = metrics_record_level(s2, tr, r)
    res["forged_reanchor_succeeded"] = ok_forge
    res["tamper_still_detected_recall_after_forge_attempt"] = m["recall"]

    # (c) LEAN contract rejects a second anchor for the same (twin, start)
    lean, _ = ch.deploy("LeanRegistry")
    ch.send(lean.functions.anchor(1, 1, 20, b"\x22" * 32), ch.admin)
    try:
        ch.send(lean.functions.anchor(1, 1, 20, b"\x33" * 32), ch.admin); res["lean_overwrite_reverted"] = False
    except Exception:
        res["lean_overwrite_reverted"] = True

    # (d) compromised *authorised* gateway appends a conflicting anchor in the FULL contract -> verifier detects conflict
    ch.send(reg.functions.anchor(1, 1, 20, b"\x44" * 32), ch.gateway)
    meta2 = read_onchain_anchors(reg, ch.w3)
    conflicts = sum(1 for u, lst in meta2.items() for k in {(s, e) for s, e, _, _ in lst} if len({r_ for s, e, r_, _ in lst if (s, e) == k}) > 1)
    res["conflicting_anchor_ranges_detected"] = conflicts

    # (e) LIMITATION: source-level falsification BEFORE anchoring. The gateway anchors whatever it receives.
    lim = []
    for m_ in [1.0, 2.0, 4.0, 8.0]:
        recs = []
        for seed in range(5):
            rng = np.random.default_rng(500 + seed)
            s_bad, tr = attack_modify(store, 0.01, m_, rng, sd=det.sd)       # data already false when it is ingested
            ing_bad = ingest(s_bad, B)
            meta_bad = {u: [(s, e, ing_bad.roots[(u, s)], 0) for (s, e) in ing_bad.ranges[u]] for u in ing_bad.ranges}
            r = verify_store(s_bad, meta_bad, ing_bad.proofs)
            ledger = metrics_record_level(s_bad, tr, r)["recall"]
            stat = anomaly_metrics(det.flag_store(s_bad), tr)["recall"]
            recs.append((ledger, stat))
        lim.append(dict(magnitude_noise_units=m_, ledger_recall=float(np.mean([x[0] for x in recs])),
                        stat_recall_mean=float(np.mean([x[1] for x in recs])), stat_recall_sd=float(np.std([x[1] for x in recs]))))
    res["source_falsification_limitation"] = lim
    (ROOT / "results" / "e6_security.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("e6", "all"):
        run_e6()
    if which in ("e1", "all"):
        run_e1()
