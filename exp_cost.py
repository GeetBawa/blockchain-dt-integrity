"""E2 gas cost, E3 latency/overhead, E4 provenance queries & lineage, E7 storage -- per batch size B.
Run on an otherwise idle machine (timings are wall-clock)."""
import json, sys, time, struct
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from eth_utils import keccak
from chain import Chain
from data import MODEL_IDX, SCALE, WINDOW, leaf_hash, DATA
from model import RulLSTM, norm, predict_rul
from merkle import verify as merkle_verify
from pipeline import *

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "results" / "models" / "lstm_seed0.pt"
CONFIG = json.dumps({"arch": "LSTM-2x64+MLP32", "window": WINDOW, "rul_cap": 125, "sensors": 14, "opt": "adam-1e-3"}, sort_keys=True)


def file_hash(path):
    return keccak(Path(path).read_bytes())


def digest_pred(x):
    return keccak(str(round(float(x), 3)).encode())


def load_model(path=MODEL_FILE):
    m = RulLSTM(); m.load_state_dict(torch.load(path)); m.eval(); return m


def window_values(store, u):
    order = np.argsort(store.cycles[u], kind="stable")
    v = store.values[u][order][:, MODEL_IDX].astype(np.float64) / SCALE
    return v[-WINDOW:]


_TRAIN_HASH = None


def read_anchors_storage(reg, ing):
    """Verifier's view read from contract STORAGE (no log scan): k-th anchor of a twin = k-th batch of the public batching rule."""
    meta = {}
    for u in sorted(ing.ranges):
        ids = reg.functions.twinAnchorIds(int(u)).call()
        meta[u] = [(s, e, bytes(reg.functions.anchors(i).call()[0]), i) for (s, e), i in zip(ing.ranges[u], ids)]
    return meta


def trace_decision(reg, did, store, ing, model, scaler, model_path=MODEL_FILE):
    """Full provenance trace of one twin decision. Returns dict of check results (storage-index path)."""
    global _TRAIN_HASH
    if _TRAIN_HASH is None:
        _TRAIN_HASH = file_hash(DATA / "train_FD001.txt")      # hash of the archived training data, computed once
    inp, outp, ts, twin, mid = reg.functions.decisions(did).call()
    mh, dh, ch_, registrant, _ = reg.functions.models(mid).call()
    chk = {}
    chk["model_hash_ok"] = (file_hash(model_path) == bytes(mh))
    chk["train_data_hash_ok"] = (_TRAIN_HASH == bytes(dh))
    n = len(store.cycles[twin])
    lo_c = n - WINDOW + 1
    B_ = ing.B
    ids = reg.functions.twinAnchorIds(int(twin)).call()
    cover = []
    for j in range((lo_c - 1) // B_, (n - 1) // B_ + 1):
        cover.append((j * B_ + 1, min((j + 1) * B_, n), bytes(reg.functions.anchors(ids[j]).call()[0])))
    chk["input_digest_ok"] = (keccak(b"".join(r for _, _, r in cover)) == bytes(inp))
    # inclusion proofs of every record of the window against the on-chain roots
    ok = True
    cyc, val = store.cycles[twin], store.values[twin]
    for i in range(len(cyc)):
        c = int(cyc[i])
        if c < lo_c:
            continue
        root = next((r for s, e, r in cover if s <= c <= e), None)
        p = ing.proofs.get((twin, c))
        if root is None or p is None or not merkle_verify(root, leaf_hash(twin, cyc[i], val[i]), p):
            ok = False
    chk["window_records_ok"] = ok
    x = norm(window_values(store, twin), *scaler)[None]
    pred = float(predict_rul(model, x)[0])
    chk["output_digest_ok"] = (digest_pred(pred) == bytes(outp))
    chk["all_ok"] = all(chk.values())
    return chk


def run_B(B, scaler):
    store = load_store("test")
    ing = ingest(store, B)
    n_rec = store.n_records()
    row = dict(B=B, n_records=n_rec, n_batches=ing.n_batches, offchain_build_s=ing.build_seconds,
               offchain_us_per_record=1e6 * ing.build_seconds / n_rec)
    # ------------------------------------------------ FULL contract
    ch = Chain()
    reg, dep_gas = ch.deploy("TwinProvenanceRegistry")
    ch.send(reg.functions.setGateway(ch.gateway, True), ch.admin)
    t0 = time.perf_counter()
    costs = anchor_all(ch, reg, ing, ch.gateway)
    row["full_total_anchor_wall_s"] = time.perf_counter() - t0
    g = np.array([c[0] for c in costs]); t = np.array([c[1] for c in costs])
    row.update(full_deploy_gas=dep_gas, full_gas_first=int(g[0]), full_gas_mean=float(g.mean()), full_gas_median=float(np.median(g)),
               full_gas_total=int(g.sum()), full_gas_per_record=float(g.sum() / n_rec),
               full_tx_ms_mean=1000 * float(t.mean()), full_tx_ms_p95=1000 * float(np.percentile(t, 95)))
    row["chain_blocks"] = ch.block_number()
    # ---- reading the trusted view + verifying the whole historian
    t0 = time.perf_counter(); meta = read_anchors_storage(reg, ing); row["read_all_anchors_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); res = verify_store(store, meta, ing.proofs); row["verify_full_store_s"] = time.perf_counter() - t0
    row["verify_flags_on_clean"] = int(sum(int(f.sum()) for f, _, _, _ in res.values()))
    row["verify_us_per_record"] = 1e6 * row["verify_full_store_s"] / n_rec
    # ---- single-record proof checks (off-chain and on-chain eth_call)
    keys = list(ing.proofs.keys())[:: max(1, len(ing.proofs) // 200)]
    t0 = time.perf_counter()
    for (u, c) in keys:
        i = int(np.where(store.cycles[u] == c)[0][0])
        merkle_verify(ing.roots[(u, ((c - 1) // B) * B + 1)], leaf_hash(u, c, store.values[u][i]), ing.proofs[(u, c)])
    row["proof_verify_offchain_us"] = 1e6 * (time.perf_counter() - t0) / len(keys)
    t0 = time.perf_counter(); gas_v = []
    anchors_id = {(u, s): a for u, lst in meta.items() for s, e, r, a in lst}
    for (u, c) in keys[:100]:
        i = int(np.where(store.cycles[u] == c)[0][0])
        aid = anchors_id[(u, ((c - 1) // B) * B + 1)]
        fn = reg.functions.verifyAnchored(aid, leaf_hash(u, c, store.values[u][i]), ing.proofs[(u, c)])
        assert fn.call()
        gas_v.append(fn.estimate_gas())
    row["proof_verify_onchain_ms"] = 1000 * (time.perf_counter() - t0) / 100
    row["proof_verify_onchain_gas_mean"] = float(np.mean(gas_v))
    row["proof_len_mean"] = float(np.mean([len(ing.proofs[k]) for k in keys]))
    # ---- model + decisions (lineage)
    model = load_model()
    Xte = np.stack([norm(window_values(store, u), *scaler) for u in sorted(store.cycles)])
    # same single-window inference path as trace_decision (batched float32 inference differs in the last bits and
    # can flip a rounded digest)
    preds = np.array([float(predict_rul(model, Xte[i:i + 1])[0]) for i in range(len(Xte))])
    r, _ = ch.send(reg.functions.registerModel(file_hash(MODEL_FILE), file_hash(DATA / "train_FD001.txt"), keccak(CONFIG.encode())), ch.gateway)
    row["register_model_gas"] = r.gasUsed
    dgas = []
    for k, u in enumerate(sorted(store.cycles)):
        n = len(store.cycles[u]); lo_c = n - WINDOW + 1
        roots = [ing.roots[(u, s)] for (s, e) in ing.ranges[u] if e >= lo_c]
        r, _ = ch.send(reg.functions.recordDecision(int(u), 0, keccak(b"".join(roots)), digest_pred(preds[k])), ch.gateway)
        dgas.append(r.gasUsed)
    row["record_decision_gas_mean"] = float(np.mean(dgas))
    row["chain_blocks_final"] = ch.block_number()
    # ---- E4 provenance queries vs chain length
    u0 = 50
    t0 = time.perf_counter()
    ids = reg.functions.twinAnchorIds(u0).call(); [reg.functions.anchors(i).call() for i in ids]
    row["query_storage_index_ms"] = 1000 * (time.perf_counter() - t0)
    if ch.block_number() <= 800:       # eth-tester's log filter scales ~quadratically; not feasible on the longer chains
        t0 = time.perf_counter(); reg.events.Anchored().get_logs(from_block=0, argument_filters={"twinId": u0})
        row["query_log_scan_ms"] = 1000 * (time.perf_counter() - t0)
    else:
        row["query_log_scan_ms"] = float("nan")
    t0 = time.perf_counter(); head = reg.functions.twinHead(u0).call()
    row["query_head_ms"] = 1000 * (time.perf_counter() - t0)
    # hash-chain completeness check for one twin: recompute head from the ordered roots
    h = b"\x00" * 32
    for (s, e) in ing.ranges[u0]:
        h = keccak(h + ing.roots[(u0, s)])
    row["twin_head_matches"] = bool(h == bytes(head))
    lat, allok = [], []
    for did in range(100):
        t0 = time.perf_counter(); c = trace_decision(reg, did, store, ing, model, scaler); lat.append(time.perf_counter() - t0); allok.append(c["all_ok"])
    row["trace_decision_ms_mean"] = 1000 * float(np.mean(lat)); row["trace_decision_ms_p95"] = 1000 * float(np.percentile(lat, 95))
    row["trace_decision_all_ok"] = float(np.mean(allok))
    # ---- tamper tests on provenance (done once, on the B=20 chain)
    if B == 20:
        out = {}
        # T1: one flipped weight in the model file the twin is *using*
        sd = torch.load(MODEL_FILE); k0 = next(iter(sd)); sd[k0] = sd[k0].clone(); sd[k0].view(-1)[0] += 1e-3
        bad_path = ROOT / "results" / "models" / "tampered_tmp.pt"; torch.save(sd, bad_path)
        bm = load_model(bad_path)
        det = [not trace_decision(reg, d, store, ing, bm, scaler, model_path=bad_path)["model_hash_ok"] for d in range(100)]
        out["T1_model_weight_tamper_detected"] = float(np.mean(det)); bad_path.unlink()
        # T2: stored sensor value in a decision's window altered afterwards
        rng = np.random.default_rng(11); det = []
        for d in range(100):
            u = sorted(store.cycles)[d]; s2 = store.copy()
            i = len(s2.cycles[u]) - 1 - int(rng.integers(0, WINDOW)); ch_ = MODEL_IDX[int(rng.integers(0, 14))]
            s2.values[u][i, ch_] += 5
            det.append(not trace_decision(reg, d, s2, ing, model, scaler)["window_records_ok"])
        out["T2_input_data_tamper_detected"] = float(np.mean(det))
        # T3: claimed output differs from what was recorded (decision record vs reported prediction)
        det = []
        for d in range(100):
            u = sorted(store.cycles)[d]
            claimed = digest_pred(preds[d] + 5.0)
            det.append(claimed != bytes(reg.functions.decisions(d).call()[1]))
        out["T3_output_tamper_detected"] = float(np.mean(det))
        (ROOT / "results" / "e4_provenance_tamper.json").write_text(json.dumps(out, indent=2)); print(out)
    # ------------------------------------------------ LEAN contract (gas only)
    ch2 = Chain()
    lean, ldep = ch2.deploy("LeanRegistry")
    lc = anchor_all(ch2, lean, ing, ch2.admin)
    lg = np.array([c[0] for c in lc]); lt = np.array([c[1] for c in lc])
    row.update(lean_deploy_gas=ldep, lean_gas_mean=float(lg.mean()), lean_gas_total=int(lg.sum()), lean_gas_per_record=float(lg.sum() / n_rec),
               lean_tx_ms_mean=1000 * float(lt.mean()))
    # ---- analytic columns (derived from contract layout / batching rule, not measured)
    row["storage_slots_full_per_anchor"] = 6    # anchors(2) + anchors.length + twinAnchors elem + length + twinHead
    row["storage_slots_lean_per_anchor"] = 1
    row["onchain_bytes_full_per_record"] = 6 * 32 / B
    row["onchain_bytes_lean_per_record"] = 32 / B
    row["raw_bytes_per_record"] = 8 + 24 * 4
    row["anchor_delay_worst_cycles"] = B - 1
    row["anchor_delay_mean_cycles"] = (B - 1) / 2
    return row


if __name__ == "__main__":
    Bs = [int(x) for x in sys.argv[1:]] or [100, 50, 20, 10, 5, 1]
    sc = np.load(ROOT / "results" / "scaler.npz"); scaler = (sc["lo"], sc["hi"])
    out_csv = ROOT / "results" / "e2_e3_e4_cost.csv"
    for B in Bs:
        t0 = time.time(); row = run_B(B, scaler)
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"B={B} done in {time.time()-t0:.0f}s: full gas/anchor {row['full_gas_mean']:.0f}, per record {row['full_gas_per_record']:.0f}", flush=True)
