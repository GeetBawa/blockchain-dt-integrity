"""Honest ingestion (batching + Merkle + on-chain anchoring), attacks, verification, anomaly baseline."""
import time
from dataclasses import dataclass
import numpy as np
from data import load_raw, to_fixed, leaf_hash, MODEL_IDX, SCALE, VALUE_COLS
from merkle import build_levels, proofs_of, verify as merkle_verify, root_of


# ------------------------------------------------------------------ store
@dataclass
class Store:
    """The twin's historian. cycles/values are per-engine arrays and may be tampered with."""
    cycles: dict      # unit -> int array [n]
    values: dict      # unit -> int32 array [n, 24]

    def copy(self):
        return Store({u: c.copy() for u, c in self.cycles.items()},
                     {u: v.copy() for u, v in self.values.items()})

    def n_records(self):
        return sum(len(c) for c in self.cycles.values())


def load_store(name="test"):
    unit, cycle, vals = to_fixed(load_raw(name))
    cyc, val = {}, {}
    for u in np.unique(unit):
        m = unit == u
        cyc[int(u)], val[int(u)] = cycle[m].copy(), vals[m].copy()
    return Store(cyc, val)


def batch_ranges(n, B):
    """Cycle ranges (1-based, inclusive) for engine with n records."""
    return [(s, min(s + B - 1, n)) for s in range(1, n + 1, B)]


# ------------------------------------------------------------------ honest ingestion
@dataclass
class Ingest:
    B: int
    ranges: dict          # unit -> [(start,end)]
    roots: dict           # (unit,start) -> bytes32
    proofs: dict          # (unit,cycle) -> proof list (kept off-chain)
    build_seconds: float
    n_batches: int


def ingest(store: Store, B: int) -> Ingest:
    t0 = time.perf_counter()
    ranges, roots, proofs = {}, {}, {}
    nb = 0
    for u in sorted(store.cycles):
        cyc, val = store.cycles[u], store.values[u]
        n = len(cyc)
        ranges[u] = batch_ranges(n, B)
        for (s, e) in ranges[u]:
            leaves = [leaf_hash(u, cyc[i], val[i]) for i in range(s - 1, e)]
            lv = build_levels(leaves)
            roots[(u, s)] = lv[-1][0]
            for k, p in enumerate(proofs_of(lv)):
                proofs[(u, s + k)] = p
            nb += 1
    return Ingest(B, ranges, roots, proofs, time.perf_counter() - t0, nb)


def anchor_all(chain, reg, ing: Ingest, sender, method="anchor"):
    """Anchor every batch root on chain. Returns list of (gas, seconds)."""
    out = []
    for u in sorted(ing.ranges):
        for (s, e) in ing.ranges[u]:
            r, dt = chain.send(getattr(reg.functions, method)(int(u), int(s), int(e), ing.roots[(u, s)]), sender)
            out.append((r.gasUsed, dt))
    return out


def read_onchain_anchors(reg, w3):
    """Read the Anchored events -> {unit: [(start,end,root,id)]} sorted by start. This is the verifier's trusted view."""
    ev = reg.events.Anchored().get_logs(from_block=0)
    meta = {}
    for e in ev:
        a = e["args"]
        meta.setdefault(a["twinId"], []).append((a["startCycle"], a["endCycle"], bytes(a["root"]), a["id"]))
    for u in meta:
        meta[u].sort()
    return meta


# ------------------------------------------------------------------ verification (record level + batch level)
def verify_store(store: Store, meta: dict, proofs: dict):
    """Verify the historian against on-chain roots.
    Returns dict unit -> (flag_mask over store records, missing_cycles list, n_batches_flagged, n_batches)."""
    out = {}
    for u, anchors in meta.items():
        cyc, val = store.cycles.get(u, np.array([], dtype=np.int32)), store.values.get(u)
        order = np.lexsort((np.arange(len(cyc)), cyc)) if len(cyc) else np.array([], dtype=int)
        flag = np.zeros(len(cyc), dtype=bool)
        missing, nbad = [], []
        leaf_of = {}
        for i in order:
            leaf_of.setdefault(int(cyc[i]), []).append(i)
        covered = set()
        for (s, e, root, _id) in anchors:
            idxs = []
            for c in range(s, e + 1):
                covered.add(c)
                if c in leaf_of:
                    idxs.extend(leaf_of[c])
                else:
                    missing.append(c)
            # recompute root exactly as the honest ingestion did: leaves in cycle order
            leaves_ordered = [leaf_hash(u, cyc[i], val[i]) for i in idxs]
            batch_ok = (root_of(leaves_ordered) == root) and len(idxs) == (e - s + 1)
            if not batch_ok:
                nbad.append(s)
            for i in idxs:   # record-level localisation by inclusion proof
                c = int(cyc[i])
                p = proofs.get((u, c))
                if p is None or not merkle_verify(root, leaf_hash(u, cyc[i], val[i]), p):
                    flag[i] = True
        for i in range(len(cyc)):    # records outside any anchored range (insertion beyond the stream)
            if int(cyc[i]) not in covered:
                flag[i] = True
        out[u] = (flag, missing, nbad, len(anchors))
    return out


# ------------------------------------------------------------------ attacks (insider modifies the historian AFTER anchoring)
@dataclass
class Truth:
    tampered: dict    # unit -> bool mask over store records (modified / fabricated)
    deleted: dict     # unit -> list of deleted cycles


def _empty_truth(store):
    return Truth({u: np.zeros(len(c), dtype=bool) for u, c in store.cycles.items()}, {u: [] for u in store.cycles})


def pick_records(store, p, rng):
    """Choose ~p fraction of all records uniformly: returns list of (unit, idx)."""
    flat = [(u, i) for u in sorted(store.cycles) for i in range(len(store.cycles[u]))]
    k = max(1, int(round(p * len(flat))))
    sel = rng.choice(len(flat), k, replace=False)
    return [flat[j] for j in sel]


def attack_modify(store, p, eps, rng, sd=None):
    """Modify 1-3 model sensors of ~p of the records.
    sd=None : relative change, value*(1 +/- eps)
    sd=array: change of +/- eps * sd[sensor] (eps in units of each sensor's own noise level)."""
    s = store.copy()
    t = _empty_truth(s)
    for (u, i) in pick_records(s, p, rng):
        pos = rng.choice(len(MODEL_IDX), size=int(rng.integers(1, 4)), replace=False)
        ch = np.array(MODEL_IDX)[pos]
        sign = rng.choice([-1, 1], size=len(ch))
        old = s.values[u][i].copy()
        if sd is None:
            s.values[u][i, ch] = np.rint(old[ch] * (1 + sign * eps)).astype(np.int32)
        else:
            s.values[u][i, ch] = np.rint(old[ch] + sign * eps * sd[pos] * SCALE).astype(np.int32)
        t.tampered[u][i] = bool((s.values[u][i] != old).any())
    return s, t


def attack_delete(store, p, rng):
    s, t = store.copy(), None
    t = _empty_truth(s)
    per = {}
    for (u, i) in pick_records(s, p, rng):
        per.setdefault(u, []).append(i)
    for u, idx in per.items():
        t.deleted[u] = [int(s.cycles[u][i]) for i in idx]
        keep = np.ones(len(s.cycles[u]), dtype=bool); keep[idx] = False
        s.cycles[u], s.values[u] = s.cycles[u][keep], s.values[u][keep]
        t.tampered[u] = np.zeros(len(s.cycles[u]), dtype=bool)
    return s, t


def attack_insert(store, p, rng):
    """Fabricate records: duplicate an existing cycle id with neighbour-averaged (plausible) values."""
    s = store.copy(); t = _empty_truth(s)
    per = {}
    for (u, i) in pick_records(s, p, rng):
        per.setdefault(u, []).append(i)
    for u, idx in per.items():
        cyc, val = s.cycles[u], s.values[u]
        n = len(cyc)
        new_c, new_v = [], []
        for i in idx:
            j0, j1 = max(i - 1, 0), min(i + 1, n - 1)
            fab = np.rint((val[j0].astype(np.int64) + val[j1]) / 2).astype(np.int32)
            if (fab == val[i]).all():
                fab[MODEL_IDX[0]] += 1
            new_c.append(cyc[i]); new_v.append(fab)
        s.cycles[u] = np.concatenate([cyc, np.array(new_c, dtype=np.int32)])
        s.values[u] = np.vstack([val, np.array(new_v, dtype=np.int32)])
        tm = np.zeros(len(s.cycles[u]), dtype=bool); tm[n:] = True
        t.tampered[u] = tm
    return s, t


def attack_replay(store, p, rng):
    """Replace a record's measurements with an earlier record of the same engine (cycle id kept)."""
    s = store.copy(); t = _empty_truth(s)
    for (u, i) in pick_records(s, p, rng):
        if i < 6:
            continue
        j = i - int(rng.integers(5, min(50, i) + 1))
        old = s.values[u][i].copy()
        s.values[u][i] = s.values[u][max(j, 0)]
        t.tampered[u][i] = bool((s.values[u][i] != old).any())
    return s, t


def metrics_record_level(store_t: Store, truth: Truth, result: dict):
    """Record-level confusion for a verifier output {unit: (flag, missing, nbad, nb)}."""
    tp = fn = fp = tn = 0
    for u, (flag, missing, nbad, nb) in result.items():
        tm = truth.tampered[u]
        tp += int((flag & tm).sum()); fn += int((~flag & tm).sum())
        fp += int((flag & ~tm).sum()); tn += int((~flag & ~tm).sum())
        dm = set(truth.deleted.get(u, []))
        ms = set(missing)
        tp += len(dm & ms); fn += len(dm - ms); fp += len(ms - dm)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, precision=prec, recall=rec, fpr=fp / (fp + tn) if fp + tn else 0.0)


# ------------------------------------------------------------------ statistical anomaly baseline (no ledger)
class AnomalyDetector:
    """Flags a record when any model sensor deviates from the rolling median of the previous 5 records of that
    engine by more than k robust standard deviations (sd estimated on the clean FD001 training engines)."""

    def __init__(self, k=4.0, win=5):
        self.k, self.win = k, win
        tr = load_store("train")
        res = []
        for u in tr.cycles:
            x = tr.values[u][:, MODEL_IDX].astype(np.float64) / SCALE
            res.append(self._residuals(x))
        res = np.concatenate(res)
        mad = np.median(np.abs(res - np.median(res, axis=0)), axis=0)
        self.sd = 1.4826 * mad + 1e-9

    def _residuals(self, x):
        r = np.zeros_like(x)
        for i in range(self.win, len(x)):
            r[i] = x[i] - np.median(x[i - self.win:i], axis=0)
        return r

    def flag_store(self, store: Store):
        """Returns (flags, missing): value test + duplicate-cycle rule per record; gap rule for missing cycles."""
        out, missing = {}, {}
        for u in store.cycles:
            cyc = store.cycles[u]
            order = np.argsort(cyc, kind="stable")
            x = store.values[u][order][:, MODEL_IDX].astype(np.float64) / SCALE
            z = np.abs(self._residuals(x)) / self.sd
            f_sorted = (z > self.k).any(axis=1)
            f = np.zeros(len(order), dtype=bool); f[order] = f_sorted
            vals, counts = np.unique(cyc, return_counts=True)       # duplicate cycle ids
            dup = set(vals[counts > 1].tolist())
            if dup:
                f |= np.array([int(c) in dup for c in cyc])
            out[u] = f
            have = set(cyc.tolist())
            missing[u] = [c for c in range(1, int(cyc.max()) + 1) if c not in have] if len(cyc) else []
        return out, missing


def anomaly_metrics(det, truth: Truth):
    flags, missing = det
    tp = fn = fp = tn = 0
    for u, f in flags.items():
        tm = truth.tampered[u]
        tp += int((f & tm).sum()); fn += int((~f & tm).sum())
        fp += int((f & ~tm).sum()); tn += int((~f & ~tm).sum())
        dm, ms = set(truth.deleted.get(u, [])), set(missing[u])
        tp += len(dm & ms); fn += len(dm - ms); fp += len(ms - dm)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, recall=tp / (tp + fn) if tp + fn else float("nan"),
                precision=tp / (tp + fp) if tp + fp else float("nan"), fpr=fp / (fp + tn) if fp + tn else 0.0)


def truth_batches(truth: Truth, store_t: Store, B: int):
    """Set of (unit, batch_start) that contain at least one tampered or deleted record (ground truth)."""
    out = set()
    for u, tm in truth.tampered.items():
        for c in store_t.cycles[u][tm]:
            out.add((u, ((int(c) - 1) // B) * B + 1))
    for u, dc in truth.deleted.items():
        for c in dc:
            out.add((u, ((int(c) - 1) // B) * B + 1))
    return out


def flagged_batches(result):
    return {(u, s) for u, (_f, _m, nbad, _nb) in result.items() for s in nbad}
