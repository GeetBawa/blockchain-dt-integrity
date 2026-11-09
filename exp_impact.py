"""E5 impact of tampering on the twin's remaining-life accuracy, with and without verification."""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from data import MODEL_IDX, SCALE, WINDOW, load_rul, load_raw, to_fixed
from model import RulLSTM, norm, predict_rul
from pipeline import *
from exp_detect import setup_chain, B

ROOT = Path(__file__).resolve().parent.parent
FRAC = 0.30          # fraction of engines attacked
N_SEEDS = 10
N_MODELS = 5


def healthy_direction():
    """Per model sensor: direction of degradation and its extent (mean of last 30 cycles minus first 30) on training data."""
    tr = load_store("train")
    first, last = [], []
    for u in tr.cycles:
        x = tr.values[u][:, MODEL_IDX].astype(np.float64)
        first.append(x[:30].mean(0)); last.append(x[-30:].mean(0))
    return np.mean(last, 0) - np.mean(first, 0)          # signed degradation extent, raw fixed-point units


def attack_final_window(store, kind, engines, rng, det, ext):
    s = store.copy(); tm = {u: np.zeros(len(c), bool) for u, c in s.cycles.items()}
    for u in engines:
        n = len(s.cycles[u]); lo = n - WINDOW
        if kind == "replay_healthy":
            old = s.values[u][lo:].copy(); s.values[u][lo:] = s.values[u][:WINDOW]
            tm[u][lo:] = (old != s.values[u][lo:]).any(axis=1)
        elif kind == "drift_mask":
            ramp = (np.arange(1, WINDOW + 1) / WINDOW)[:, None]
            shift = -0.5 * ext[None, :] * ramp                        # hide half of the degradation, ramped in
            old = s.values[u][lo:].copy()
            s.values[u][lo:, MODEL_IDX] = np.rint(old[:, MODEL_IDX] + shift).astype(np.int32)
            tm[u][lo:] = (old != s.values[u][lo:]).any(axis=1)
        elif kind == "random_edits":
            k = max(1, int(round(0.2 * WINDOW)))
            for j in rng.choice(np.arange(lo, n), k, replace=False):
                pos = rng.choice(14, size=int(rng.integers(1, 4)), replace=False); ch = np.array(MODEL_IDX)[pos]
                sign = rng.choice([-1, 1], size=len(ch)); old = s.values[u][j].copy()
                s.values[u][j, ch] = np.rint(old[ch] + sign * 3.0 * det.sd[pos] * SCALE).astype(np.int32)
                tm[u][j] = bool((s.values[u][j] != old).any())
    return s, tm


def predict_with_policy(model, scaler, store, u, flagged):
    """Latest 30-record window that is entirely unflagged; RUL(t') minus elapsed cycles. Returns (pred, used_fallback, no_window)."""
    v = store.values[u][:, MODEL_IDX].astype(np.float64) / SCALE
    n = len(v)
    ok = ~flagged
    for t in range(n - 1, WINDOW - 2, -1):
        if ok[t - WINDOW + 1: t + 1].all():
            p = float(predict_rul(model, norm(v[t - WINDOW + 1: t + 1], *scaler)[None])[0])
            return max(p - (n - 1 - t), 0.0), t != n - 1, False
    return 125.0, True, True


def main():
    torch.set_num_threads(1)
    rul = load_rul()
    sc = np.load(ROOT / "results" / "scaler.npz"); scaler = (sc["lo"], sc["hi"])
    models = []
    for k in range(N_MODELS):
        m = RulLSTM(); m.load_state_dict(torch.load(ROOT / "results" / "models" / f"lstm_seed{k}.pt")); m.eval(); models.append(m)
    store = load_store("test"); units = sorted(store.cycles)
    ch, reg, ing, meta, _ = setup_chain(store)
    det = AnomalyDetector(); ext = healthy_direction()
    rows = []

    def evaluate(tag, attack, seed, s_eval, tm, attacked, flags_led, flags_stat):
        for mi, model in enumerate(models):
            for policy, fl in [("none", None), ("ledger", flags_led), ("stat", flags_stat)]:
                preds, fb, nw = [], 0, 0
                for k, u in enumerate(units):
                    if fl is None:
                        p = float(predict_rul(model, norm(s_eval.values[u][:, MODEL_IDX][-WINDOW:].astype(np.float64) / SCALE, *scaler)[None])[0])
                    else:
                        p, used, none_ = predict_with_policy(model, scaler, s_eval, u, fl[u]); fb += used; nw += none_
                    preds.append(p)
                preds = np.array(preds); err = preds - rul
                att = np.array([u in attacked for u in units])
                rows.append(dict(attack=attack, seed=seed, model=mi, policy=policy, rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                                 rmse_attacked=float(np.sqrt(np.mean(err[att] ** 2))) if att.any() else np.nan,
                                 mean_signed_err_attacked=float(err[att].mean()) if att.any() else np.nan,
                                 fallback_engines=fb, no_window_engines=nw))

    # clean reference
    allv = {u: [] for u in units}
    flags_clean_led = {u: v[0] for u, v in verify_store(store, meta, ing.proofs).items()}
    flags_clean_stat = det.flag_store(store)[0]
    evaluate("clean", "clean", 0, store, None, set(), flags_clean_led, flags_clean_stat)
    for kind in ["replay_healthy", "drift_mask", "random_edits"]:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(2000 + seed)
            attacked = set(rng.choice(units, int(FRAC * len(units)), replace=False).tolist())
            s2, tm = attack_final_window(store, kind, attacked, rng, det, ext)
            fl_led = {u: v[0] for u, v in verify_store(s2, meta, ing.proofs).items()}
            fl_stat = det.flag_store(s2)[0]
            # detection among attacked engines (>=1 truly tampered record flagged in the last window)
            evaluate(kind, kind, seed, s2, tm, attacked, fl_led, fl_stat)
            det_led = np.mean([fl_led[u][-WINDOW:][tm[u][-WINDOW:]].any() for u in attacked if tm[u].any()])
            det_stat = np.mean([fl_stat[u][-WINDOW:][tm[u][-WINDOW:]].any() for u in attacked if tm[u].any()])
            rows.append(dict(attack=kind, seed=seed, model=-1, policy="engine_detection", rmse=np.nan, mae=np.nan, rmse_attacked=np.nan,
                             mean_signed_err_attacked=np.nan, fallback_engines=det_led, no_window_engines=det_stat))
        print("E5", kind, "done", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "e5_impact.csv", index=False)


if __name__ == "__main__":
    main()
