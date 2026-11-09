"""LSTM remaining-useful-life model = the twin's prediction layer."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from data import (load_raw, load_rul, to_fixed, MODEL_IDX, SCALE, WINDOW, RUL_CAP, make_windows)

ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(2)


class RulLSTM(nn.Module):
    def __init__(self, n_in=14, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1]).squeeze(-1)


def engine_arrays(df):
    unit, cycle, vals = to_fixed(df)
    out = {}
    for u in np.unique(unit):
        m = unit == u
        out[int(u)] = vals[m][:, MODEL_IDX].astype(np.float64) / SCALE
    return out


def fit_scaler(train_eng):
    allv = np.concatenate(list(train_eng.values()))
    return allv.min(0), allv.max(0)


def norm(x, lo, hi):
    return ((x - lo) / (hi - lo + 1e-9)).astype(np.float32)


def train_windows(eng, lo, hi, units):
    X, y = [], []
    for u in units:
        v = eng[u]
        n = len(v)
        w = make_windows(norm(v, lo, hi))
        rul = np.minimum(n - np.arange(WINDOW, n + 1), RUL_CAP)      # RUL at the last cycle of each window
        X.append(w); y.append(rul / RUL_CAP)
    return torch.tensor(np.concatenate(X)), torch.tensor(np.concatenate(y), dtype=torch.float32)


def predict_rul(model, windows_norm):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(windows_norm)).numpy() * RUL_CAP


def train_one(seed, train_eng, lo, hi, epochs=40, patience=8, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    units = np.array(sorted(train_eng))
    rng = np.random.RandomState(seed)
    val_units = rng.choice(units, 20, replace=False)
    tr_units = [u for u in units if u not in set(val_units)]
    Xtr, ytr = train_windows(train_eng, lo, hi, tr_units)
    Xva, yva = train_windows(train_eng, lo, hi, val_units)
    model = RulLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        t0 = time.time()
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), yva).item()
        log(f"seed {seed} ep {ep+1:02d} val_rmse={np.sqrt(vl)*RUL_CAP:.2f} ({time.time()-t0:.1f}s)")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def last_windows(test_eng, lo, hi):
    return np.stack([norm(test_eng[u][-WINDOW:], lo, hi) for u in sorted(test_eng)])


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
    tr, te, rul = load_raw("train"), load_raw("test"), load_rul()
    tr_eng, te_eng = engine_arrays(tr), engine_arrays(te)
    lo, hi = fit_scaler(tr_eng)
    np.savez(ROOT / "results" / "scaler.npz", lo=lo, hi=hi)
    (ROOT / "results" / "models").mkdir(exist_ok=True)
    Xte = last_windows(te_eng, lo, hi)
    metrics = {}
    for s in seeds:
        m = train_one(s, tr_eng, lo, hi, log=lambda x: print(x, flush=True))
        torch.save(m.state_dict(), ROOT / "results" / "models" / f"lstm_seed{s}.pt")
        pred = predict_rul(m, Xte)
        rmse = float(np.sqrt(np.mean((pred - rul) ** 2)))
        mae = float(np.mean(np.abs(pred - rul)))
        d = pred - rul
        score = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))   # NASA PHM08 asymmetric score
        metrics[s] = {"rmse": rmse, "mae": mae, "phm_score": score}
        print(f"=== seed {s}: test RMSE {rmse:.2f}  MAE {mae:.2f}  PHM score {score:.0f}", flush=True)
        (ROOT / "results" / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
