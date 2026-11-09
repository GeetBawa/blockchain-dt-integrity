"""FD001 loading, fixed-point serialisation and record hashing."""
import struct
from pathlib import Path
import numpy as np
import pandas as pd
from eth_utils import keccak

DATA = Path(__file__).resolve().parent.parent / "data"
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
VALUE_COLS = COLS[2:]                                   # 24 measured channels
SCALE = 10_000                                          # 4-decimal fixed point (all FD001 values have <=4 dp)
# 14 informative sensors (the 7 constant channels + near-constant settings are dropped for the twin model)
MODEL_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
MODEL_IDX = [VALUE_COLS.index(c) for c in MODEL_SENSORS]
WINDOW = 30
RUL_CAP = 125


def load_raw(name):
    return pd.read_csv(DATA / f"{name}_FD001.txt", sep=r"\s+", header=None, names=COLS)


def load_rul():
    return pd.read_csv(DATA / "RUL_FD001.txt", sep=r"\s+", header=None, names=["rul"]).rul.values


def to_fixed(df):
    """DataFrame -> (unit, cycle, values[int32, N x 24]); exact because FD001 has <=4 decimals."""
    vals = np.rint(df[VALUE_COLS].values * SCALE).astype(np.int32)
    return df.unit.values.astype(np.int32), df.cycle.values.astype(np.int32), vals


def serialise(unit, cycle, row):
    """Canonical byte encoding of one record: big-endian uint32 unit, uint32 cycle, 24 x int32 value."""
    return struct.pack(">II24i", int(unit), int(cycle), *[int(v) for v in row])


def leaf_hash(unit, cycle, row) -> bytes:
    return keccak(serialise(unit, cycle, row))


def make_windows(values_float, window=WINDOW):
    """All sliding windows of one engine, shape (n-window+1, window, 14)."""
    n = len(values_float)
    return np.stack([values_float[i:i + window] for i in range(n - window + 1)])
