# -*- coding: utf-8 -*-
# main/ml_models.py
import os, glob, threading, collections
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

__all__ = [
    "available_tasks", "get_model", "predict", "preprocess_12_features",
    "MyNN1", "MyNN2", "MyNN3", "PositiveNN1", "PositiveNN2", "PositiveNN3",
    "predict_all_models"
]

# ---------- weights root (هر دو نام پوشه را امتحان می‌کنیم) ----------
_BASE = os.path.dirname(__file__)
CANDIDATE_ROOTS = [
    os.path.join(_BASE, "weights", "final_weights_31_7_2025"),
    os.path.join(_BASE, "weights", "final weights 31_7_2025"),
    os.path.join(_BASE, "weights", "final weights 31 7 2025"),
]
for _p in CANDIDATE_ROOTS:
    if os.path.isdir(_p):
        WEIGHTS_ROOT = _p
        break
else:
    # اگر هیچ‌کدام نبود، اولی را ست می‌کنیم تا پیام‌های خطا قابل‌فهم بماند
    WEIGHTS_ROOT = CANDIDATE_ROOTS[0]

# --------- Activation map ---------
_ACTS = {"Tanh": nn.Tanh(), "ReLU": nn.ReLU(), "Sigmoid": nn.Sigmoid(), "ELU": nn.ELU()}

# --------- Architectures (کلید 'layers' برای سازگاری state_dict) ---------
class MyNN1(nn.Module):
    def __init__(self, input_dim, h1, out_dim, act):
        super().__init__()
        A = _ACTS[act]
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), A,
            nn.Linear(h1, out_dim)
        )
    def forward(self, x): return self.layers(x)

class MyNN2(nn.Module):
    def __init__(self, input_dim, h1, h2, out_dim, act):
        super().__init__()
        A = _ACTS[act]
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), A,
            nn.Linear(h1, h2),       nn.BatchNorm1d(h2), A,
            nn.Linear(h2, out_dim)
        )
    def forward(self, x): return self.layers(x)

class MyNN3(nn.Module):
    def __init__(self, input_dim, h1, h2, h3, out_dim, act):
        super().__init__()
        A = _ACTS[act]
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), A,
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), A,
            nn.Linear(h2, h3),        nn.BatchNorm1d(h3), A,
            nn.Linear(h3, out_dim)
        )
    def forward(self, x): return self.layers(x)
class MyNN2Drop(nn.Module):
    def __init__(self, input_dim, h1, h2, out_dim, act, p=0.2):
        super().__init__()
        A = _ACTS[act]
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), A, nn.Dropout(p),
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), A, nn.Dropout(p),
            nn.Linear(h2, out_dim)
        )
    def forward(self, x): return self.layers(x)

class MyNN3Drop(nn.Module):
    def __init__(self, input_dim, h1, h2, h3, out_dim, act, p=0.2):
        super().__init__()
        A = _ACTS[act]
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), A, nn.Dropout(p),
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), A, nn.Dropout(p),
            nn.Linear(h2, h3),        nn.BatchNorm1d(h3), A, nn.Dropout(p),
            nn.Linear(h3, out_dim)
        )
    def forward(self, x): return self.layers(x)



class PositiveNN1(MyNN1): pass
class PositiveNN2(MyNN2): pass
class PositiveNN3(MyNN3): pass
class PositiveNN2Drop(MyNN2Drop): pass
class PositiveNN3Drop(MyNN3Drop): pass
# --------- Model specs ---------
MODEL_SPECS = {
    "GRF_Z":      dict(cls=PositiveNN3, args=(12, 20, 56, 40, 2, 'ReLU')),
    "GRF_X_Y":    dict(cls=MyNN3,       args=(12, 40, 64, 40, 4, 'ReLU')),
    "COP_X_Y":    dict(cls=MyNN3,       args=(12, 14, 58, 54, 4, 'ReLU')),
    "Spine_X":    dict(cls=MyNN1,       args=(12, 110, 6, 'ReLU')),
    "Spine_Y":    dict(cls=MyNN2,       args=(12, 36, 96, 6, 'ReLU')),
    "Spine_Z":    dict(cls=MyNN2,       args=(12, 30, 56, 6, 'ELU')),
    "Muscle_ES":  dict(cls=PositiveNN3, args=(12, 56, 48, 48, 2, 'ReLU')),
    "Muscle_MU":  dict(cls=PositiveNN1, args=(12, 180, 2, 'ReLU')),
    "Muscle_OBL": dict(cls=PositiveNN2, args=(12, 30, 34, 2, 'ELU')),
    "Muscle_PS":    dict(cls=PositiveNN2Drop, args=(12, 30, 68, 2, 'ELU')),
    "Muscle_QL":    dict(cls=PositiveNN3Drop, args=(12, 18, 10, 50, 2, 'Tanh')),
    "Muscle_Ex_OB": dict(cls=PositiveNN3Drop, args=(12, 10, 26, 30, 2, 'Tanh')),

}

_MODELS, _SCALES, _LOCK = {}, {}, threading.Lock()

def available_tasks():
    return sorted([name for name in MODEL_SPECS if os.path.isdir(os.path.join(WEIGHTS_ROOT, name))])

# ---------- utils: load best .pt ----------
def _best_pt(folder: str) -> str:
    cand = (
        glob.glob(os.path.join(folder, "best_val_model_params*.pt"))
        or glob.glob(os.path.join(folder, "last_val_model_params*.pt"))
        or glob.glob(os.path.join(folder, "*.pt"))
    )
    if not cand:
        raise FileNotFoundError(f"No .pt file in {folder}")
    cand.sort()
    return cand[0]

# ---------- read scaler (mean/std) ----------
def _read_mean_std(folder: str) -> Tuple[np.ndarray, np.ndarray]:
    mean_path = os.path.join(folder, "mean.csv")
    std_path  = os.path.join(folder, "std.csv")

    def _fallback() -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        return mean, std

    if not (os.path.isfile(mean_path) and os.path.isfile(std_path)):
        return _fallback()

    def _read_csv(p):
        df = pd.read_csv(p)
        if df.shape[1] >= 2:
            idx, val = df.columns[0], df.columns[1]
            s = df.set_index(idx)[val]
        else:
            s = df.iloc[:, 0]
        return s

    mean_s = _read_csv(mean_path)
    std_s  = _read_csv(std_path)
    order = ['x', 'y', 'z', 'W', 'H', 'Load']

    def pick(s, key):
        if key in s.index:
            return float(s.loc[key])
        for k in s.index:
            if str(k).strip().lower() == key.lower():
                return float(s.loc[k])
        raise KeyError(f"{key} not found in scaler index: {list(s.index)}")

    try:
        mean = np.array([pick(mean_s, k) for k in order], dtype=np.float32)
        std  = np.array([pick(std_s,  k) for k in order], dtype=np.float32)
        std[std == 0] = 1.0
        return mean, std
    except Exception:
        return _fallback()

# ---------- state_dict compatibility ----------
def _load_state_safely(model: nn.Module, state: dict):
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    def _swap_prefix(sd, src_prefix, dst_prefix):
        out = collections.OrderedDict()
        for k, v in sd.items():
            if k.startswith(src_prefix):
                out[k.replace(src_prefix, dst_prefix, 1)] = v
            else:
                out[k] = v
        return out

    model_keys = list(model.state_dict().keys())
    st_keys = list(state.keys())

    if any(k.startswith("net.") for k in st_keys) and any(k.startswith("layers.") for k in model_keys):
        state2 = _swap_prefix(state, "net.", "layers.")
        model.load_state_dict(state2, strict=True)
        return

    if any(k.startswith("layers.") for k in st_keys) and any(k.startswith("net.") for k in model_keys):
        state2 = _swap_prefix(state, "layers.", "net.")
        model.load_state_dict(state2, strict=True)
        return

    model.load_state_dict(state, strict=False)

# ---------- load model ----------
def get_model(task: str):
    """return (model, (mean,std), folder)"""
    with _LOCK:
        if task not in _MODELS:
            folder = os.path.join(WEIGHTS_ROOT, task)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Folder not found: {folder}")
            mean, std = _read_mean_std(folder)
            spec = MODEL_SPECS[task]
            model = spec['cls'](*spec['args'])
            state = torch.load(_best_pt(folder), map_location='cpu')
            _load_state_safely(model, state)
            model.eval()
            _MODELS[task] = model
            _SCALES[task] = (mean, std)
    folder = os.path.join(WEIGHTS_ROOT, task)
    return _MODELS[task], _SCALES[task], folder

# ---------- feature build ----------
def preprocess_12_features(payload: Dict[str, Any], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    ترتیب ورودی‌ها (۱۲تایی):
    x, y, z, W, H, Load, Handling_1.0, Handling_2.0, Lifting_0.0..Lifting_3.0
    """
    def _f(key, default=0.0):
        try:
            return float(payload.get(key, default))
        except Exception:
            return float(default)

    num = np.array([
        _f("x", 0), _f("y", 0), _f("z", 0),
        _f("W", 0), _f("H", 0), _f("Load", 0),
    ], dtype=np.float32)
    num = (num - mean) / std

    # Handling: 1(one-hand) یا 2(two-hand)
    try:
        handling = int(payload.get("Handling", 1))
    except Exception:
        handling = 1
    handling_oh = np.array([1.0 if handling == 1 else 0.0,
                            1.0 if handling == 2 else 0.0], dtype=np.float32)

    # Lifting: 0..3
    try:
        lifting = int(payload.get("Lifting", 0))
    except Exception:
        lifting = 0
    lifting_oh = np.zeros(4, dtype=np.float32)
    if 0 <= lifting < 4:
        lifting_oh[lifting] = 1.0

    feats = np.concatenate([num, handling_oh, lifting_oh])
    return feats

# ---------- single task predict ----------
def predict(task: str, payload: Dict[str, Any]):
    """x(12,) و خروجی مدل را برمی‌گرداند: (input_features, output_list)"""
    model, (mean, std), _ = get_model(task)
    x = preprocess_12_features(payload, mean, std)
    with torch.no_grad():
        y = model(torch.tensor(x[None, :])).cpu().numpy()[0]
    return x.tolist(), y.tolist()

# ---------- compat wrapper used by views.py ----------
def _coerce_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    بدنه را به فرم موردنیاز این فایل تبدیل می‌کند.
    ورودی ممکن است یکی از این دو باشد:
      - سبک جدید: x,y,z, W,H,Load, Handling(1/2), Lifting(0..3)
      - سبک قدیم: weight,height,load, HT(0/1), LT(0..3)
    """
    out: Dict[str, Any] = {}
    # موقعیت
    out["x"] = float(body.get("x", 0))
    out["y"] = float(body.get("y", 0))
    out["z"] = float(body.get("z", 0))
    # قدیم → جدید
    W = body.get("W", body.get("weight", 70))
    H = body.get("H", body.get("height", 1700))
    Load = body.get("Load", body.get("load", 15))
    out["W"] = float(W); out["H"] = float(H); out["Load"] = float(Load)

    if "Handling" in body:
        out["Handling"] = int(body.get("Handling"))
    elif "HT" in body:
        # HT: 0(one-hand), 1(two-hand) → Handling: 1 یا 2
        ht = int(body.get("HT"))
        out["Handling"] = 1 if ht == 0 else 2
    else:
        out["Handling"] = 1

    if "Lifting" in body:
        out["Lifting"] = int(body.get("Lifting"))
    elif "LT" in body:
        out["Lifting"] = int(body.get("LT"))
    else:
        out["Lifting"] = 0

    return out

def predict_all_models(body: Dict[str, Any]) -> Tuple[Dict[str, list], Dict[str, Any]]:
    """
    رپری که views.py انتظار دارد.
    تمام تسک‌ها را با همان ورودی اجرا می‌کند و خروجی‌ها را با کلیدهای UI برمی‌گرداند.
    """
    payload = _coerce_payload(body)

    tasks = [
        "GRF_X_Y", "GRF_Z", "COP_X_Y",
        "Spine_X", "Spine_Y", "Spine_Z",
        "Muscle_ES", "Muscle_MU", "Muscle_OBL",
        "Muscle_PS", "Muscle_QL", "Muscle_Ex_OB",
    ]
    results: Dict[str, list] = {}
    per_task_debug: Dict[str, Any] = {}

    for t in tasks:
        try:
            x12, y = predict(t, payload)
            results[t] = [float(v) for v in y]
            per_task_debug[t] = {
                "x12": x12,
                "out_len": len(y),
                "weights_dir": os.path.join(WEIGHTS_ROOT, t),
            }
        except Exception as e:
            # اگر مدل یا وزن پیدا نشد، خروجی خالی بده و پیام ثبت کن
            results[t] = []
            per_task_debug[t] = {"error": str(e), "weights_dir": os.path.join(WEIGHTS_ROOT, t)}

    debug = {
        "payload_used": payload,
        "tasks_loaded": available_tasks(),
        "per_task": per_task_debug,
        "weights_root": WEIGHTS_ROOT,
    }
    return results, debug
