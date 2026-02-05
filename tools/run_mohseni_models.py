# -*- coding: utf-8 -*-
"""
run_mohseni_models.py — نسخه نهایی و هماهنگ با MATLAB
- نرمال‌سازی ورودی طبق Min_In/Max_In
- سپس mapminmax ورودی (طبق net.inputs.processSettings)
- فوروارد شبکه (tansig -> purelin)
- واگردانی mapminmax خروجی (net.outputs{...}.processSettings)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat

# ===== مسیر فایل MAT (تغییر نده) =====
MAT_FILE = Path(r"C:\darsi\M.s.c\My Thesis\site\training\tools\Mohseni Static 3D Coordinates Prediction Model.mat")

# ---------- توابع پایه ----------
def tansig(n):
    return 2.0 / (1.0 + np.exp(-2.0 * n)) - 1.0

def mapminmax_forward(x, ps):
    """y = (x - xoffset) * gain + ymin | x,y: (features, batch)"""
    ymin = np.array(ps.ymin).reshape(-1, 1)
    gain = np.array(ps.gain).reshape(-1, 1)
    xoffset = np.array(ps.xoffset).reshape(-1, 1)
    return (x - xoffset) * gain + ymin

def mapminmax_inverse(y, ps):
    """x = (y - ymin)/gain + xoffset"""
    ymin = np.array(ps.ymin).reshape(-1, 1)
    gain = np.array(ps.gain).reshape(-1, 1)
    xoffset = np.array(ps.xoffset).reshape(-1, 1)
    return (y - ymin) / gain + xoffset

def normalize_minmax_user(X_raw, Min_In, Max_In):
    """نرمال‌سازی طبق کد متلب شما (به 0..1) | X_raw: (n,7)"""
    return (X_raw - Min_In) / (Max_In - Min_In)

def extract_two_layer_params(net):
    """استخراج پارامترهای شبکه (2 لایه: tansig -> purelin)"""
    W1 = np.array(net.IW.item(0), dtype=float)        # (hidden, in)
    W2 = np.array(net.LW[1, 0], dtype=float)          # (out, hidden)
    b1 = np.array(net.b[0], dtype=float).reshape(-1, 1)
    b2 = np.array(net.b[1], dtype=float).reshape(-1, 1)
    in_ps = net.inputs.processSettings                # mapminmax ورودی
    out_ps = net.outputs[1].processSettings           # mapminmax خروجی
    return W1, b1, W2, b2, in_ps, out_ps

def run_one_net(net, X_raw, Min_In, Max_In, field_names=None):
    """
    X_raw: (n,7) با ترتیب: x,y,z,LT,HT,weight,height
    خروجی: DataFrame (n, outdim)
    """
    # 1) نرمال‌سازی (همان روش MATLAB)
    Xn_user = normalize_minmax_user(X_raw, Min_In, Max_In)
    Xn_user = Xn_user.T  # (7,n)

    # 2) mapminmax ورودی شبکه
    W1, b1, W2, b2, in_ps, out_ps = extract_two_layer_params(net)
    Xproc = mapminmax_forward(Xn_user, in_ps)

    # 3) فوروارد: tansig -> purelin
    A1 = tansig(W1 @ Xproc + b1)
    Yn = W2 @ A1 + b2

    # 4) واگردانی mapminmax خروجی
    Y = mapminmax_inverse(Yn, out_ps)

    # 5) خروجی DataFrame
    cols = list(field_names) if field_names is not None else [f"out_{i}" for i in range(Y.shape[0])]
    cols = [c.decode() if isinstance(c, (bytes, bytearray)) else c for c in cols]
    return pd.DataFrame(Y.T, columns=cols)

def predict_all(mat_path, inputs):
    """اجرای چهار شبکه (Head, BP, Arms, Legs)"""
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    nets = {}
    for key in ["net_Head", "net_BP", "net_Arms", "net_Legs"]:
        if key in data:
            nets[key] = data[key]

    fields = {
        "net_Head": data.get("Headfields", None),
        "net_BP":   data.get("BPfields",   None),
        "net_Arms": data.get("Armsfields", None),
        "net_Legs": data.get("Legsfields", None),
    }

    Min_In = np.array(data["Min_In"], dtype=float).reshape(1, -1)
    Max_In = np.array(data["Max_In"], dtype=float).reshape(1, -1)

    outputs = {}
    for k, net in nets.items():
        outputs[k] = run_one_net(net, inputs, Min_In, Max_In, fields.get(k))
    return outputs

# =============== اجرای نمونه‌ی تستی ===============
if __name__ == "__main__":
    print("Loading MAT file from:", MAT_FILE)
    if not MAT_FILE.exists():
        print("❗ فایل MAT پیدا نشد.")
        sys.exit(1)

    # نمونه ورودی هماهنگ با MATLAB
    # 🔹 دقت کن: Lifting و Handling از 1 شروع می‌شن (نه 0)
    sample = np.array([[400, 600, 900, 1, 1, 70, 2000]], dtype=float)

    outs = predict_all(MAT_FILE, sample)
    if not outs:
        print("هیچ شبکه‌ای در فایل پیدا نشد.")
        sys.exit(0)

    for name, df in outs.items():
        print(f"\n=== {name.replace('net_','')} ===")
        print(df.round(3).iloc[:1])
