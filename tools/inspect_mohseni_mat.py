# tools/inspect_mohseni_mat.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def mat_to_dict(obj):
    """تبدیل ساختارهای متلب به دیکشنری/لیست پایتونی."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:  # struct array
            return [{k: mat_to_dict(obj[k][i]) for k in obj.dtype.names} for i in range(obj.shape[0])]
        elif obj.size == 1:
            return mat_to_dict(obj.item())
        else:
            return [mat_to_dict(x) for x in obj]
    elif hasattr(obj, "__dict__"):
        return {k: mat_to_dict(v) for k,v in obj.__dict__.items()}
    else:
        return obj

def short_shape(x):
    try:
        return f"shape={np.array(x).shape}"
    except Exception:
        return ""

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_mohseni_mat.py <path_to_mat_file>")
        return
    mat_path = Path(sys.argv[1])
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # حذف کلیدهای سیستمی
    top = {k:v for k,v in d.items() if not k.startswith("__")}
    print("Top-level keys:", list(top.keys()))

    # تلاش برای پیدا کردن بخش‌های معروف
    for k,v in top.items():
        print(f"\n[{k}] type={type(v).__name__} {short_shape(v)}")

    # اگر ساختارها تودرتو باشند
    # تلاش برای پیدا کردن norm و بخش‌های Head/BP/Arms/Legs
    try:
        model = top.get("model", top)  # بعضی‌ها داخل کلید model هستند
        # به دیکشنری پایتونی تبدیل
        model_p = mat_to_dict(model)
        # چاپ کلیدها سطح اول
        if isinstance(model_p, dict):
            print("\nModel-level keys:", list(model_p.keys()))
            # norm
            norm = model_p.get("norm") or model_p.get("Normalization") or {}
            if norm:
                print("Normalization keys:", list(norm.keys()))
                for key in ("mu","mean","sigma","std"):
                    if key in norm:
                        arr = np.array(norm[key], dtype=float).reshape(-1)
                        print(f"  {key}: {arr} (len={len(arr)})")
            # بخش‌ها
            for part in ("Head","BP","Arms","Legs"):
                p = model_p.get(part)
                if p:
                    keys = list(p.keys()) if isinstance(p, dict) else type(p).__name__
                    print(f"\n[{part}] keys:", keys)
                    flds = p.get("fields") if isinstance(p, dict) else None
                    if flds is not None:
                        print(f"  fields count: {len(flds)}   example: {flds[:6]}")
                    # اگر وزن‌ها لایه‌ای داشته باشد (فرضی)
                    if "layers" in p:
                        L = p["layers"]
                        print(f"  layers: {len(L)}")
                        # نمونه
                        L0 = L[0] if isinstance(L, list) and L else None
                        if isinstance(L0, dict):
                            print("  layer[0] keys:", list(L0.keys()))
        else:
            print("\n(model not a dict-like after conversion)")
    except Exception as e:
        print(">> Error while parsing nested model:", e)

if __name__ == "__main__":
    main()
