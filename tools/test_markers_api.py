# tools/test_markers_api.py
# -*- coding: utf-8 -*-
import json, time, sys, csv
from pathlib import Path
import requests
import numpy as np

HOST = "http://127.0.0.1:8000"  # اگر پورت/آدرس فرق دارد عوض کن

# ---- کمک‌ها ---------------------------------------------------------------

def post_json(url, payload, timeout=20):
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        print("! Response is not JSON:", r.text[:300])
        raise
    return r.status_code, data

def group_xyz(fields, values):
    """fields: لیست نام‌ها با پسوند X/Y/Z ؛ values: لیست مقادیر به همان ترتیب
       خروجی: dict{name -> (x,y,z)}"""
    out = {}
    for i in range(0, len(fields), 3):
        try:
            name_x = str(fields[i])
            base = name_x[:-1]  # حذف X/Y/Z
            x = float(values[i])
            y = float(values[i+1])
            z = float(values[i+2])
            out[base] = (x, y, z)
        except Exception:
            continue
    return out

def parse_markers(outputs):
    """خروجی api_predict_markers را به dict یکپارچه تبدیل می‌کند."""
    all_pts = {}
    for part in ("Head", "BP", "Arms", "Legs"):
        obj = outputs.get(part)
        if not obj:
            continue
        fields = [str(x) for x in obj.get("fields", [])]
        values = obj.get("values", [])
        part_pts = group_xyz(fields, values)
        # ادغام
        for k, v in part_pts.items():
            all_pts[f"{part}:{k}"] = v
    return all_pts

def bbox_and_checks(marker_dict, hard_abs_limit=4000.0):
    """چک sanity: محدوده مختصات (میلی‌متر)، وجود NaN/Inf، و هشدار اگر خیلی غیرعادی بود."""
    if not marker_dict:
        return None
    xs, ys, zs = [], [], []
    bad = []
    for name, (x, y, z) in marker_dict.items():
        arr = np.array([x, y, z], dtype=float)
        if not np.isfinite(arr).all():
            bad.append(name)
        xs.append(x); ys.append(y); zs.append(z)
    bbox = dict(
        min_x=float(np.min(xs)), max_x=float(np.max(xs)),
        min_y=float(np.min(ys)), max_y=float(np.max(ys)),
        min_z=float(np.min(zs)), max_z=float(np.max(zs)),
    )
    bbox["span_x"] = bbox["max_x"] - bbox["min_x"]
    bbox["span_y"] = bbox["max_y"] - bbox["min_y"]
    bbox["span_z"] = bbox["max_z"] - bbox["min_z"]

    warnings = []
    if bad:
        warnings.append(f"NaN/Inf at markers: {', '.join(bad[:8])}{'...' if len(bad)>8 else ''}")
    # حدود تقریبی (بدن انسان نباید چند متر پخش شود)
    for key in ("min_x","max_x","min_y","max_y","min_z","max_z"):
        if abs(bbox[key]) > hard_abs_limit:
            warnings.append(f"Out-of-range: {key}={bbox[key]:.1f} mm (>|{hard_abs_limit}|)")
    for key in ("span_x","span_y","span_z"):
        if bbox[key] > 3000:  # 3 متر برای یک محور خیلی زیاد است
            warnings.append(f"Unusually large span: {key}={bbox[key]:.1f} mm")

    return bbox, warnings

def save_markers_csv(marker_dict, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name","X(mm)","Y(mm)","Z(mm)"])
        for name, (x,y,z) in marker_dict.items():
            w.writerow([name, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])
    return str(out_path)

# ---- کیس‌های آزمایشی ------------------------------------------------------

def sample_cases():
    # ورودی‌ها طبق توضیح شما: x,y,z (mm)، Handling: 1/2، Lifting: 0..3، وزن(kg) و قد(mm)
    # X,Y,Z حتماً نسبت به مبدأ میانی پاشنه‌ها (ارتفاع 0) تعریف شوند.
    return [
        dict(name="CASE_A_calm",
             payload=dict(x=30,  y=30,  z=150, W=70, H=1720, Load=10, Handling=2, Lifting=0)),
        dict(name="CASE_B_forward",
             payload=dict(x=250, y=300, z=400, W=80, H=1780, Load=15, Handling=2, Lifting=2)),
        dict(name="CASE_C_side_reach",
             payload=dict(x=500, y=50,  z=200, W=60, H=1650, Load=8,  Handling=1, Lifting=1)),
    ]

# ---- اجرای تست ------------------------------------------------------------

def run_one_case(case, export_csv=False):
    name = case["name"]; payload = case["payload"]
    print(f"\n=== {name} ===")
    print("payload:", payload)

    # 1) scalar outputs
    t0 = time.time()
    s_all, data_all = post_json(f"{HOST}/api/predict_all/", payload)
    t1 = time.time()
    print(f"[predict_all] status={s_all} in {1000*(t1-t0):.1f} ms")
    if not (s_all==200 and isinstance(data_all, dict) and data_all.get("ok")):
        print("!! predict_all failed:", data_all)
    else:
        # فقط چند نمونه نشون بده
        res = data_all.get("results", {})
        for k in ("GRF_Z","GRF_X_Y","COP_X_Y","Spine_Z"):
            if k in res:
                arr = res[k]
                print(f"  {k}: {arr if isinstance(arr, list) else arr}")

    # 2) markers
    t0 = time.time()
    s_m, data_m = post_json(f"{HOST}/api/predict_markers/", payload)
    t1 = time.time()
    print(f"[predict_markers] status={s_m} in {1000*(t1-t0):.1f} ms")
    if not (s_m==200 and isinstance(data_m, dict) and data_m.get("ok")):
        print("!! predict_markers failed:", data_m)
        return

    markers = parse_markers(data_m.get("outputs", {}))
    print(f"markers count: {len(markers)}")
    bb, warns = bbox_and_checks(markers)
    if bb:
        print("bbox (mm):", bb)
    if warns:
        print("WARNINGS:")
        for w in warns:
            print("  -", w)

    if export_csv:
        out = save_markers_csv(markers, Path("tests_out")/f"{name}_markers.csv")
        print("CSV saved ->", out)

def main():
    cases = sample_cases()
    export = "--csv" in sys.argv
    for c in cases:
        run_one_case(c, export_csv=export)

if __name__ == "__main__":
    main()
