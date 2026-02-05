# -*- coding: utf-8 -*-
# main/ml_mohseni.py

import numpy as np

# از همان کدی که قبلاً برای تست استفاده می‌کردی
from tools.run_mohseni_models import predict_all as moh_predict_all, MAT_FILE


def _to_float(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def _to_int(d, key, default=0):
    try:
        return int(d.get(key, default))
    except Exception:
        return int(default)


def _df_to_fields_values(df):
    """
    df: pandas DataFrame (n, outdim)
    خروجی را به شکل {fields: [...], values: [...]} برای ردیف اول برمی‌گرداند.
    """
    if df is None or df.empty:
        return {"fields": [], "values": []}
    cols = [
        c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
        for c in df.columns
    ]
    vals = df.iloc[0].tolist()
    return {"fields": cols, "values": vals}


# ---------- ابزار برای تبدیل Left_Direction (سمت چپ / x منفی) ----------

def _fields_values_to_marker_map(fields, values):
    """
    fields: مثل ['LFHDX','LFHDY','LFHDZ', 'RFHDX', ...]
    values: لیست عددی با طول برابر

    خروجی: dict مثل {'LFHD': (x,y,z), 'RFHD': (x,y,z), ...}
    """
    markers = {}
    if not fields or not values:
        return markers

    n = min(len(fields), len(values))
    # هر سه ستون = یک مارکر (X,Y,Z)
    for i in range(0, n, 3):
        if i + 2 >= n:
            break
        name_x = str(fields[i])
        base = name_x[:-1]  # حذف X/Y/Z انتهایی → مثلاً 'LFHDX' → 'LFHD'
        try:
            x = float(values[i])
            y = float(values[i + 1])
            z = float(values[i + 2])
        except Exception:
            x, y, z = 0.0, 0.0, 0.0
        markers[base] = (x, y, z)
    return markers


def _marker_map_to_values(fields, marker_map):
    """
    marker_map: {'LFHD': (x,y,z), ...}
    fields: همان لیست اسامی ستون‌ها

    خروجی: لیست values به همان ترتیب fields
    """
    vals = []
    n = len(fields)
    for i in range(0, n, 3):
        if i + 2 >= n:
            break
        name_x = str(fields[i])
        base = name_x[:-1]
        x, y, z = marker_map.get(base, (0.0, 0.0, 0.0))
        vals.extend([x, y, z])
    return vals


def _apply_left_direction_on_part(part_fv):
    """
    تقریب تابع Left_Direction روی یک بخش (Head / BP / Arms / Legs)
    part_fv: {"fields":[...], "values":[...]}

    منطق:
      - اگر نام مارکر با 'L' یا 'R' شروع شود:
          * مختصات را از سمت مقابل می‌گیریم (L<->R) و سپس X را منفی می‌کنیم.
      - در غیر این صورت (مارکرهای مرکزی مثل C7, T10, ...):
          * فقط X را منفی می‌کنیم.
    """
    fields = part_fv.get("fields") or []
    values = part_fv.get("values") or []
    if not fields or not values:
        return part_fv

    markers = _fields_values_to_marker_map(fields, values)
    if not markers:
        return part_fv

    new_markers = {}

    for base, (x, y, z) in markers.items():
        if not base:
            new_markers[base] = (x, y, z)
            continue

        side = base[0]
        rest = base[1:]

        # مارکرهای چپ/راست
        if side in ("L", "R") and rest:
            opp_side = "R" if side == "L" else "L"
            opp_base = opp_side + rest
            # اگر مارکر سمت مقابل وجود داشته باشد، از آن استفاده می‌کنیم
            sx, sy, sz = markers.get(opp_base, (x, y, z))
        else:
            # مارکرهای مرکزی بدن
            sx, sy, sz = x, y, z

        # آینه کردن روی محور X
        new_x = -sx
        new_y = sy
        new_z = sz
        new_markers[base] = (new_x, new_y, new_z)

    new_values = _marker_map_to_values(fields, new_markers)
    return {"fields": fields, "values": new_values}


def _apply_left_direction(outputs):
    """
    outputs: dict مثل {'Head':{fields,values}, 'BP':..., 'Arms':..., 'Legs':...}
    روی همه بخش‌ها Left_Direction اعمال می‌کند.
    """
    out2 = {}
    for part in ("Head", "BP", "Arms", "Legs"):
        fv = outputs.get(part)
        if not fv:
            continue
        out2[part] = _apply_left_direction_on_part(fv)
    return out2


# ---------- تابع اصلی که ویوها صدا می‌زنند ----------

def predict_markers_from_payload(payload: dict):
    """
    ورودی خام از فرم را می‌گیرد، نگاشت صحیح برای مدل Mohseni انجام می‌دهد،
    سپس مستقیماً از tools.run_mohseni_models.predict_all استفاده می‌کند.

    اضافه: اگر x < 0 باشد، منطق سمت چپ (Left_Direction) روی خروجی اعمال می‌شود.
    """

    # ---- نگاشت ورودی‌ها برای مدل Mohseni ----
    # فرم: Handling: 1=One-handed, 2=Two-handed  → Mohseni: 1 یا 2 (همان)
    raw_handling = _to_int(payload, "Handling", 1)
    handling_mohseni = 1 if raw_handling == 1 else 2

    # فرم: Lifting: 0..3  → Mohseni: 1..4 (شاخص MATLAB)
    lifting_mohseni = _to_int(payload, "Lifting", 0) + 1

    x = _to_float(payload, "x", 0.0)
    y = _to_float(payload, "y", 0.0)
    z = _to_float(payload, "z", 0.0)
    w = _to_float(payload, "W", 70.0)        # وزن فرد (kg)
    h = _to_float(payload, "H", 172.0) * 10  # قد فرد (mm)

    # آیا کاربر x منفی داده؟
    is_negative = (x < 0.0)

    # مثل Left_Direction: برای مدل همیشه x مثبت می‌دهیم
    x_for_model = -x if is_negative else x

    # ترتیب ورودی دقیقا مطابق کدی که قبلاً تست می‌کردی:
    # [x, y, z, Lifting(1..4), Handling(1..2), weight(kg), height(mm)]
    X = np.array([[x_for_model, y, z, lifting_mohseni, handling_mohseni, w, h]], dtype=float)

    # ---- اجرای مدل Mohseni ----
    outs_df_map = moh_predict_all(MAT_FILE, X)  # dict: {'net_Head': df, 'net_BP': df, ...}

    # تبدیل به ساختار JSON‌ پایه
    base_outputs = {}
    for key, df in outs_df_map.items():
        short = key.replace("net_", "")  # Head, BP, Arms, Legs
        base_outputs[short] = _df_to_fields_values(df)

    # اگر x منفی باشد، منطق سمت چپ را روی خروجی اعمال کن
    if is_negative:
        outputs = _apply_left_direction(base_outputs)
    else:
        outputs = base_outputs

    debug = {
        "model": "Mohseni",
        "mat_file": str(MAT_FILE),
        "mapped_input_order": ["x", "y", "z", "Lifting(+1)", "Handling", "W(kg)", "H(mm)"],
        "mapped_input_row": X.tolist(),
        "original_x": float(x),
        "x_used_for_model": float(x_for_model),
        "used_left_direction": bool(is_negative),
        "notes": "If x < 0, outputs are mirrored and L/R swapped approximately like MATLAB Left_Direction.",
    }

    return outputs, debug
