# -*- coding: utf-8 -*-
# main/views.py
import json
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Mohseni (همان قبلی)
from .ml_mohseni import predict_markers_from_payload
# Hosseini (classic)
from .ml_models import predict_all_models
# 🔹 داینامیک مسیر
from .ml_path_dynamic import get_smooth_path


def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("/")
        return render(request, "login.html", {"error": "Invalid username or password"})
    return render(request, "login.html")


@login_required
def dashboard(request):
    return render(request, "dashboard.html")


def user_logout(request):
    logout(request)
    return redirect("/accounts/login/")


# ---------- کمک‌تابع‌ها ----------

def _as_float(d, k, default):
    try:
        return float(d.get(k, default))
    except Exception:
        return float(default)


def _as_int(d, k, default):
    try:
        return int(float(d.get(k, default)))
    except Exception:
        return int(default)


# ---------- مپینگ برای «حسینی» در حالت تک‌فریمی (استاتیک) ----------

def _map_ui_to_models_for_hosseini(body):
    """
    UI payload -> Hosseini payload (فقط Classic، تک فریم)
    قوانین از خودت:
      - HT: اگر One-handed => Hosseini=0 ، اگر Two-handed => Hosseini=1
      - LT: همان مقدار UI (0..3)
      - Height: از UI برحسب cm → برای مدل برحسب mm
    """
    # از UI:
    x = _as_float(body, "x", 0.0)
    y = _as_float(body, "y", 0.0)
    z = _as_float(body, "z", 0.0)
    W = _as_float(body, "W", 70.0)
    H_cm = _as_float(body, "H", 172.0)   # کاربر cm می‌دهد
    H = H_cm * 10.0                      # مدل mm می‌خواهد
    load = _as_float(body, "Load", 15.0)

    # LT: UI = 0..3 (Standing, Stoop, Semi, Full)
    LT_ui = _as_int(body, "Lifting", 0)  # 0..3
    LT_hosseini = LT_ui

    # HT: UI select value = 1(one) یا 2(two)
    HT_ui = _as_int(body, "Handling", 1)  # 1 یا 2
    HT_hosseini = 0 if HT_ui == 1 else 1  # قانون تو

    payload_hosseini = {
        "x": x, "y": y, "z": z,
        "LT": LT_hosseini,
        "HT": HT_hosseini,
        "weight": W,
        "height": H,
        "load": load,
    }

    # فقط برای دیباگ مپینگ «محسنی» (استفاده عملی نمی‌کنیم)
    LT_mohseni = LT_ui + 1
    HT_mohseni = 1 if HT_ui == 1 else 2
    payload_mohseni_debug = {
        "x": x, "y": y, "z": z,
        "LT": LT_mohseni,
        "HT": HT_mohseni,
        "weight": W,
        "height": H,
        "load": load,
    }

    return payload_hosseini, payload_mohseni_debug


# ---------- API استاتیک (قدیمی) ----------

@csrf_exempt
def api_predict_all(request):
    """
    فقط «حسینی» (classic). «محسنی» در api_predict_markers اجرا می‌شود.
    اینجا مپینگ را مطابق قوانین درستِ تو انجام می‌دهیم:
      HT: one-hand -> Hosseini=0 ، two-hand -> Hosseini=1
      LT: عین UI (0..3)
      H: از cm به mm
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Only POST allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

    hossein_payload, mohseni_payload_debug = _map_ui_to_models_for_hosseini(body)
    try:
        results, dbg = predict_all_models(hossein_payload)
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"classic model error: {e}"}, status=500)

    debug = {
        "ui_payload": body,
        "sent_to_hosseini": hossein_payload,
        "mohseni_expected_payload_debug": mohseni_payload_debug,
        "classic_debug": dbg,
    }
    return JsonResponse({"ok": True, "results": results, "debug": debug})


@csrf_exempt
def api_predict_markers(request):
    """
    اجرای مدل محسنی (تک فریم). همان کدی که قبلا داشتی می‌ماند.
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Only POST is allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON body"}, status=400)

    try:
        outputs, debug = predict_markers_from_payload(body)
        return JsonResponse({"ok": True, "outputs": outputs, "debug": debug})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


# ---------- API داینامیک (جدید) ----------

@csrf_exempt
def api_predict_dynamic(request):
    """
    حالت داینامیک:
      - از UI: مبدا (x,y,z) و مقصد (dst_x, dst_y, dst_z) + بقیه پارامترها
      - با get_smooth_path یک مسیر 100 نقطه‌ای بین مبدا و مقصد می‌سازیم.
      - برای هر فریم:
          * Hosseini (predict_all_models) با x,y,z همان فریم
          * Mohseni (predict_markers_from_payload) با x,y,z همان فریم
      - خروجی:
          frames: [
            {
              x, y, z,
              results: {... مثل استاتیک ...},
              markers: {... مثل api_predict_markers ...}
            }, ...
          ]
    فرانت از روی این سری زمانی، نمودار خروجی برحسب زمان می‌کشد.
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Only POST allowed"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

    # 1) گرفتن مبدا/مقصد از UI
    x0 = _as_float(body, "x", 0.0)
    y0 = _as_float(body, "y", 0.0)
    z0 = _as_float(body, "z", 0.0)

    x1 = _as_float(body, "dst_x", x0)
    y1 = _as_float(body, "dst_y", y0)
    z1 = _as_float(body, "dst_z", z0)

    P0 = [x0, y0, z0]
    Pf = [x1, y1, z1]

    # 2) پایه‌ی مپینگ حسینی از روی UI (برای W,H,Load, HT,LT)
    base_hosseini_payload, _moh_dbg = _map_ui_to_models_for_hosseini(body)

    # 3) پارامترهای عمومی برای محسنی از UI (همه فریم‌ها مشترک)
    W_ui = _as_float(body, "W", 70.0)
    H_cm_ui = _as_float(body, "H", 172.0)       # cm
    Handling_ui = _as_int(body, "Handling", 1)  # 1 or 2
    Lifting_ui = _as_int(body, "Lifting", 0)    # 0..3

    # 4) تولید مسیر
    try:
        path = get_smooth_path(P0, Pf, n_points=100)
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"path generation error: {e}"}, status=500)

    frames = []
    for i, (px, py, pz) in enumerate(path):
        # --- 4-1) Hosseini برای این فریم ---
        hos_pay = dict(base_hosseini_payload)
        hos_pay["x"] = float(px)
        hos_pay["y"] = float(py)
        hos_pay["z"] = float(pz)

        try:
            res_h, dbg_h = predict_all_models(hos_pay)
        except Exception as e:
            return JsonResponse({"ok": False, "error": f"classic model error at frame {i}: {e}"}, status=500)

        # --- 4-2) Mohseni برای این فریم ---
        # توجه: predict_markers_from_payload خودش H (cm) را به mm تبدیل می‌کند.
        moh_pay = {
            "x": float(px),
            "y": float(py),
            "z": float(pz),
            "W": W_ui,
            "H": H_cm_ui,
            "Handling": Handling_ui,
            "Lifting": Lifting_ui,
        }
        try:
            out_m, dbg_m = predict_markers_from_payload(moh_pay)
        except Exception as e:
            return JsonResponse({"ok": False, "error": f"mohseni model error at frame {i}: {e}"}, status=500)

        frames.append({
            "x": float(px),
            "y": float(py),
            "z": float(pz),
            "results": res_h,
            "markers": out_m,
        })

    debug = {
        "ui_payload": body,
        "P0": P0,
        "Pf": Pf,
        "n_frames": len(frames),
    }
    return JsonResponse({"ok": True, "frames": frames, "debug": debug})
