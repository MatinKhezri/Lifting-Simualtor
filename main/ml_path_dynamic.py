# -*- coding: utf-8 -*-
# main/ml_path_dynamic.py

import os
import numpy as np

# ← فقط همین سه تابع واقعاً در فایل path_predict_knn.py تو وجود دارند
from .path_predict_knn import (
    load_trajectories,
    predict_trajectory_knn,
    smooth_knn_with_basis,
)

# ---------- تنظیم مسیر newoutput.mat ----------
_BASE = os.path.dirname(os.path.dirname(__file__))   # مسیر پروژه
MAT_PATH = os.path.join(os.path.dirname(__file__), "tools", "newoutput.mat")

# ---------- کش دیتاست یک‌بار بارگذاری ----------
_TRAJ_CACHE = None  # (trajectories, endpoints, mu, sigma)


def _get_dataset():
    """
    یکبار دیتاست را از newoutput.mat می‌خوانیم و کش می‌کنیم
    """
    global _TRAJ_CACHE
    if _TRAJ_CACHE is None:
        trajectories, endpoints, mu, sigma = load_trajectories(MAT_PATH)
        _TRAJ_CACHE = (trajectories, endpoints, mu, sigma)
    return _TRAJ_CACHE


def get_smooth_path(P0, Pf, n_points=100, k=5, alpha=3.0):
    """
    P0, Pf = [x, y, z] نقطه مبدا و مقصد
    خروجی = مسیر صاف شده (n_points, 3)
    """
    P0 = np.asarray(P0, dtype=np.float32)
    Pf = np.asarray(Pf, dtype=np.float32)

    trajectories, endpoints, mu, sigma = _get_dataset()

    # مسیر خام KNN
    P_knn, nn_idx, nn_dist = predict_trajectory_knn(
        P0, Pf,
        trajectories, endpoints,
        mu, sigma,
        k=k,
        n_points=n_points,
        alpha=alpha,
    )

    # مسیر صاف‌شده + تضمین نقطه اول و آخر دقیقاً P0 و Pf هستند
    P_smooth = smooth_knn_with_basis(P0, Pf, P_knn, n_points=n_points)

    return P_smooth  # np.ndarray (100,3)
