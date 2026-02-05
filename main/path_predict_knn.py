# path_predict_knn.py
import numpy as np
import scipy.io as sio


# ============================================
# 1) Load trajectories + endpoints
# ============================================

def load_trajectories(mat_path="newoutput.mat"):
    """
    Loads newoutput.mat and extracts:
      - trajectories: list of arrays (Ni, 4) [t, x, y, z]
      - endpoints: (N_traj, 6) [x0, y0, z0, xf, yf, zf]
      - endpoints_mean, endpoints_std: for normalization in KNN
    """
    mat = sio.loadmat(mat_path)
    newoutput = mat["newoutput"]  # shape (1, num_traj)

    num_traj = newoutput.shape[1]
    trajectories = []
    endpoints = []

    for i in range(num_traj):
        traj = newoutput[0, i]  # (Ni, 4): [t, x, y, z]
        if traj.shape[0] < 2:
            continue

        P = traj[:, 1:4]  # (Ni, 3)
        P0 = P[0, :]
        Pf = P[-1, :]

        trajectories.append(traj)
        endpoints.append(np.concatenate([P0, Pf], axis=0))  # (6,)

    endpoints = np.array(endpoints, dtype=np.float32)  # (N_use, 6)

    endpoints_mean = endpoints.mean(axis=0)
    endpoints_std = endpoints.std(axis=0) + 1e-8

    return trajectories, endpoints, endpoints_mean, endpoints_std


# ============================================
# 2) Resample a trajectory to fixed n_points
# ============================================

def resample_traj(traj, n_points=100):
    """
    traj: (Ni, 4) [t,x,y,z]
    returns: (n_points, 3) [x,y,z] resampled in normalized time s∈[0,1]
    """
    t = traj[:, 0]
    P = traj[:, 1:4]

    if t[-1] != t[0]:
        s = (t - t[0]) / (t[-1] - t[0])
    else:
        s = np.linspace(0.0, 1.0, len(t))

    s_new = np.linspace(0.0, 1.0, n_points)
    P_new = np.zeros((n_points, 3), dtype=np.float32)

    for d in range(3):
        P_new[:, d] = np.interp(s_new, s, P[:, d])

    return P_new


# ============================================
# 3) KNN prediction (endpoints space, weighted)
# ============================================

def predict_trajectory_knn(
    P0, Pf,
    trajectories, endpoints,
    endpoints_mean, endpoints_std,
    k=5,
    n_points=100,
    alpha=2.0,
):
    """
    P0, Pf: [x,y,z]
    trajectories: list of (Ni,4)
    endpoints: (N_traj,6)
    endpoints_mean/std: for normalization
    k: neighbours
    alpha: weight exponent (w ~ 1/d^alpha)

    returns:
      P_knn: (n_points,3) averaged KNN trajectory
      nn_idx: indices of used neighbours (0..len(trajectories)-1)
      nn_dist: distances (in normalized endpoint space)
    """
    P0 = np.array(P0, dtype=np.float32)
    Pf = np.array(Pf, dtype=np.float32)
    query = np.concatenate([P0, Pf], axis=0)  # (6,)

    # normalize endpoints and query
    endpoints_norm = (endpoints - endpoints_mean) / endpoints_std
    query_norm = (query - endpoints_mean) / endpoints_std

    diff = endpoints_norm - query_norm[None, :]    # (N,6)
    dist = np.linalg.norm(diff, axis=1)           # (N,)

    k = min(k, endpoints.shape[0])
    nn_idx = np.argsort(dist)[:k]
    nn_dist = dist[nn_idx]

    # distance weights (nearer → bigger weight)
    eps = 1e-6
    w = 1.0 / ((nn_dist + eps) ** alpha)
    w = w / w.sum()       # normalize

    # resample each neighbour trajectory
    resampled = []
    for idx in nn_idx:
        traj_i = trajectories[idx]
        P_res = resample_traj(traj_i, n_points=n_points)  # (n_points,3)
        resampled.append(P_res)

    resampled = np.stack(resampled, axis=0)  # (k, n_points, 3)
    w_reshaped = w[:, None, None]           # (k,1,1)

    P_knn = (resampled * w_reshaped).sum(axis=0)  # (n_points,3)

    return P_knn.astype(np.float32), nn_idx, nn_dist


# ============================================
# 4) Smooth KNN using residual basis (mapping)
# ============================================

def _basis_matrix(s):
    """
    Build basis matrix Φ(s) of shape (N,M) for residual mapping.

    φ1 = s(1-s)
    φ2 = s(1-s)(2s-1)
    φ3 = s(1-s)(2s-1)^2

    All vanish at s=0,1 → endpoints stay exact.
    """
    s = np.asarray(s, dtype=np.float64)
    b1 = s * (1.0 - s)
    b2 = b1 * (2.0 * s - 1.0)
    b3 = b1 * (2.0 * s - 1.0) ** 2
    Phi = np.stack([b1, b2, b3], axis=1)  # (N,3)
    return Phi


def smooth_knn_with_basis(P0, Pf, P_knn, n_points=100):
    """
    P0, Pf  : [x,y,z]
    P_knn   : (N,3) raw KNN trajectory (in world coords)
    n_points: output sample count

    Returns:
      P_smooth: (n_points,3) trajectory:
        - passes exactly through P0,Pf
        - smoothed & fitted to P_knn by least squares on residuals
    """
    P0 = np.asarray(P0, dtype=np.float64)
    Pf = np.asarray(Pf, dtype=np.float64)
    P_knn = np.asarray(P_knn, dtype=np.float64)

    N = P_knn.shape[0]
    if N < 4:
        return P_knn.astype(np.float32)

    # parameter for KNN samples
    s_samples = np.linspace(0.0, 1.0, N)

    # straight line
    P_line_samples = P0[None, :] + s_samples[:, None] * (Pf - P0)[None, :]
    R_samples = P_knn - P_line_samples     # residuals (N,3)

    # basis on sample s
    Phi = _basis_matrix(s_samples)        # (N,3)

    # least squares: Phi * W ≈ R_samples
    # W has shape (3,3): 3 basis × 3 dims
    W = np.linalg.lstsq(Phi, R_samples, rcond=None)[0]  # (3,3)

    # now build on fine grid
    s_new = np.linspace(0.0, 1.0, n_points)
    P_line_new = P0[None, :] + s_new[:, None] * (Pf - P0)[None, :]
    Phi_new = _basis_matrix(s_new)        # (n_points,3)

    R_new = Phi_new @ W                   # (n_points,3)

    P_smooth = P_line_new + R_new         # (n_points,3)

    # sanity: enforce exact endpoints numerically
    P_smooth[0, :] = P0
    P_smooth[-1, :] = Pf

    return P_smooth.astype(np.float32)


# ============================================
# quick manual test (اختیاری)
# ============================================

if __name__ == "__main__":
    print("Quick test...")
    trajectories, endpoints, mu, sigma = load_trajectories("newoutput.mat")
    print("num traj:", len(trajectories))
    traj = trajectories[0]
    P0 = traj[0, 1:4]
    Pf = traj[-1, 1:4]

    P_knn, nn_idx, nn_dist = predict_trajectory_knn(
        P0, Pf, trajectories, endpoints, mu, sigma, k=3, n_points=100, alpha=2.0
    )
    P_smooth = smooth_knn_with_basis(P0, Pf, P_knn, n_points=100)
    print("KNN shape:", P_knn.shape, "Smooth shape:", P_smooth.shape)
