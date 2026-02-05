# tools/test_mohseni_model_exact.py
import os
import numpy as np
from scipy.io import loadmat

# ---------- activations (stable, MATLAB-compatible) ----------
def act_tansig(n):  return np.tanh(n)
def act_purelin(n): return n
ACT = {"tansig": act_tansig, "purelin": act_purelin}

# ---------- robust unwrappers ----------
def unwrap(x):
    """Unwrap nested numpy object arrays to the first non-array object."""
    while isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        x = x.ravel()[0]
    return x

def as_float_vec(x):
    a = np.array(x)
    if a.dtype == object:
        a = np.array(a.tolist(), dtype=float)
    else:
        a = a.astype(float, copy=False)
    return a.reshape(-1)

def first_with_attrs(x, attrs=("xoffset","gain")):
    """Breadth-first search to find a struct that has given attrs (e.g., mapminmax ps)."""
    queue = [x]
    seen = set()
    while queue:
        cur = queue.pop(0)
        if id(cur) in seen: 
            continue
        seen.add(id(cur))
        if isinstance(cur, np.ndarray):
            queue.extend(list(cur.ravel()))
            continue
        if cur is None:
            continue
        try:
            if all(hasattr(cur, a) for a in attrs):
                return cur
        except Exception:
            pass
        # explore known fields
        for name in ("processSettings", "outputs", "inputs"):
            if hasattr(cur, name):
                queue.append(getattr(cur, name))
    return None

# ---------- mapminmax (apply / inverse) ----------
def mapminmax_apply(X, ps):
    # X: (features, N)
    ps = unwrap(ps)
    if ps is None:
        # pass-through if not found
        return X.astype(float, copy=False)
    X = np.asarray(X, dtype=float)
    xoffset = as_float_vec(getattr(ps, "xoffset", np.zeros(X.shape[0]))).reshape(-1,1)
    gain    = as_float_vec(getattr(ps, "gain",    np.ones(X.shape[0]))).reshape(-1,1)
    ymin    = getattr(ps, "ymin", -1.0)
    if np.isscalar(ymin):
        ymin = np.full((X.shape[0], 1), float(ymin))
    else:
        ymin = as_float_vec(ymin).reshape(-1,1)
    return (X - xoffset) * gain + ymin

def mapminmax_inverse(Y, ps):
    # x = (y - ymin)/gain + xoffset
    ps = unwrap(ps)
    if ps is None:
        # pass-through if not found
        return Y.astype(float, copy=False)
    Y = np.asarray(Y, dtype=float)
    xoffset = as_float_vec(getattr(ps, "xoffset", np.zeros(Y.shape[0]))).reshape(-1,1)
    gain    = as_float_vec(getattr(ps, "gain",    np.ones(Y.shape[0]))).reshape(-1,1)
    ymin    = getattr(ps, "ymin", -1.0)
    if np.isscalar(ymin):
        ymin = np.full((Y.shape[0], 1), float(ymin))
    else:
        ymin = as_float_vec(ymin).reshape(-1,1)
    gain_safe = np.where(gain == 0, 1.0, gain)
    return (Y - ymin) / gain_safe + xoffset

# ---------- pick non-empty matrices ----------
def pick_first_matrix(arr):
    arr = np.array(arr, dtype=object).ravel()
    for w in arr:
        W = np.array(w, dtype=float)
        if W.ndim == 2 and W.size and W.shape[0] > 0 and W.shape[1] > 0:
            return W
    return None

def build_two_layer_from_net(net):
    layers = np.array(net.layers, dtype=object).ravel()
    b_list = np.array(net.b, dtype=object).ravel()
    IW_flat = np.array(net.IW, dtype=object).ravel()
    LW_flat = np.array(net.LW, dtype=object).ravel()

    # layer1: IW
    W1 = pick_first_matrix(IW_flat)
    if W1 is None:
        raise RuntimeError("No non-empty IW found")
    b1 = np.array(b_list[0], dtype=float).reshape(-1,1)
    act1 = "purelin"
    if hasattr(layers[0], "transferFcn"):
        act1 = str(layers[0].transferFcn).strip().lower()

    # layer2: LW (prefer shape (out,hid))
    hid = W1.shape[0]
    b2  = np.array(b_list[1], dtype=float).reshape(-1,1)
    out = b2.shape[0]
    W2 = None
    for w in LW_flat:
        W = np.array(w, dtype=float)
        if W.ndim == 2 and W.shape == (out, hid):
            W2 = W; break
    if W2 is None:
        W2 = pick_first_matrix(LW_flat)
        if W2 is None:
            raise RuntimeError("No non-empty LW found")
    act2 = "purelin"
    if len(layers) > 1 and hasattr(layers[1], "transferFcn"):
        act2 = str(layers[1].transferFcn).strip().lower()

    return (W1,b1,act1), (W2,b2,act2)

def forward_two_layer(W1,b1,act1, W2,b2,act2, Xn):
    f1 = ACT.get(act1, act_purelin)
    f2 = ACT.get(act2, act_purelin)
    A1 = f1(W1 @ Xn + b1)
    A2 = f2(W2 @ A1 + b2)
    return A2

# ===================== MAIN =====================
base = os.path.dirname(__file__)
mat_path = os.path.join(base, "Mohseni Static 3D Coordinates Prediction Model.mat")
d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

net = d["net_Head"]
fields = np.array(d["Headfields"], dtype=object).ravel()

# robust find processSettings for input/output
inp_obj = unwrap(net.inputs)
out_obj = unwrap(net.outputs)
ps_in  = first_with_attrs(inp_obj, attrs=("xoffset","gain"))
ps_out = first_with_attrs(out_obj, attrs=("xoffset","gain"))
if ps_in is None:
    print("WARNING: input processSettings not found — using raw input (no mapminmax).")
if ps_out is None:
    print("WARNING: output processSettings not found — returning raw network output (no inverse mapminmax).")

# weights
(W1,b1,act1), (W2,b2,act2) = build_two_layer_from_net(net)
print(f"W1 {W1.shape} | W2 {W2.shape} | acts: {act1} -> {act2}")

# sample input: [x y z Lifting Handling weight height]  -> shape (7,1)
X = np.array([[40, 50, 120, 0, 1, 70, 1750]], dtype=float).T

# forward
Xn = mapminmax_apply(X, ps_in)
Yn = forward_two_layer(W1,b1,act1, W2,b2,act2, Xn)
Y  = mapminmax_inverse(Yn, ps_out).flatten()

print("\n✅ Output count:", len(Y))
for name, val in zip(fields, Y):
    print(f"{str(name):10s}: {val:10.4f}")
