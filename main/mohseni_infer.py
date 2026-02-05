# main/mohseni_infer.py
import numpy as np
from scipy.io import loadmat

def _tansig(x):
    return np.tanh(x)  # MATLAB tansig

def _mm_apply(x, ps):
    # mapminmax forward: y = (x - xoffset) * gain + ymin
    return (x - ps['xoffset'][:, None]) * ps['gain'][:, None] + ps['ymin']

def _mm_reverse(y, ps):
    # inverse mapminmax: x = (y - ymin) / gain + xoffset
    return (y - ps['ymin']) / ps['gain'][:, None] + ps['xoffset'][:, None]

def _extract_ps(ps_struct, expected_dim):
    gain = np.array(ps_struct.gain, dtype=float).reshape(-1)
    xoffset = np.array(ps_struct.xoffset, dtype=float).reshape(-1)
    ymin = float(ps_struct.ymin)
    assert gain.shape[0] == expected_dim and xoffset.shape[0] == expected_dim
    return {"gain": gain, "xoffset": xoffset, "ymin": ymin}

def _extract_net(mat_net, in_dim, out_dim):
    W1 = np.array(mat_net.IW[0], dtype=float)         # (h, in)
    b1 = np.array(mat_net.b[0], dtype=float).reshape(-1)
    W2 = np.array(mat_net.LW[1][0], dtype=float)      # (out, h)
    b2 = np.array(mat_net.b[1], dtype=float).reshape(-1)
    ps_in  = _extract_ps(mat_net.inputs.processSettings, in_dim)
    out_struct = mat_net.outputs[1]  # struct با processSettings
    ps_out = _extract_ps(out_struct.processSettings, out_dim)
    tf = [ly.transferFcn for ly in mat_net.layers]
    assert tf[0] == 'tansig' and tf[1] == 'purelin', f"Unexpected transferFcn {tf}"
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "ps_in": ps_in, "ps_out": ps_out}

def _forward_batch(net, X):  # X: (n,7)
    W1, b1, W2, b2 = net["W1"], net["b1"], net["W2"], net["b2"]
    ps_in, ps_out = net["ps_in"], net["ps_out"]
    X = np.asarray(X, dtype=float)
    assert X.ndim == 2 and X.shape[1] == W1.shape[1]
    Xn = _mm_apply(X.T, ps_in)             # (in, n)
    A1 = _tansig(W1 @ Xn + b1[:, None])    # (h, n)
    YN = W2 @ A1 + b2[:, None]             # (out, n) normalized
    Y  = _mm_reverse(YN, ps_out)           # (out, n) physical
    return Y.T                              # (n, out)

class MohseniModel:
    def __init__(self, mat_path):
        m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        def out_size(net): return net.layers[-1].size
        self.nets = {
            "Head": _extract_net(m['net_Head'], 7, out_size(m['net_Head'])),
            "BP":   _extract_net(m['net_BP'],   7, out_size(m['net_BP'])),
            "Arms": _extract_net(m['net_Arms'], 7, out_size(m['net_Arms'])),
            "Legs": _extract_net(m['net_Legs'], 7, out_size(m['net_Legs'])),
        }
        self.fields = {
            "Head": [str(s) for s in m['Headfields'].tolist()],
            "BP":   [str(s) for s in m['BPfields'].tolist()],
            "Arms": [str(s) for s in m['Armsfields'].tolist()],
            "Legs": [str(s) for s in m['Legsfields'].tolist()],
        }

    def predict(self, X):  # X: (n,7)
        return {name: _forward_batch(net, X) for name, net in self.nets.items()}
