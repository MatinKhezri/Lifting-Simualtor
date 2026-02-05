import numpy as np
from scipy.io import loadmat
import torch
import os

# مسیر فایل مدل
base = os.path.dirname(__file__)
mat_path = os.path.join(base, "Mohseni Static 3D Coordinates Prediction Model.mat")
print("Loading model from:", mat_path)

# بارگذاری داده
data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

min_in, max_in = data["Min_In"], data["Max_In"]

models = {
    "Head": ("net_Head", "Min_Head", "Max_Head", "Headfields"),
    "BP": ("net_BP", "Min_BP", "Max_BP", "BPfields"),
    "Arms": ("net_Arms", "Min_Arms", "Max_Arms", "Armsfields"),
    "Legs": ("net_Legs", "Min_Legs", "Max_Legs", "Legsfields"),
}

# ورودی تست
inp = np.array([[40, 50, 120, 0, 1, 70, 1750]], dtype=np.float32)
print("\nInput shape:", inp.shape)

# نرمال‌سازی ورودی
inp_norm = (inp - min_in) / (max_in - min_in)
inp_norm = np.nan_to_num(inp_norm, nan=0.0)
print("Normalized input:", inp_norm)


def safe_denorm(y_pred, min_v, max_v):
    y_pred = np.array(y_pred).flatten()
    min_v, max_v = np.array(min_v).flatten(), np.array(max_v).flatten()
    n = min(len(y_pred), len(min_v), len(max_v))
    y_pred, min_v, max_v = y_pred[:n], min_v[:n], max_v[:n]
    return y_pred * (max_v - min_v) + min_v


def mat_to_torch(net_struct):
    layers = []
    if not hasattr(net_struct, "layers"):
        print("⚠️  Net has no layers attribute")
        return torch.nn.Identity()

    for layer in net_struct.layers:
        if hasattr(layer, "weights"):
            W = torch.tensor(layer.weights[0].T, dtype=torch.float32)
            b = torch.tensor(layer.weights[1].flatten(), dtype=torch.float32)
            lin = torch.nn.Linear(W.shape[1], W.shape[0])
            lin.weight.data = W
            lin.bias.data = b
            layers.append(lin)
            layers.append(torch.nn.ReLU())
    if not layers:
        print("⚠️  Empty layer list, using Identity()")
        return torch.nn.Identity()
    return torch.nn.Sequential(*layers[:-1])


outputs = {}
for name, (net_key, min_key, max_key, field_key) in models.items():
    print(f"\n--- Running model: {name} ---")
    net = data[net_key]
    model = mat_to_torch(net)
    inp_tensor = torch.tensor(inp_norm, dtype=torch.float32)
    y_pred = model(inp_tensor).detach().numpy().flatten()
    fields = data[field_key]
    y_real = safe_denorm(y_pred, data[min_key], data[max_key])
    outputs[name] = dict(zip(fields[:len(y_real)], y_real))
    print(f"✅ {name}: {len(y_real)} outputs mapped to {len(fields)} fields")

# نمایش 10 مقدار اول از هر بخش
for part, vals in outputs.items():
    print(f"\n---- {part} ----")
    for k, v in list(vals.items())[:10]:
        print(f"{k:8s}: {v:10.2f}")
