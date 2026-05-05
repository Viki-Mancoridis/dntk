import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torch.func import functional_call, grad, vmap
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WMDD_DIR     = os.environ.get("WMDD_DIR",     f"{os.environ['SCRATCH']}/dntk/wmdd")
DATA_DIR     = os.environ.get("DATA_DIR",     f"{WMDD_DIR}/imagenette_ipc50")
TEACHER_PATH = os.environ.get("TEACHER_PATH", f"{WMDD_DIR}/teacher_model/imagenette_resnet18.pth")
OUT_DIR      = os.environ.get("OUT_DIR",      f"{os.environ['SCRATCH']}/dntk/outputs")

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

N_PER_CLASS = 50      # paper's setting: 50 distilled images per class
K_PROJ      = 4096    # JL projection dimension
CHUNK       = 100_000 # max chunk size for the random-projection matrix
SEED        = 42
BATCH       = 32      # vmap batch size for per-sample gradients
EPS         = 0.05    # truncation-rank threshold per Definition 3.1


LABELS = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]

# ---------------------------------------------------------------------------
# Load distilled data
# ---------------------------------------------------------------------------
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tfm = transforms.Compose([
    transforms.Resize(176), transforms.CenterCrop(160),
    transforms.ToTensor(), NORM,
])
ds = ImageFolder(DATA_DIR, transform=tfm)

# Group sample indices by class (so we know which 50 samples belong to each class)
samples_by_class = defaultdict(list)
for idx, (_, y) in enumerate(ds.samples):
    samples_by_class[y].append(idx)

# Take the first 50 of each class (deterministic; their ordering is filename-sorted)
selected_per_class = {c: samples_by_class[c][:N_PER_CLASS] for c in range(10)}
all_selected = sum(selected_per_class.values(), [])
print(f"Loaded {len(all_selected)} distilled images "
      f"({N_PER_CLASS}/class x 10 classes).")

# ---------------------------------------------------------------------------
# Load teacher (handles both standard torchvision and WMDD's checkpoint formats)
# ---------------------------------------------------------------------------
print(f"Loading teacher from {TEACHER_PATH}")
ckpt = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
if isinstance(ckpt, dict):
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
else:
    sd = ckpt

# WMDD's checkpoint uses downsample.{conv,norm}.* instead of torchvision's
# downsample.{0,1}.* naming. Try both.
def load_with_remapping(model, sd):
    try:
        model.load_state_dict(sd)
        return
    except RuntimeError:
        pass
    remapped = {k.replace("module.", "").replace("model.", "")
                 .replace("downsample.conv.", "downsample.0.")
                 .replace("downsample.norm.", "downsample.1."): v
                for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    missing    = [k for k in missing    if "num_batches_tracked" not in k]
    unexpected = [k for k in unexpected if "num_batches_tracked" not in k]
    if missing or unexpected:
        raise RuntimeError(f"unmapped keys: missing={missing}, unexpected={unexpected}")

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 10)
load_with_remapping(model, sd)
model = model.to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)

# ---------------------------------------------------------------------------
# Per-sample gradients via torch.func + vmap
# ---------------------------------------------------------------------------
params  = {k: v.detach().clone() for k, v in model.named_parameters()}
buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
sorted_param_names = sorted(params.keys())

def logit_c(p, b, x, c):
    out = functional_call(model, (p, b), x.unsqueeze(0))
    return out[0, c]

batched_per_sample_grad = vmap(grad(logit_c), in_dims=(None, None, 0, None))

def jl_project_batched(grad_dict):
    """Standard Gaussian JL: g(u) = Q^T u / sqrt(k), Q ~ N(0, I).
    Q is drawn deterministically chunk-by-chunk so the same Q is used for every
    sample, which preserves inner products across samples."""
    B = next(iter(grad_dict.values())).shape[0]
    proj = torch.zeros(B, K_PROJ, device=device)
    chunk_id = 0
    for name in sorted_param_names:
        g = grad_dict[name].detach().reshape(B, -1)
        n = g.shape[1]
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            gen = torch.Generator(device=device)
            gen.manual_seed(SEED * 1_000_003 + chunk_id)
            Q = torch.randn(end - start, K_PROJ, device=device, generator=gen)
            proj.add_(g[:, start:end] @ Q)
            chunk_id += 1
            del Q
    return proj / math.sqrt(K_PROJ)

# ---------------------------------------------------------------------------
# Compute per-class kernels K^c on class-c data only (50 x 50 each)
# ---------------------------------------------------------------------------
print("Computing per-class kernels (50x50 each, K_PROJ=%d)..." % K_PROJ)
K_per_class = torch.zeros(10, N_PER_CLASS, N_PER_CLASS)

for c in range(10):
    indices = selected_per_class[c]
    xs = torch.stack([ds[i][0] for i in indices]).to(device)  # (50, 3, 160, 160)

    # Project the per-sample logit-c gradients of all 50 class-c samples
    phi_c = torch.zeros(N_PER_CLASS, K_PROJ, device=device)
    for start in tqdm(range(0, N_PER_CLASS, BATCH), desc=f"class {c} ({LABELS[c]})"):
        end = min(start + BATCH, N_PER_CLASS)
        x_batch = xs[start:end]
        g_dict = batched_per_sample_grad(params, buffers, x_batch, c)
        phi_c[start:end] = jl_project_batched(g_dict)
        del g_dict

    K_per_class[c] = (phi_c @ phi_c.T).cpu()
    del phi_c

# ---------------------------------------------------------------------------
# Save kernels and metadata
# ---------------------------------------------------------------------------
out_pt = f"{OUT_DIR}/per_class_kernels.pt"
torch.save({
    "K_per_class": K_per_class,
    "n_per_class": N_PER_CLASS,
    "k_proj": K_PROJ,
    "labels": LABELS,
    "teacher_path": TEACHER_PATH,
    "data_dir": DATA_DIR,
    "selected_per_class": selected_per_class,
}, out_pt)
print(f"Saved kernels: {out_pt}")

# ---------------------------------------------------------------------------
# Plot Figure 2
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
ranks = []
for c in range(10):
    K = K_per_class[c].numpy()
    eig = np.linalg.eigvalsh(K)[::-1]
    eig = np.maximum(eig, 1e-12)

    # Singular values of Phi^c[Ic] = sqrt(eigenvalues of K^c[Ic, Ic])
    sv = np.sqrt(eig)

    # Truncation rank, paper Definition 3.1
    cumvar = np.cumsum(eig) / eig.sum()
    r = int(np.searchsorted(cumvar, 1 - EPS) + 1)
    ranks.append(r)

    ax = axes.flat[c]
    ax.semilogy(np.arange(1, len(sv) + 1), sv, lw=1, color="k")
    ax.axvline(r, color="red", ls="--", lw=0.8)
    ax.text(0.96, 0.94, f"r_trunc = {r}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="red")
    ax.set_title(LABELS[c], fontsize=10)
    if c % 5 == 0:
        ax.set_ylabel(r"$\sigma_i$")
    if c >= 5:
        ax.set_xlabel("index $i$")

plt.suptitle(
    f"Per-class NTK spectra on WMDD-distilled ImageNette  "
    f"(per-class kernel size n={N_PER_CLASS}, ε={EPS})",
    fontsize=12,
)
plt.tight_layout()
out_png = f"{OUT_DIR}/figure2_reproduction.png"
out_pdf = f"{OUT_DIR}/figure2_reproduction.pdf"
plt.savefig(out_png, dpi=150)
plt.savefig(out_pdf)
print(f"Saved figure: {out_png}")

print()
print("Per-class truncation ranks:")
for label, r in zip(LABELS, ranks):
    print(f"  {label:<18s} r_trunc = {r}")
print(f"Range: [{min(ranks)}, {max(ranks)}]   mean: {np.mean(ranks):.1f}")
print(f"Paper's reported range: [31, 41]")