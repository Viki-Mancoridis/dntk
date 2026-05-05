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
WMDD_DIR  = os.environ.get("WMDD_DIR",  f"{os.environ['SCRATCH']}/dntk/wmdd")
TRAIN_DIR = os.environ.get("TRAIN_DIR", f"{WMDD_DIR}/imagenette_ipc50")
VAL_DIR   = os.environ.get("VAL_DIR",
                           f"{os.environ['SCRATCH']}/dntk/data/imagenette2-160/val")
OUT_DIR   = os.environ.get("OUT_DIR",   f"{os.environ['SCRATCH']}/dntk/outputs")

TEACHERS = [
    ("Pretrained (ours)",  f"{OUT_DIR}/resnet18_imagenette.pt",                 "C0"),
    ("Pretrained (WMDD)",  f"{WMDD_DIR}/teacher_model/imagenette_resnet18.pth", "C2"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN_PER_CLASS = 50
K_PROJ            = 4096
CHUNK             = 100_000
SEED              = 42
BATCH             = 32
LAMBDA_REG        = 1e-4

SWEEP_SIZES = [10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500]

LABELS = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]

# ---------------------------------------------------------------------------
# Datasets (loaded once, reused across teachers)
# ---------------------------------------------------------------------------
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tfm = transforms.Compose([
    transforms.Resize(176), transforms.CenterCrop(160),
    transforms.ToTensor(), NORM,
])
train_ds = ImageFolder(TRAIN_DIR, transform=tfm)
val_ds   = ImageFolder(VAL_DIR,   transform=tfm)

train_idx_by_class = defaultdict(list)
for idx, (_, y) in enumerate(train_ds.samples):
    train_idx_by_class[y].append(idx)
train_idx_by_class = {c: train_idx_by_class[c][:N_TRAIN_PER_CLASS] for c in range(10)}
train_indices = sum([train_idx_by_class[c] for c in range(10)], [])
test_indices  = list(range(len(val_ds)))
print(f"Training pool: {len(train_indices)} distilled images "
      f"({N_TRAIN_PER_CLASS}/class)")
print(f"Test set: {len(test_indices)} real ImageNette val images")
assert len(train_ds.classes) == len(val_ds.classes) == 10

# ---------------------------------------------------------------------------
# Helpers shared across teachers
# ---------------------------------------------------------------------------
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

def select_train_subset(n_per_class):
    sel = []
    for c in range(10):
        sel += list(range(c * N_TRAIN_PER_CLASS, c * N_TRAIN_PER_CLASS + n_per_class))
    return sel

# ---------------------------------------------------------------------------
# Per-teacher pipeline
# ---------------------------------------------------------------------------
def run_teacher(teacher_label, teacher_path):
    print()
    print("=" * 70)
    print(f"Running pipeline for teacher: {teacher_label}  ({teacher_path})")
    print("=" * 70)

    ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        sd = ckpt

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    load_with_remapping(model, sd)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    params  = {k: v.detach().clone() for k, v in model.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    sorted_param_names = sorted(params.keys())

    def logit_c(p, b, x, c):
        out = functional_call(model, (p, b), x.unsqueeze(0))
        return out[0, c]

    batched_per_sample_grad = vmap(grad(logit_c), in_dims=(None, None, 0, None))

    def jl_project_batched(grad_dict):
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

    def compute_phi_and_logits(dataset, indices, name,
                               _model=model, _params=params, _buffers=buffers,
                               _grad_fn=batched_per_sample_grad):
        n = len(indices)
        phi = torch.zeros(10, n, K_PROJ, device=device)
        logits = torch.zeros(n, 10, device=device)
        labels = torch.zeros(n, dtype=torch.long)
        xs = torch.zeros(n, 3, 160, 160, device=device)
        for i, idx in enumerate(indices):
            x, y = dataset[idx]
            xs[i] = x
            labels[i] = y
        with torch.no_grad():
            for start in range(0, n, 64):
                end = min(start + 64, n)
                logits[start:end] = _model(xs[start:end])
        for c in range(10):
            for start in tqdm(range(0, n, BATCH), desc=f"{name} logit {c}"):
                end = min(start + BATCH, n)
                g_dict = _grad_fn(_params, _buffers, xs[start:end], c)
                phi[c, start:end] = jl_project_batched(g_dict)
                del g_dict
        return phi.cpu(), logits.cpu(), labels

    print("Computing Phi_train, teacher_logits_train...")
    phi_train, y_logits_train, _ = compute_phi_and_logits(
        train_ds, train_indices, "train")
    print("Computing Phi_test, teacher_logits_test...")
    phi_test, y_logits_test, y_labels_test = compute_phi_and_logits(
        val_ds, test_indices, "test")

    del model, params, buffers
    torch.cuda.empty_cache()

    teacher_test_pred = y_logits_test.argmax(dim=1).numpy()
    teacher_test_acc  = (teacher_test_pred == y_labels_test.numpy()).mean()
    print(f"Teacher accuracy on test set: {teacher_test_acc:.4f}")

    # --- KRR sweep with trace-normalized per-class kernels
    results = {"n": [], "fidelity": [], "accuracy": [], "mse": [],
               "cond_number": [], "min_eig": []}
    n_test = phi_test.shape[1]

    for n_total in SWEEP_SIZES:
        n_per_class = n_total // 10
        sel = select_train_subset(n_per_class)
        krr_pred = torch.zeros(n_test, 10)
        cond_nums, min_eigs = [], []

        for c in range(10):
            Phi_tr = phi_train[c, sel, :]
            Phi_te = phi_test[c, :, :]
            K_tr = Phi_tr @ Phi_tr.T            # (n, n)
            K_te = Phi_tr @ Phi_te.T            # (n, n_test)

            n = K_tr.shape[0]
            scale = K_tr.diagonal().mean()
            K_tr_n = K_tr / scale
            K_te_n = K_te / scale

            y_c = y_logits_train[sel, c]
            I = torch.eye(n)
            try:
                alpha = torch.linalg.solve(K_tr_n + LAMBDA_REG * I, y_c)
            except torch._C._LinAlgError:
                alpha = torch.linalg.lstsq(K_tr_n + LAMBDA_REG * I, y_c).solution
            krr_pred[:, c] = K_te_n.T @ alpha

            eig = torch.linalg.eigvalsh(K_tr_n)
            eig = torch.clamp(eig, min=1e-12)
            cond_nums.append((eig.max() / eig.min()).item())
            min_eigs.append(eig.min().item())

        krr_test_pred = krr_pred.argmax(dim=1).numpy()
        fidelity = (krr_test_pred == teacher_test_pred).mean()
        accuracy = (krr_test_pred == y_labels_test.numpy()).mean()
        mse      = ((krr_pred - y_logits_test) ** 2).mean().item()

        results["n"].append(n_total)
        results["fidelity"].append(fidelity)
        results["accuracy"].append(accuracy)
        results["mse"].append(mse)
        results["cond_number"].append(float(np.mean(cond_nums)))
        results["min_eig"].append(float(np.mean(min_eigs)))

        print(f"  n={n_total:4d}  fid={fidelity:.4f}  acc={accuracy:.4f}  "
              f"mse={mse:.4f}  cond={np.mean(cond_nums):.2e}  "
              f"min_eig={np.mean(min_eigs):.2e}")

    return results, teacher_test_acc

# ---------------------------------------------------------------------------
# Run all teachers
# ---------------------------------------------------------------------------
all_results = {}
for label, path, color in TEACHERS:
    if not os.path.exists(path):
        print(f"WARNING: teacher checkpoint not found at {path}, skipping.")
        continue
    results, teacher_acc = run_teacher(label, path)
    all_results[label] = {
        "results": results,
        "teacher_acc": teacher_acc,
        "color": color,
        "path": path,
    }

# ---------------------------------------------------------------------------
# Save and plot
# ---------------------------------------------------------------------------
torch.save({
    "all_results": all_results,
    "lambda_reg": LAMBDA_REG,
    "k_proj": K_PROJ,
    "trace_normalized": True,
}, f"{OUT_DIR}/figure1_results.pt")
print(f"\nSaved combined results: {OUT_DIR}/figure1_results.pt")

fig, axes = plt.subplots(3, 2, figsize=(10, 10))

ax = axes[0, 0]
for label, d in all_results.items():
    ax.plot(d["results"]["n"], 100 * np.array(d["results"]["fidelity"]),
            "o-", color=d["color"], label=label)
ax.set_xlabel("Training set size")
ax.set_ylabel("Fidelity (%)")
ax.set_title("Test fidelity vs training set size")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.legend()

ax = axes[0, 1]
for label, d in all_results.items():
    ax.semilogy(d["results"]["n"], d["results"]["mse"],
                "o-", color=d["color"], label=label)
ax.set_xlabel("Training set size")
ax.set_ylabel("Mean squared error")
ax.set_title("Test MSE vs training set size")
ax.grid(alpha=0.3)
ax.legend()

ax = axes[1, 0]
for label, d in all_results.items():
    ax.plot(d["results"]["n"], 100 * np.array(d["results"]["accuracy"]),
            "o-", color=d["color"], label=label)
    ax.axhline(100 * d["teacher_acc"], color=d["color"], ls=":", lw=0.8,
               alpha=0.7,
               label=f"Original {label}: {d['teacher_acc']*100:.1f}%")
ax.set_xlabel("Training set size")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Test accuracy vs training set size")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

axes[1, 1].axis("off")

ax = axes[2, 0]
for label, d in all_results.items():
    ax.semilogy(d["results"]["n"], d["results"]["cond_number"],
                "o-", color=d["color"], label=label)
ax.set_xlabel("Training set size")
ax.set_ylabel("Condition number")
ax.set_title("Condition number vs training set size  (trace-normalized)")
ax.grid(alpha=0.3)
ax.legend()

ax = axes[2, 1]
for label, d in all_results.items():
    ax.semilogy(d["results"]["n"], d["results"]["min_eig"],
                "o-", color=d["color"], label=label)
ax.set_xlabel("Training set size")
ax.set_ylabel("Minimum eigenvalue")
ax.set_title("Min eigenvalue vs training set size  (trace-normalized)")
ax.grid(alpha=0.3)
ax.legend()

plt.suptitle(
    "Figure 1 reproduction (trace-normalized kernels): KRR on per-class NTK features, "
    "ResNet-18 / WMDD-distilled ImageNette",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figure1_reproduction.png", dpi=150)
plt.savefig(f"{OUT_DIR}/figure1_reproduction.pdf")
print(f"Saved figure: {OUT_DIR}/figure1_reproduction.png")